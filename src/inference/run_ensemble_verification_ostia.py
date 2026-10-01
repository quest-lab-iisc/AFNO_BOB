"""Verify AFNO ensemble SST forecast against OSTIA L4 analysis.

Computes ensemble verification metrics for sea-surface temperature (thetao)
only, using OSTIA as the independent truth instead of GLORYS reanalysis.

For each lead day, the SST ensemble members are:
  1. Interpolated from 224×224 to the GLORYS 229×265 grid.
  2. Denormalised (add climatological thetao mean).
  3. Compared against the OSTIA SST field (converted from K to °C) that has
     been pre-interpolated to the GLORYS grid.

Metrics computed per lead day:
  rmse          — RMSE of ensemble mean vs OSTIA SST
  mae           — MAE of ensemble mean vs OSTIA SST
  bias          — mean signed error (mean - OSTIA)
  corr          — Pearson correlation (ensemble mean vs OSTIA)
  spread        — mean ensemble std over ocean pixels (denormalised °C)
  spread_skill  — spread / rmse  (1 = perfectly calibrated)
  crps          — mean Continuous Ranked Probability Score over ocean pixels

Inputs:
    --ensemble_dir (str): Directory with ensemble_predictions.npy, ensemble_mean.npy,
        ensemble_std.npy, ensemble_metadata.json.
    --ostia_file (str): Path to ostia_2020.nc (default: data/1993_2020/ostia_2020.nc).
    --mean_dir (str): Directory with normalisation .npy files.
    --output_dir (str): Output directory (default: ensemble_dir/verification_ostia).

Outputs:
    {output_dir}/verification_metrics_ostia.csv — scalar metrics per lead day.

Example:
    conda activate BoB_Surf_2
    python src/inference/run_ensemble_verification_ostia.py \\
        --ensemble_dir results/AFNO_BoB_Surf_E14/ensemble_ic_era5_15-01-2020 \\
        --ostia_file data/1993_2020/ostia_2020.nc
"""

import sys
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent))

from inference.run_ensemble_verification import (
    load_norm_stats,
    interp_to_glorys,
    compute_crps,
    compute_metrics,
)

GLORYS_SHAPE    = (229, 265)
OSTIA_KELVIN_OFFSET = 273.15


def _precompute_ostia_on_glorys(ostia_file: Path,
                                 glorys_lat: np.ndarray,
                                 glorys_lon: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Load and interpolate OSTIA SST to the GLORYS grid.

    Args:
        ostia_file (Path): Path to ostia_2020.nc.
        glorys_lat (np.ndarray): GLORYS latitude values, shape (229,).
        glorys_lon (np.ndarray): GLORYS longitude values, shape (265,).

    Returns:
        tuple:
            np.ndarray: OSTIA SST in °C on GLORYS grid, shape (366, 229, 265).
            list[str]: ISO date strings ('YYYY-MM-DD') for each time step.

    Example:
        >>> sst, dates = _precompute_ostia_on_glorys(f, lat, lon)
        >>> sst.shape
        (366, 229, 265)
    """
    ds   = xr.open_dataset(ostia_file)
    sst  = (ds['analysed_sst'] - OSTIA_KELVIN_OFFSET).interp(
        latitude=glorys_lat, longitude=glorys_lon, method='linear'
    )
    dates = [str(t)[:10] for t in sst.time.values]
    arr   = sst.values.astype(np.float32)   # (366, 229, 265)
    ds.close()
    return arr, dates


def main():
    """Load ensemble outputs and OSTIA truth; compute and save verification metrics.

    Args:
        None: All settings read from sys.argv (see module docstring).

    Returns:
        None: Writes verification_metrics_ostia.csv as a side effect.

    Example:
        >>> # python src/inference/run_ensemble_verification_ostia.py \\
        >>> #     --ensemble_dir results/AFNO_BoB_Surf_E14/ensemble_ic_era5_15-01-2020
    """
    parser = argparse.ArgumentParser(
        description='Ensemble SST verification against OSTIA L4 analysis'
    )
    parser.add_argument('--ensemble_dir', required=True)
    parser.add_argument('--ostia_file',  default='data/1993_2020/ostia_2020.nc')
    parser.add_argument('--mean_dir',    default='data/1993_2020/mean')
    parser.add_argument('--output_dir',  default=None)
    args = parser.parse_args()

    ens_dir = Path(args.ensemble_dir)
    out_dir = Path(args.output_dir) if args.output_dir else ens_dir / 'verification_ostia'
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Normalisation stats (for denormalising SST) ---
    mean_dict  = load_norm_stats(Path(args.mean_dir))
    thetao_mean = np.squeeze(mean_dict['thetao'])       # (229, 265)
    land_mask   = np.isnan(thetao_mean)                 # True = land
    ocean_mask  = ~land_mask

    # --- GLORYS grid coordinates from mean file shape ---
    glorys_lat = np.linspace(4.0, 23.0, GLORYS_SHAPE[0])
    glorys_lon = np.linspace(77.0, 99.0, GLORYS_SHAPE[1])

    # --- Pre-interpolate OSTIA to GLORYS grid ---
    print(f"Loading OSTIA from {args.ostia_file} ...", flush=True)
    ostia_sst, ostia_dates = _precompute_ostia_on_glorys(
        Path(args.ostia_file), glorys_lat, glorys_lon
    )
    ostia_date_idx = {d: i for i, d in enumerate(ostia_dates)}

    # --- Load ensemble outputs ---
    meta      = json.load(open(ens_dir / 'ensemble_metadata.json'))
    init_date = meta['input_date']
    lead_h    = meta['lead_times_h']
    n_steps   = len(lead_h)
    init_dt   = datetime.strptime(init_date, '%d-%m-%Y')

    print(f"Init date : {init_date}  ({n_steps} lead steps)")
    print(f"Output dir: {out_dir}", flush=True)

    ens_mean  = np.load(ens_dir / 'ensemble_mean.npy')        # (n_steps, 5, 224, 224)
    ens_std   = np.load(ens_dir / 'ensemble_std.npy')         # (n_steps, 5, 224, 224)
    ens_preds = np.load(ens_dir / 'ensemble_predictions.npy') # (50, n_steps, 5, 224, 224)
    n_members = ens_preds.shape[0]
    print(f"Members   : {n_members}  |  Preds shape: {ens_preds.shape}")

    # --- CSV ---
    csv_path = out_dir / 'verification_metrics_ostia.csv'
    csv_cols = ['init_date', 'lead_day', 'variable',
                'rmse', 'mae', 'bias', 'corr', 'spread', 'spread_skill', 'crps']
    csv_file = open(csv_path, 'w', newline='')
    writer   = csv.DictWriter(csv_file, fieldnames=csv_cols)
    writer.writeheader()

    # --- Loop over lead steps (SST / thetao only, channel index 0) ---
    print(f"\nVerifying SST at {n_steps} lead steps against OSTIA ...\n")

    for s in range(n_steps):
        lead_day = s + 1
        valid_dt = init_dt + timedelta(hours=int(lead_h[s]))
        date_key = valid_dt.strftime('%Y-%m-%d')

        if date_key not in ostia_date_idx:
            print(f"  Lead +{lead_day}d ({date_key}) — no OSTIA data, skipping")
            continue

        truth_sst = ostia_sst[ostia_date_idx[date_key]]   # (229, 265) °C
        print(f"  Lead +{lead_day}d  ({date_key})", flush=True)

        # Ensemble mean SST denormalised
        pred_norm   = ens_mean[s, 0]                      # (224, 224)
        pred_glorys = interp_to_glorys(pred_norm)         # (229, 265)
        pred_sst    = pred_glorys + thetao_mean           # °C
        pred_sst[land_mask] = np.nan

        # Ensemble members SST denormalised
        members_norm   = ens_preds[:, s, 0]               # (50, 224, 224)
        members_glorys = np.stack(
            [interp_to_glorys(members_norm[m]) for m in range(n_members)]
        )                                                  # (50, 229, 265)
        members_sst = members_glorys + thetao_mean[np.newaxis]  # (50, 229, 265) °C

        # Scalar metrics
        m = compute_metrics(pred_sst, truth_sst, ocean_mask)

        # Ensemble spread (in physical °C space)
        std_glorys = interp_to_glorys(ens_std[s, 0])      # (229, 265)
        spread     = float(std_glorys[ocean_mask].mean())
        spread_skill = spread / m['rmse'] if (m['rmse'] and m['rmse'] > 0) else np.nan

        # CRPS
        crps = compute_crps(members_sst, truth_sst, ocean_mask)

        writer.writerow({
            'init_date'   : init_date,
            'lead_day'    : lead_day,
            'variable'    : 'thetao',
            'rmse'        : round(m['rmse'],  4),
            'mae'         : round(m['mae'],   4),
            'bias'        : round(m['bias'],  4),
            'corr'        : round(m['corr'],  4),
            'spread'      : round(spread,     4),
            'spread_skill': round(spread_skill, 4) if not np.isnan(spread_skill) else '',
            'crps'        : round(crps,       4),
        })

        print(f"    RMSE={m['rmse']:.4f}  Spread={spread:.4f}  "
              f"SS={spread_skill:.3f}  CRPS={crps:.4f}  r={m['corr']:.3f}")

    csv_file.close()
    print(f"\nSaved: {csv_path}")


if __name__ == '__main__':
    main()

"""Compute per-pixel autoregressive bias correction fields from ERA5 forecasts.

Runs the ocean AFNO model autoregressively for every sampled initialisation
date in a given year (typically the 2019 validation year), compares each
9-day-ahead prediction against GLORYS reanalysis, and accumulates the mean
signed error (prediction − truth) per variable, per lead day, and per grid
pixel.

The resulting bias fields can be subtracted from future ensemble forecasts
via apply_bias_correction.py to partially correct the systematic warm drift
that accumulates in autoregressive rollout.

Inputs:
    --config_file (str): YAML config filename in the config/ directory.
    --year (int): Calendar year to sweep for initialisation dates (default: 2019).
    --step_days (int): Sample every N-th calendar day (default: 7 for speed;
        use 1 for the full 365-date sweep).
    --output_dir (str): Directory to write bias .npy files
        (default: data/bias_correction).
    --mean_dir (str): Directory with normalisation stat .npy files
        (default: data/1993_2020/mean).
    --model_path (str): Override model weights path
        (default: results/models/{config.name}.pth).
    --device (str): Torch device string (default: from config).

Outputs:
    data/bias_correction/
        bias_2019_all.npy               — float32 shape (9, 5, 229, 265),
            mean signed error (pred − truth) averaged over all init dates.
        bias_2019_{var}_lead{d:02d}.npy — individual (229, 265) file per
            variable per lead day for convenient per-step loading.
        bias_metadata.json              — sweep parameters and date count.

Example:
    conda activate BoB_Surf_2
    python src/inference/run_bias_sweep.py \\
        --config_file afno_bob_surf_e00.yaml \\
        --year 2019 \\
        --step_days 7 \\
        --output_dir data/bias_correction \\
        --device cuda:0
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from configmypy import ConfigPipeline, YamlConfig
from models.architectures.afno.afnonet import AFNONet
from data_pipeline.preprocessing.transformers.normalize_transform import PreprocessTransform
from inference.run_inference import load_normalization_stats
from inference.run_ensemble_verification import (
    load_norm_stats,
    interp_to_glorys,
    denormalize,
    load_glorys_truth,
)


OCEAN_VARS  = ['thetao', 'so', 'uo', 'vo', 'zos']
ATM_VARS    = ['ssr', 'tp', 'u10', 'v10', 'msl', 'tcc']
TARGET_SIZE = (229, 265)
N_LEAD      = 9
REF_DATE    = datetime(1993, 1, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(config, model_path: Path, device: torch.device) -> torch.nn.Module:
    """Load AFNO model weights from a .pth checkpoint.

    Args:
        config: Configuration object with architecture settings.
        model_path (Path): Path to the .pth weights file.
        device (torch.device): Target device.

    Returns:
        torch.nn.Module: Model in eval mode on the specified device.

    Example:
        >>> model = _load_model(config, Path('results/models/AFNO_BoB_Surf_E00.pth'), device)
    """
    model = AFNONet(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def _load_atm_from_ds(atm_ds, day_index: int, transform: PreprocessTransform,
                      mean_dict: dict, variance_dict: dict, config) -> torch.Tensor:
    """Load and normalise one day of ERA5 atmospheric forcing from a pre-opened Dataset.

    Replicates the preprocessing applied inside load_atmospheric_forcing() in
    run_inference.py but reads from an already-open xarray Dataset to avoid
    repeated file opens across a multi-date sweep.

    Args:
        atm_ds: Open xarray Dataset (atm.nc).
        day_index (int): Time index to load.
        transform (PreprocessTransform): Normalisation + interpolation transform.
        mean_dict (dict): Climatological means keyed by variable name.
        variance_dict (dict): Climatological stds keyed by '{var}_std'.
        config: Configuration object.

    Returns:
        torch.Tensor: Shape (6, 224, 224) — normalised atmospheric forcing.

    Example:
        >>> atm = _load_atm_from_ds(atm_ds, 9496, transform, mean_dict, variance_dict, config)
        >>> atm.shape
        torch.Size([6, 224, 224])
    """
    atm_inputs = []
    for var in ATM_VARS:
        raw = atm_ds[var][day_index:day_index + 1].values   # (1, H, W)
        if var in ('ssr', 'tp', 'msl'):
            t = transform(raw, mean_dict[var], variable=var, type='atm',
                          variance=variance_dict[f'{var}_std'])
        else:
            t = transform(raw, mean_dict[var], variable=var, type='atm')
        atm_inputs.append(t.squeeze(1))   # (1, 224, 224)
    return torch.cat(atm_inputs, dim=0)   # (6, 224, 224)


def _load_ocean_init(ocean_ds, day_index: int, transform: PreprocessTransform,
                     mean_dict: dict, north_mask_rows: int) -> dict:
    """Load and normalise initial ocean state from a pre-opened xarray Dataset.

    Applies the same NaN fill, mean subtraction, and 224×224 bilinear
    interpolation as load_initial_ocean_state() in run_inference.py.

    Args:
        ocean_ds: Open xarray Dataset (ocean.nc).
        day_index (int): Time index for the initial state (time t).
        transform (PreprocessTransform): Normalisation + interpolation transform.
        mean_dict (dict): Climatological means keyed by variable name.
        north_mask_rows (int): Number of northernmost rows to zero (from config.data.north_mask_rows).

    Returns:
        dict: Variable name → torch.Tensor of shape (1, 224, 224).

    Example:
        >>> ocean = _load_ocean_init(ocean_ds, 9496, transform, mean_dict, north_mask_rows=20)
        >>> ocean['thetao'].shape
        torch.Size([1, 224, 224])
    """
    ocean_state = {}
    for var in OCEAN_VARS:
        raw = ocean_ds[var][day_index:day_index + 1].values   # (1, [depth,] H, W)
        t = transform(raw, mean_dict[var], variable=var)
        if north_mask_rows > 0:
            t[:, :, -north_mask_rows:, :] = 0.0
        ocean_state[var] = t.squeeze(1)   # (1, 224, 224)
    return ocean_state


def _run_forecast(model: torch.nn.Module, ocean_state: dict, atm_ds,
                  day_index: int, transform: PreprocessTransform,
                  mean_dict: dict, variance_dict: dict,
                  config, device: torch.device) -> np.ndarray:
    """Run 9-step autoregressive ocean forecast using ERA5 atmospheric forcing.

    Args:
        model (torch.nn.Module): Trained AFNO model in eval mode.
        ocean_state (dict): Initial ocean state; variable → (1, 224, 224) tensor.
        atm_ds: Open xarray Dataset for atm.nc.
        day_index (int): Time index of the initial ocean state (t=0).
        transform (PreprocessTransform): Preprocessing transform.
        mean_dict (dict): Climatological means.
        variance_dict (dict): Climatological stds.
        config: Configuration object.
        device (torch.device): Torch device.

    Returns:
        np.ndarray: Shape (9, 5, 224, 224) float32 — normalised predictions
            at lead days +1 through +9.

    Example:
        >>> preds = _run_forecast(model, ocean_state, atm_ds, 9496,
        ...                       transform, mean_dict, variance_dict, config, device)
        >>> preds.shape
        (9, 5, 224, 224)
    """
    preds = np.zeros((N_LEAD, 5, 224, 224), dtype=np.float32)
    current = {v: s.clone() for v, s in ocean_state.items()}

    with torch.no_grad():
        for lead in range(N_LEAD):
            atm = _load_atm_from_ds(
                atm_ds, day_index + lead + 1,
                transform, mean_dict, variance_dict, config,
            )
            ocean_t = torch.cat([current[v] for v in OCEAN_VARS], dim=0)   # (5, 224, 224)
            inp = torch.cat([atm, ocean_t], dim=0).unsqueeze(0).to(device) # (1, 11, 224, 224)
            out = model(inp).squeeze(0).cpu()                               # (5, 224, 224)
            preds[lead] = out.numpy()
            for vi, var in enumerate(OCEAN_VARS):
                current[var] = out[vi:vi + 1]   # (1, 224, 224) — update ocean state

    return preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Sweep ERA5-driven forecasts over a full year and accumulate per-pixel bias fields.

    Args:
        None: All settings are read from sys.argv (see module docstring).

    Returns:
        None: Writes bias .npy files and bias_metadata.json to --output_dir.

    Example:
        >>> # python src/inference/run_bias_sweep.py \\
        >>> #     --config_file afno_bob_surf_e00.yaml --year 2019 --step_days 7
    """
    parser = argparse.ArgumentParser(
        description='Accumulate per-pixel forecast bias over a calendar year'
    )
    parser.add_argument('--config_file', default='afno_bob_surf_e00.yaml')
    parser.add_argument('--year', type=int, default=2019,
                        help='Initialisation year (default 2019 = validation year)')
    parser.add_argument('--step_days', type=int, default=7,
                        help='Calendar-day stride for sampling (default 7)')
    parser.add_argument('--output_dir', default='data/bias_correction')
    parser.add_argument('--mean_dir', default='data/1993_2020/mean')
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Config ---
    pipe = ConfigPipeline([
        YamlConfig(args.config_file, config_name='default', config_folder='config/')
    ])
    config = pipe.read_conf()
    device = torch.device(args.device or config.device)
    print(f"Config   : {args.config_file}  |  Device: {device}")

    # --- Model ---
    model_path = Path(args.model_path or f"results/models/{config.name}.pth")
    model = _load_model(config, model_path, device)
    print(f"Model    : {model_path}")

    # --- Normalisation stats ---
    mean_dict, variance_dict = load_normalization_stats(config)
    transform = PreprocessTransform(config)
    glorys_mean = load_norm_stats(Path(args.mean_dir))
    land_mask   = np.isnan(np.squeeze(glorys_mean['thetao']))   # (229, 265)

    # --- Open ERA5 + GLORYS datasets once ---
    data_dir   = Path(config.data.data_dir)
    ocean_file = data_dir / f'{config.data.file_prefix}.nc'
    atm_file   = data_dir / f'{config.data.file_prefix_atm}.nc'
    atm_ds     = xr.open_dataset(atm_file)
    ocean_ds   = xr.open_dataset(ocean_file)
    print(f"Ocean    : {ocean_file}")
    print(f"Atm      : {atm_file}")

    # --- Build initialisation-date list ---
    start = datetime(args.year, 1, 1)
    end   = datetime(args.year, 12, 22)   # need 9 lead days → stop at Dec 22
    dates, d = [], start
    while d <= end:
        dates.append(d)
        d += timedelta(days=args.step_days)
    print(f"Dates    : {len(dates)}  "
          f"({start.strftime('%d-%m-%Y')} → {dates[-1].strftime('%d-%m-%Y')}, "
          f"step={args.step_days}d)\n")

    # --- Accumulation arrays: shape (N_LEAD, 5, 229, 265) ---
    bias_sum    = np.zeros((N_LEAD, 5, *TARGET_SIZE), dtype=np.float64)
    count_valid = np.zeros((N_LEAD, 5, *TARGET_SIZE), dtype=np.int32)
    n_ok = 0

    for dt in tqdm(dates, desc='Bias sweep', unit='date'):
        day_index = (dt - REF_DATE).days
        try:
            ocean_state = _load_ocean_init(ocean_ds, day_index, transform, mean_dict,
                                           north_mask_rows=config.data.north_mask_rows)
            preds = _run_forecast(
                model, ocean_state, atm_ds, day_index,
                transform, mean_dict, variance_dict, config, device,
            )                                  # (9, 5, 224, 224) normalised

            for lead in range(N_LEAD):
                truth = load_glorys_truth(ocean_file, day_index + lead + 1)
                for vi, var in enumerate(OCEAN_VARS):
                    pred_g = interp_to_glorys(preds[lead, vi])       # 224 → 229×265
                    pred_p = denormalize(pred_g, var, glorys_mean)    # physical units
                    pred_p[land_mask] = np.nan
                    t     = truth[var]
                    valid = ~np.isnan(pred_p) & ~np.isnan(t)
                    bias_sum[lead, vi][valid]    += (pred_p - t)[valid].astype(np.float64)
                    count_valid[lead, vi][valid] += 1
            n_ok += 1

        except Exception as exc:
            tqdm.write(f"  Skip {dt.strftime('%d-%m-%Y')}: {exc}")

    atm_ds.close()
    ocean_ds.close()

    # --- Compute mean bias ---
    bias_mean = np.where(
        count_valid > 0,
        bias_sum / np.maximum(count_valid, 1),
        np.nan,
    ).astype(np.float32)

    # --- Save outputs ---
    np.save(out_dir / 'bias_2019_all.npy', bias_mean)
    for lead in range(N_LEAD):
        for vi, var in enumerate(OCEAN_VARS):
            np.save(out_dir / f'bias_2019_{var}_lead{lead + 1:02d}.npy',
                    bias_mean[lead, vi])

    meta = {
        'year'                    : args.year,
        'step_days'               : args.step_days,
        'n_init_dates_attempted'  : len(dates),
        'n_init_dates_succeeded'  : n_ok,
        'n_lead_days'             : N_LEAD,
        'variables'               : OCEAN_VARS,
        'config_file'             : args.config_file,
        'model_path'              : str(model_path),
        'shape'                   : list(bias_mean.shape),
        'grid'                    : 'GLORYS 229x265',
    }
    with open(out_dir / 'bias_metadata.json', 'w') as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nSaved: {out_dir}/bias_2019_all.npy  shape={bias_mean.shape}")
    print(f"       {N_LEAD} × {len(OCEAN_VARS)} per-variable files")
    print(f"       {out_dir}/bias_metadata.json  (n_ok={n_ok}/{len(dates)})")

    # --- Summary table ---
    print(f"\nDomain-averaged mean bias summary:")
    print(f"  {'Variable':8s}  {'Lead':5s}  {'Mean bias':>12s}  {'Max count/px':>14s}")
    for lead in range(N_LEAD):
        for vi, var in enumerate(OCEAN_VARS):
            b   = bias_mean[lead, vi]
            cnt = int(count_valid[lead, vi].max())
            print(f"  {var:8s}  +{lead+1}d  {np.nanmean(b):+12.4f}  {cnt:>14d}")


if __name__ == '__main__':
    main()

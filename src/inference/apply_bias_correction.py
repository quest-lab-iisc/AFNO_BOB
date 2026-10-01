"""Apply pre-computed bias correction fields to an ensemble forecast mean.

Loads the normalised ensemble mean (224×224, normalised space) from a completed
ensemble run, denormalises it to physical units on the GLORYS 229×265 grid, and
subtracts the mean per-pixel bias produced by run_bias_sweep.py.  The corrected
mean is saved alongside optional updated verification statistics.

Inputs:
    --ensemble_dir (str): Directory containing ensemble_mean.npy and
        ensemble_metadata.json (output of run_ensemble_inference.py).
    --bias_dir (str): Directory containing bias_2019_all.npy and
        bias_metadata.json (output of run_bias_sweep.py).
    --config_file (str): YAML config filename for data_dir / ocean.nc path.
    --mean_dir (str): Directory with normalisation mean .npy files
        (default: data/1993_2020/mean).
    --output_dir (str): Where to write corrected files
        (default: ensemble_dir).

Outputs:
    ensemble_mean_corrected.npy     — float32 shape (9, 5, 229, 265),
        physical units, land pixels NaN.
    bias_correction_applied.json    — metadata about the correction run.

Example:
    conda activate BoB_Surf_2
    python src/inference/apply_bias_correction.py \\
        --ensemble_dir results/AFNO_BoB_Surf_E00/ensemble_Jan_14012020 \\
        --bias_dir data/bias_correction \\
        --config_file afno_bob_surf_e00.yaml
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from configmypy import ConfigPipeline, YamlConfig
from inference.run_ensemble_verification import (
    load_norm_stats,
    interp_to_glorys,
    denormalize,
    load_glorys_truth,
    compute_metrics,
)
from inference.utils import date_to_day_index

OCEAN_VARS = ['thetao', 'so', 'uo', 'vo', 'zos']


def main():
    """Load ensemble mean and bias fields, subtract bias, and report updated metrics.

    Args:
        None: All settings read from sys.argv (see module docstring).

    Returns:
        None: Writes ensemble_mean_corrected.npy and bias_correction_applied.json.

    Example:
        >>> # python src/inference/apply_bias_correction.py \\
        >>> #     --ensemble_dir results/AFNO_BoB_Surf_E00/ensemble_Jan_14012020 \\
        >>> #     --bias_dir data/bias_correction \\
        >>> #     --config_file afno_bob_surf_e00.yaml
    """
    parser = argparse.ArgumentParser(
        description='Subtract pre-computed bias from ensemble forecast mean'
    )
    parser.add_argument('--ensemble_dir', required=True,
                        help='Directory with ensemble_mean.npy + ensemble_metadata.json')
    parser.add_argument('--bias_dir', default='data/bias_correction',
                        help='Directory with bias_2019_all.npy')
    parser.add_argument('--config_file', default='afno_bob_surf_e00.yaml',
                        help='YAML config filename for ocean.nc path')
    parser.add_argument('--mean_dir', default='data/1993_2020/mean',
                        help='Directory with normalisation mean .npy files')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory (default: ensemble_dir)')
    args = parser.parse_args()

    ens_dir  = Path(args.ensemble_dir)
    out_dir  = Path(args.output_dir) if args.output_dir else ens_dir
    bias_dir = Path(args.bias_dir)

    # --- Config (ocean.nc path) ---
    pipe = ConfigPipeline([
        YamlConfig(args.config_file, config_name='default', config_folder='config/')
    ])
    config    = pipe.read_conf()
    ocean_file = Path(config.data.data_dir) / f'{config.data.file_prefix}.nc'

    # --- Normalisation stats ---
    mean_dict  = load_norm_stats(Path(args.mean_dir))
    land_mask  = np.isnan(np.squeeze(mean_dict['thetao']))   # (229, 265)
    ocean_mask = ~land_mask

    # --- Load ensemble mean ---
    ens_mean = np.load(ens_dir / 'ensemble_mean.npy')        # (9, 5, 224, 224) normalised
    meta     = json.load(open(ens_dir / 'ensemble_metadata.json'))
    init_date     = meta['input_date']
    lead_times_h  = meta['lead_times_h']
    n_steps  = ens_mean.shape[0]
    print(f"Ensemble  : {ens_dir.name}  |  init={init_date}  |  steps={n_steps}")

    # --- Load bias field ---
    bias_path = bias_dir / 'bias_2019_all.npy'
    if not bias_path.exists():
        raise FileNotFoundError(
            f"Bias file not found: {bias_path}\n"
            "Run run_bias_sweep.py first."
        )
    bias = np.load(bias_path)   # (9, 5, 229, 265)
    print(f"Bias      : {bias_path}  shape={bias.shape}")

    bias_meta_path = bias_dir / 'bias_metadata.json'
    if bias_meta_path.exists():
        bm = json.load(open(bias_meta_path))
        print(f"            derived from year={bm.get('year')}  "
              f"n_dates={bm.get('n_init_dates_succeeded')}/{bm.get('n_init_dates_attempted')}")

    # --- Denormalise and correct ---
    corrected   = np.full((n_steps, 5, 229, 265), np.nan, dtype=np.float32)
    uncorrected = np.full_like(corrected, np.nan)

    for s in range(n_steps):
        for vi, var in enumerate(OCEAN_VARS):
            pred_g = interp_to_glorys(ens_mean[s, vi])           # 224×224 → 229×265
            pred_p = denormalize(pred_g, var, mean_dict)          # add climatological mean
            pred_p[land_mask] = np.nan
            uncorrected[s, vi] = pred_p

            b = bias[s, vi]   # (229, 265) — NaN where no training data
            corr_field = pred_p - np.where(np.isnan(b), 0.0, b)  # leave NaN-bias pixels unchanged
            corr_field[land_mask] = np.nan
            corrected[s, vi] = corr_field

    # --- Save corrected mean ---
    out_path = out_dir / 'ensemble_mean_corrected.npy'
    np.save(out_path, corrected)
    print(f"\nSaved     : {out_path}  shape={corrected.shape}")

    # --- Optional verification: compare before/after against GLORYS truth ---
    init_dt = datetime.strptime(init_date, '%d-%m-%Y')
    print(f"\nVerification (before → after bias correction):")
    hdr = (f"  {'Variable':8s}  {'Lead':5s}  "
           f"{'RMSE before':>12s}  {'RMSE after':>10s}  "
           f"{'Bias before':>12s}  {'Bias after':>10s}")
    print(hdr)
    try:
        for s in range(n_steps):
            lead_day = s + 1
            valid_dt = init_dt + timedelta(hours=int(lead_times_h[s]))
            day_index = date_to_day_index(valid_dt.strftime('%d-%m-%Y'), '01-01-1993')
            try:
                truth = load_glorys_truth(ocean_file, day_index)
            except Exception as exc:
                print(f"  (truth unavailable for +{lead_day}d: {exc})")
                continue
            for vi, var in enumerate(OCEAN_VARS):
                m_before = compute_metrics(uncorrected[s, vi], truth[var], ocean_mask)
                m_after  = compute_metrics(corrected[s, vi],   truth[var], ocean_mask)
                print(f"  {var:8s}  +{lead_day}d  "
                      f"  {m_before['rmse']:12.4f}  {m_after['rmse']:10.4f}  "
                      f"  {m_before['bias']:+12.4f}  {m_after['bias']:+10.4f}")
    except Exception as exc:
        print(f"  (verification skipped: {exc})")

    # --- Write metadata ---
    corr_meta = {
        'ensemble_dir'       : str(ens_dir),
        'bias_dir'           : str(bias_dir),
        'init_date'          : init_date,
        'correction_applied' : True,
        'output_shape'       : list(corrected.shape),
        'units'              : 'physical (°C / psu / m s⁻¹ / m)',
    }
    with open(out_dir / 'bias_correction_applied.json', 'w') as fh:
        json.dump(corr_meta, fh, indent=2)


if __name__ == '__main__':
    main()

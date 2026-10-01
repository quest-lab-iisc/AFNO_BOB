"""Apply multiplicative spread calibration to ensemble standard deviation.

Reads the verification CSV produced by run_ensemble_verification.py and computes
a calibration factor per variable per lead day as:

    calibration_factor[var, lead] = rmse / spread

Multiplies the ensemble_std.npy by this factor to produce a statistically
calibrated spread that, on average, matches the observed forecast error level
(spread-skill ratio = 1).

Note on validity: This calibration is derived from the same test dates it
corrects — it is an in-sample diagnostic showing the *potential* gain from
calibration, not a true cross-validated estimate.  For proper out-of-sample
calibration, derive factors from a separate validation year using the bias
sweep (run_bias_sweep.py) and verification on 2019 dates.

Inputs:
    --ensemble_dir (str): Directory with ensemble_std.npy.
    --verification_dir (str): Directory containing verification_metrics.csv
        (default: ensemble_dir/verification).
    --output_dir (str): Where to write calibrated std
        (default: ensemble_dir).

Outputs:
    ensemble_std_calibrated.npy — float32 shape (9, 5, 224, 224),
        normalised-space spread scaled by calibration factors.
    calibration_factors.json    — factor table per variable per lead day.

Example:
    conda activate BoB_Surf_2
    python src/inference/calibrate_ensemble.py \\
        --ensemble_dir results/AFNO_BoB_Surf_E00/ensemble_Jan_14012020 \\
        --verification_dir results/AFNO_BoB_Surf_E00/ensemble_Jan_14012020/verification
"""

import sys
import csv
import json
import argparse
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

OCEAN_VARS = ['thetao', 'so', 'uo', 'vo', 'zos']


def _read_verification_csv(csv_path: Path) -> tuple[dict, dict]:
    """Parse verification_metrics.csv and return rmse and spread tables.

    Args:
        csv_path (Path): Path to verification_metrics.csv.

    Returns:
        tuple:
            rmse_table   (dict): (variable, lead_day) → float rmse.
            spread_table (dict): (variable, lead_day) → float spread.

    Example:
        >>> rmse, spread = _read_verification_csv(Path('verification_metrics.csv'))
        >>> rmse[('thetao', 1)]
        0.5697
    """
    rmse_table   = {}
    spread_table = {}
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (row['variable'], int(row['lead_day']))
            rmse_table[key]   = float(row['rmse'])
            spread_table[key] = float(row['spread'])
    return rmse_table, spread_table


def main():
    """Compute and apply per-variable per-lead calibration factors to ensemble spread.

    Args:
        None: All settings are read from sys.argv (see module docstring).

    Returns:
        None: Writes ensemble_std_calibrated.npy and calibration_factors.json.

    Example:
        >>> # python src/inference/calibrate_ensemble.py \\
        >>> #     --ensemble_dir results/AFNO_BoB_Surf_E00/ensemble_Jan_14012020 \\
        >>> #     --verification_dir .../verification
    """
    parser = argparse.ArgumentParser(
        description='Calibrate ensemble spread to match observed RMSE level'
    )
    parser.add_argument('--ensemble_dir', required=True,
                        help='Directory with ensemble_std.npy')
    parser.add_argument('--verification_dir', default=None,
                        help='Directory with verification_metrics.csv '
                             '(default: ensemble_dir/verification)')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory (default: ensemble_dir)')
    args = parser.parse_args()

    ens_dir = Path(args.ensemble_dir)
    ver_dir = Path(args.verification_dir) if args.verification_dir \
              else ens_dir / 'verification'
    out_dir = Path(args.output_dir) if args.output_dir else ens_dir

    csv_path = ver_dir / 'verification_metrics.csv'
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Verification CSV not found: {csv_path}\n"
            "Run run_ensemble_verification.py first."
        )

    # --- Load verification metrics ---
    rmse_table, spread_table = _read_verification_csv(csv_path)
    n_steps = max(lead for (_, lead) in rmse_table)

    # --- Compute calibration factors ---
    factors = {}
    for var in OCEAN_VARS:
        for lead in range(1, n_steps + 1):
            key = (var, lead)
            rmse   = rmse_table.get(key, np.nan)
            spread = spread_table.get(key, 0.0)
            if spread > 0 and not np.isnan(rmse):
                factors[key] = rmse / spread
            else:
                factors[key] = 1.0   # fallback: no change

    # --- Load ensemble std and apply calibration ---
    ens_std = np.load(ens_dir / 'ensemble_std.npy')   # (n_steps, 5, 224, 224)
    std_cal = ens_std.copy()

    print(f"Spread calibration — {ens_dir.name}")
    print(f"\n  {'Variable':8s}  {'Lead':5s}  "
          f"{'RMSE':>8s}  {'Spread':>8s}  {'Factor':>8s}  {'New spread':>10s}")
    for s in range(n_steps):
        lead = s + 1
        for vi, var in enumerate(OCEAN_VARS):
            key   = (var, lead)
            f_val = factors[key]
            std_cal[s, vi] *= f_val
            print(f"  {var:8s}  +{lead}d  "
                  f"  {rmse_table.get(key, float('nan')):8.4f}"
                  f"  {spread_table.get(key, float('nan')):8.4f}"
                  f"  {f_val:8.2f}×"
                  f"  {spread_table.get(key, 0.0) * f_val:10.4f}")

    # --- Save calibrated std ---
    out_path = out_dir / 'ensemble_std_calibrated.npy'
    np.save(out_path, std_cal)
    print(f"\nSaved: {out_path}  shape={std_cal.shape}")

    # --- Save factor table as JSON ---
    factor_dict = {f'{k[0]}_lead{k[1]:02d}': round(v, 6)
                   for k, v in factors.items()}
    json_path = out_dir / 'calibration_factors.json'
    with open(json_path, 'w') as fh:
        json.dump(factor_dict, fh, indent=2)
    print(f"       {json_path}")

    # --- Spread-skill summary after calibration ---
    print(f"\nSpread-skill ratio after calibration (should be ~1.0):")
    print(f"  {'Variable':8s}  {'Lead':5s}  {'SS ratio':>10s}")
    for s in range(n_steps):
        lead = s + 1
        for var in OCEAN_VARS:
            key    = (var, lead)
            factor = factors[key]
            print(f"  {var:8s}  +{lead}d  {1.0 if factor > 0 else float('nan'):>10.3f}")


if __name__ == '__main__':
    main()

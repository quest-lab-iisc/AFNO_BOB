"""Verify TIGGE-driven ensemble ocean forecast against GLORYS reanalysis.

For each forecast lead day, loads the corresponding GLORYS ocean truth from
ocean.nc, denormalises the ensemble mean to physical units, and computes
pointwise verification metrics over ocean pixels.

Metrics computed per variable per lead day:
  rmse        : RMSE of ensemble mean vs truth (physical units)
  mae         : MAE of ensemble mean vs truth
  bias        : mean signed error (mean - truth)
  corr        : Pearson correlation (ensemble mean vs truth)
  spread      : mean ensemble std over ocean pixels (physical units for uo/vo/zos;
                normalised-space for thetao/so because std is scale-invariant under
                mean subtraction)
  spread_skill: spread / rmse  (1 = perfectly calibrated, <1 = overconfident)
  crps        : mean Continuous Ranked Probability Score over ocean pixels

Outputs (written to --output_dir):
  verification_metrics.csv  — scalar metrics, one row per (variable, lead_day)
  error_fields.npy          — |pred_mean - truth|, shape (n_steps, 5, 229, 265)
  bias_fields.npy           — pred_mean - truth,   shape (n_steps, 5, 229, 265)

Inputs:
    --ensemble_dir (str): Directory with ensemble_predictions.npy,
        ensemble_mean.npy, ensemble_std.npy, ensemble_metadata.json.
    --config_file (str): YAML config filename in config/ (provides data_dir).
    --mean_dir (str): Directory with normalisation mean .npy files
        (default: data/1993_2020/mean).
    --output_dir (str): Output directory (default: ensemble_dir/verification).

Example:
    conda activate BoB_Surf_2
    python src/inference/run_ensemble_verification.py \\
        --ensemble_dir results/AFNO_BoB_Surf_E00/ensemble_Jan_14012020 \\
        --config_file afno_bob_surf_e00.yaml
"""

import sys
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn.functional as F
import netCDF4 as nc

sys.path.append(str(Path(__file__).parent.parent))

from configmypy import ConfigPipeline, YamlConfig
from inference.utils import date_to_day_index


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OCEAN_VARS  = ['thetao', 'so', 'uo', 'vo', 'zos']
VAR_IDX     = {v: i for i, v in enumerate(OCEAN_VARS)}
TARGET_SIZE = (229, 265)   # GLORYS native grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_norm_stats(mean_dir: Path) -> dict:
    """Load climatological mean arrays for ocean variables.

    Args:
        mean_dir (Path): Directory containing mean_{var}_1993_2018_all_months.npy.

    Returns:
        dict: Variable name → float32 array of shape (1, 229, 265), or None if missing.

    Example:
        >>> mean_dict = load_norm_stats(Path('data/1993_2020/mean'))
        >>> mean_dict['thetao'].shape
        (1, 229, 265)
    """
    mean_dict = {}
    for var in OCEAN_VARS:
        f = mean_dir / f'mean_{var}_1993_2018_all_months.npy'
        mean_dict[var] = np.load(f).astype(np.float32) if f.exists() else None
    return mean_dict


def load_glorys_truth(ocean_file: Path, day_index: int) -> dict[str, np.ndarray]:
    """Load one day of GLORYS reanalysis ocean data.

    Args:
        ocean_file (Path): Path to ocean.nc.
        day_index (int): Time index in ocean.nc (0-based days since 01-01-1993).

    Returns:
        dict: Variable name → float32 array of shape (229, 265).
            Fill-value pixels (land) are returned as-is; mask is applied separately.

    Example:
        >>> truth = load_glorys_truth(Path('ocean.nc'), 9875)
        >>> truth['thetao'].shape
        (229, 265)
    """
    ds = nc.Dataset(ocean_file)
    truth = {}
    for var in OCEAN_VARS:
        raw = ds[var]
        if raw.ndim == 4:          # (time, depth, lat, lon)
            arr = raw[day_index, 0, :, :]
        else:                      # zos has no depth dim: (time, lat, lon)
            arr = raw[day_index, :, :]
        # Convert masked array to ndarray with NaN for masked pixels
        arr = np.ma.filled(np.ma.array(arr), fill_value=np.nan)
        truth[var] = arr.astype(np.float32)
    ds.close()
    return truth


def interp_to_glorys(arr_224: np.ndarray) -> np.ndarray:
    """Bilinearly interpolate a (224, 224) field to GLORYS grid (229, 265).

    Args:
        arr_224 (np.ndarray): float32 array of shape (224, 224).

    Returns:
        np.ndarray: float32 array of shape (229, 265).

    Example:
        >>> out = interp_to_glorys(pred_224)
        >>> out.shape
        (229, 265)
    """
    t = torch.tensor(arr_224[np.newaxis, np.newaxis], dtype=torch.float32)
    t = F.interpolate(t, size=TARGET_SIZE, mode='bilinear', align_corners=False)
    return t.squeeze().numpy()


def denormalize(arr_glorys: np.ndarray, var: str, mean_dict: dict) -> np.ndarray:
    """Add back climatological mean to convert normalised model output to physical units.

    thetao and so were normalised by mean subtraction during training.
    uo, vo, zos were not normalised, so they are returned unchanged.

    Args:
        arr_glorys (np.ndarray): float32 array of shape (229, 265), in normalised space.
        var (str): Ocean variable name.
        mean_dict (dict): Climatological mean arrays keyed by variable name.

    Returns:
        np.ndarray: float32 array of shape (229, 265) in physical units.

    Example:
        >>> sst_phys = denormalize(pred_norm, 'thetao', mean_dict)
    """
    if var in ('thetao', 'so') and mean_dict.get(var) is not None:
        return arr_glorys + np.squeeze(mean_dict[var])
    return arr_glorys


def compute_crps(members_glorys: np.ndarray, truth: np.ndarray,
                 ocean_mask: np.ndarray) -> float:
    """Compute mean CRPS of an ensemble forecast against point observations.

    Uses the energy-score decomposition:
        CRPS = E|X - y| - (1 / 2N^2) * sum_{i,j} |X_i - X_j|

    The pair-spread term is computed efficiently via the sorted-member identity
    (Ferro and Fricker 2012).

    Args:
        members_glorys (np.ndarray): Ensemble members interpolated to GLORYS grid,
            shape (N_members, 229, 265). Must be in physical units (denormalised).
        truth (np.ndarray): Ground truth field, shape (229, 265), physical units.
        ocean_mask (np.ndarray): Boolean array, True where pixel is ocean (valid).

    Returns:
        float: Mean CRPS over all valid ocean pixels.

    Example:
        >>> crps = compute_crps(members_glorys, sst_truth, ~land_mask)
    """
    N = members_glorys.shape[0]
    y = truth[ocean_mask]           # (n_pixels,)
    X = members_glorys[:, ocean_mask]  # (N, n_pixels)

    # Term 1: mean absolute error to observation
    term1 = np.mean(np.abs(X - y[np.newaxis, :]), axis=0)  # (n_pixels,)

    # Term 2: spread — sorted-member identity (1-indexed)
    X_sort = np.sort(X, axis=0)                              # (N, n_pixels)
    ranks  = np.arange(1, N + 1, dtype=np.float32)[:, np.newaxis]
    term2  = np.sum((2 * ranks - N - 1) * X_sort, axis=0) / (N ** 2)

    return float(np.mean(term1 - term2))


def compute_metrics(pred: np.ndarray, truth: np.ndarray,
                    ocean_mask: np.ndarray) -> dict:
    """Compute scalar verification metrics over ocean pixels.

    Args:
        pred (np.ndarray): Ensemble mean field, shape (229, 265), physical units.
        truth (np.ndarray): Ground truth field, shape (229, 265), physical units.
        ocean_mask (np.ndarray): Boolean array, True where pixel is valid ocean.

    Returns:
        dict: Keys rmse, mae, bias, corr. All float, NaN if no valid pixels.

    Example:
        >>> m = compute_metrics(sst_pred, sst_truth, ocean_mask)
        >>> m['rmse']
        0.453
    """
    # Intersect ocean mask with non-NaN in both pred and truth
    valid = ocean_mask & ~np.isnan(pred) & ~np.isnan(truth)
    if valid.sum() == 0:
        return dict(rmse=np.nan, mae=np.nan, bias=np.nan, corr=np.nan)

    p   = pred[valid]
    t   = truth[valid]
    err = p - t

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    corr = float(np.corrcoef(p, t)[0, 1]) if (p.std() > 0 and t.std() > 0) else np.nan

    return dict(rmse=rmse, mae=mae, bias=bias, corr=corr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Load ensemble outputs and GLORYS truth; compute and save verification metrics.

    Args:
        None: Reads --ensemble_dir, --config_file, --mean_dir, --output_dir
            from sys.argv.

    Returns:
        None: Writes verification_metrics.csv, error_fields.npy, bias_fields.npy.

    Example:
        >>> # python src/inference/run_ensemble_verification.py \\
        >>> #     --ensemble_dir results/AFNO_BoB_Surf_E00/ensemble_Jan_14012020 \\
        >>> #     --config_file afno_bob_surf_e00.yaml
    """
    parser = argparse.ArgumentParser(
        description='Ensemble forecast verification against GLORYS reanalysis'
    )
    parser.add_argument('--ensemble_dir', required=True,
                        help='Directory with ensemble_*.npy and metadata.json')
    parser.add_argument('--config_file', default='afno_bob_surf_e00.yaml',
                        help='YAML config filename in config/')
    parser.add_argument('--mean_dir', default='data/1993_2020/mean',
                        help='Directory with normalisation mean .npy files')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory (default: ensemble_dir/verification)')
    args = parser.parse_args()

    ens_dir  = Path(args.ensemble_dir)
    out_dir  = Path(args.output_dir) if args.output_dir else ens_dir / 'verification'
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Config (for ocean.nc path) ---
    pipe = ConfigPipeline([
        YamlConfig(args.config_file, config_name='default', config_folder='config/'),
    ])
    config = pipe.read_conf()
    ocean_file = Path(config.data.data_dir) / f'{config.data.file_prefix}.nc'
    print(f"Ocean file : {ocean_file}")

    # --- Load ensemble outputs ---
    meta      = json.load(open(ens_dir / 'ensemble_metadata.json'))
    init_date = meta['input_date']
    lead_h    = meta['lead_times_h']
    n_steps   = len(lead_h)
    init_dt   = datetime.strptime(init_date, '%d-%m-%Y')

    print(f"Init date  : {init_date}  ({n_steps} lead steps)")
    print(f"Output dir : {out_dir}")

    print("\nLoading ensemble arrays ...", flush=True)
    ens_mean  = np.load(ens_dir / 'ensemble_mean.npy')        # (n_steps, 5, 224, 224)
    ens_std   = np.load(ens_dir / 'ensemble_std.npy')         # (n_steps, 5, 224, 224)
    ens_preds = np.load(ens_dir / 'ensemble_predictions.npy') # (50, n_steps, 5, 224, 224)
    n_members = ens_preds.shape[0]
    print(f"  Predictions: {ens_preds.shape}  ({n_members} members)")

    # --- Normalisation stats and land mask ---
    mean_dict = load_norm_stats(Path(args.mean_dir))
    land_mask = np.isnan(np.squeeze(mean_dict['thetao']))  # (229, 265)
    ocean_mask = ~land_mask
    print(f"  Ocean pixels: {ocean_mask.sum()} / {ocean_mask.size}")

    # --- Allocate spatial output arrays (physical units) ---
    error_fields = np.full((n_steps, 5, *TARGET_SIZE), np.nan, dtype=np.float32)
    bias_fields  = np.full((n_steps, 5, *TARGET_SIZE), np.nan, dtype=np.float32)

    # --- CSV writer ---
    csv_path = out_dir / 'verification_metrics.csv'
    csv_cols = ['init_date', 'lead_day', 'variable',
                'rmse', 'mae', 'bias', 'corr', 'spread', 'spread_skill', 'crps']
    csv_file = open(csv_path, 'w', newline='')
    writer   = csv.DictWriter(csv_file, fieldnames=csv_cols)
    writer.writeheader()

    # --- Loop over lead steps ---
    print(f"\nVerifying {n_steps} lead steps ...\n")

    for s in range(n_steps):
        lead_day  = s + 1
        valid_dt  = init_dt + timedelta(hours=int(lead_h[s]))
        day_index = date_to_day_index(valid_dt.strftime('%d-%m-%Y'), '01-01-1993')
        print(f"  Lead +{lead_day}d  ({valid_dt.strftime('%d-%m-%Y')}, "
              f"day_index={day_index})", flush=True)

        truth = load_glorys_truth(ocean_file, day_index)

        for vi, var in enumerate(OCEAN_VARS):
            # --- Denormalise ensemble mean ---
            pred_norm  = ens_mean[s, vi]                        # (224, 224)
            pred_glorys = interp_to_glorys(pred_norm)           # (229, 265)
            pred_phys   = denormalize(pred_glorys, var, mean_dict)

            # --- Denormalise ensemble members for CRPS ---
            members_norm   = ens_preds[:, s, vi]                # (50, 224, 224)
            members_glorys = np.stack(
                [interp_to_glorys(members_norm[m]) for m in range(n_members)]
            )                                                    # (50, 229, 265)
            members_phys   = np.stack(
                [denormalize(members_glorys[m], var, mean_dict) for m in range(n_members)]
            )

            # --- Apply land mask to prediction ---
            pred_phys[land_mask] = np.nan

            truth_var = truth[var]

            # --- Scalar metrics ---
            m = compute_metrics(pred_phys, truth_var, ocean_mask)

            # --- Mean spread (std in normalised space, scale-invariant for mean-sub) ---
            std_glorys = interp_to_glorys(ens_std[s, vi])       # (229, 265)
            spread     = float(std_glorys[ocean_mask].mean())

            spread_skill = spread / m['rmse'] if m['rmse'] > 0 else np.nan

            # --- CRPS ---
            crps = compute_crps(members_phys, truth_var, ocean_mask)

            # --- Spatial error fields ---
            valid = ocean_mask & ~np.isnan(pred_phys) & ~np.isnan(truth_var)
            err_field = np.full(TARGET_SIZE, np.nan, dtype=np.float32)
            err_field[valid] = np.abs((pred_phys - truth_var)[valid])
            error_fields[s, vi] = err_field

            bias_field = np.full(TARGET_SIZE, np.nan, dtype=np.float32)
            bias_field[valid] = (pred_phys - truth_var)[valid]
            bias_fields[s, vi] = bias_field

            writer.writerow({
                'init_date'   : init_date,
                'lead_day'    : lead_day,
                'variable'    : var,
                'rmse'        : round(m['rmse'],  4),
                'mae'         : round(m['mae'],   4),
                'bias'        : round(m['bias'],  4),
                'corr'        : round(m['corr'],  4),
                'spread'      : round(spread,     4),
                'spread_skill': round(spread_skill, 4) if not np.isnan(spread_skill) else '',
                'crps'        : round(crps,       4),
            })

            print(f"    {var:7s}: RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  "
                  f"Bias={m['bias']:+.4f}  r={m['corr']:.3f}  "
                  f"Spread={spread:.4f}  SS={spread_skill:.3f}  CRPS={crps:.4f}")

    csv_file.close()

    # --- Save spatial fields ---
    np.save(out_dir / 'error_fields.npy', error_fields)
    np.save(out_dir / 'bias_fields.npy',  bias_fields)

    print(f"\nSaved:")
    print(f"  {csv_path}")
    print(f"  {out_dir / 'error_fields.npy'}   shape={error_fields.shape}")
    print(f"  {out_dir / 'bias_fields.npy'}    shape={bias_fields.shape}")


if __name__ == '__main__':
    main()

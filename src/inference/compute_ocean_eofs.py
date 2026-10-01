"""Compute seasonal empirical orthogonal functions (EOFs) of ocean state variables.

For each of the four Bay of Bengal seasons and each of the five ocean variables,
loads all daily normalized ocean snapshots from the GLORYS training record
(default 1993–2018), computes the leading EOFs via randomised SVD, and retains
enough modes to explain a target fraction of variance (default 90%).  The EOF
patterns, per-mode standard deviations, and seasonal means are saved to disk so
that run_ensemble_inference_ic.py can draw statistically consistent initial
condition perturbations at inference time.

Season definitions:
    Winter:      December, January, February  (months 12, 1, 2)
    PreMonsoon:  March, April, May            (months 3, 4, 5)
    Monsoon:     June, July, August, September (months 6, 7, 8, 9)
    PostMonsoon: October, November            (months 10, 11)

Normalization applied before EOF computation matches the model's forward pass:
    thetao, so  →  sample − climatological_mean   (mean subtraction only)
    uo, vo, zos →  no normalization

Inputs:
    --data_dir (str): Directory containing ocean.nc (GLORYS reanalysis).
    --mean_dir (str): Directory with per-variable mean .npy files.
    --output_dir (str): Where to write EOF .npy files and metadata JSON.
    --train_years (str): Inclusive year range 'YYYY-YYYY' (default: 1993-2018).
    --variance_threshold (float): Fraction of variance EOFs must explain (default: 0.9).
    --n_components_max (int): Cap on SVD components requested per variable
        (default: 200; actual retained count is usually far smaller).
    --img_size (int): Spatial resolution fed to the model (default: 224).

Outputs:
    {output_dir}/{season}_{var}_eofs.npy    — (k, img_size, img_size) EOF patterns
    {output_dir}/{season}_{var}_pcstd.npy   — (k,) per-mode PC standard deviations
    {output_dir}/{season}_{var}_mean.npy    — (img_size, img_size) seasonal mean
    {output_dir}/eof_metadata.json          — config, n_modes, variance explained

Example:
    conda activate BoB_Surf_2
    python src/inference/compute_ocean_eofs.py \\
        --data_dir data/1993_2020 \\
        --mean_dir data/1993_2020/mean \\
        --output_dir data/ocean_eofs
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, date

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from sklearn.utils.extmath import randomized_svd


# ---------------------------------------------------------------------------
# Season / variable configuration
# ---------------------------------------------------------------------------

SEASONS = {
    'Winter':      [12, 1, 2],
    'PreMonsoon':  [3, 4, 5],
    'Monsoon':     [6, 7, 8, 9],
    'PostMonsoon': [10, 11],
}

OCEAN_VARS = ['thetao', 'so', 'uo', 'vo', 'zos']

NORTH_MASK_ROWS = 20  # rows zeroed at northern boundary (matches trainer)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_mean(mean_dir: Path, var: str) -> np.ndarray:
    """Load climatological mean array for one ocean variable.

    Args:
        mean_dir (Path): Directory containing mean .npy files.
        var (str): Variable name (e.g. 'thetao').

    Returns:
        np.ndarray: Mean array at native GLORYS resolution or scalar 0 if not
            applicable.

    Example:
        >>> mu = load_mean(Path('/data/mean'), 'thetao')
    """
    path = mean_dir / f'mean_{var}_1993_2018_all_months.npy'
    if path.exists():
        return np.load(path).squeeze()   # drop singleton depth dim → (H, W)
    return np.float32(0.0)


def normalize_ocean(data: np.ndarray, var: str, mean: np.ndarray) -> np.ndarray:
    """Apply the same per-variable normalization used during model training.

    Args:
        data (np.ndarray): Raw ocean snapshot, shape (H, W), float32.
        var (str): Variable name.
        mean (np.ndarray): Climatological mean at native resolution.

    Returns:
        np.ndarray: Normalized array, same shape as data.

    Example:
        >>> norm = normalize_ocean(raw, 'thetao', mean)
    """
    if var in ('thetao', 'so'):
        data = data - mean
    return data


def interp_to_img(data: np.ndarray, img_size: int) -> np.ndarray:
    """Bilinear interpolation of a 2-D field to (img_size, img_size).

    NaNs are replaced with the per-snapshot spatial mean before interpolation
    (matching PreprocessTransform behaviour).

    Args:
        data (np.ndarray): Input 2-D field, shape (H, W).
        img_size (int): Target spatial size.

    Returns:
        np.ndarray: Interpolated field, shape (img_size, img_size).

    Example:
        >>> field = interp_to_img(raw_229x265, 224)
        >>> field.shape
        (224, 224)
    """
    data = data.astype(np.float32)
    nan_mask = np.isnan(data)
    if nan_mask.any():
        fill = float(np.nanmean(data))
        data[nan_mask] = fill

    t = torch.tensor(data).unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
    t = F.interpolate(t, size=(img_size, img_size),
                      mode='bilinear', align_corners=False)
    return t.squeeze().numpy()


def apply_north_mask(field: np.ndarray, rows: int = NORTH_MASK_ROWS) -> np.ndarray:
    """Zero out the northern boundary rows to match the training mask.

    Args:
        field (np.ndarray): 2-D or 3-D array with the last axis being latitude,
            shape (..., H, W).
        rows (int): Number of top rows to zero (default 20).

    Returns:
        np.ndarray: Field with top-rows zeroed.

    Example:
        >>> masked = apply_north_mask(field_224x224)
    """
    out = field.copy()
    if rows > 0:
        out[..., -rows:, :] = 0.0
    return out


# ---------------------------------------------------------------------------
# Data loading for one variable / season
# ---------------------------------------------------------------------------

def collect_season_snapshots(
    ocean_ds: xr.Dataset,
    var: str,
    mean: np.ndarray,
    season_months: list,
    year_start: int,
    year_end: int,
    img_size: int,
) -> np.ndarray:
    """Load, normalize, and interpolate all daily snapshots for one season/variable.

    Args:
        ocean_ds (xr.Dataset): Open GLORYS ocean dataset.
        var (str): Variable name.
        mean (np.ndarray): Climatological mean at native resolution.
        season_months (list[int]): Month numbers belonging to this season.
        year_start (int): First training year (inclusive).
        year_end (int): Last training year (inclusive).
        img_size (int): Target spatial resolution.

    Returns:
        np.ndarray: Float32 array of shape (n_samples, img_size, img_size).

    Example:
        >>> snaps = collect_season_snapshots(ds, 'thetao', mean, [12,1,2], 1993, 2018, 224)
        >>> snaps.shape
        (2340, 224, 224)
    """
    times = ocean_ds['time'].values
    snapshots = []

    for ti, t in enumerate(times):
        dt = t.astype('datetime64[D]').astype(date)
        if not (year_start <= dt.year <= year_end):
            continue
        if dt.month not in season_months:
            continue

        raw = ocean_ds[var][ti].values.astype(np.float32)
        raw = raw.squeeze()                              # drop singleton depth/time dims → (H, W)
        norm = normalize_ocean(raw, var, mean)
        field = interp_to_img(norm, img_size)
        field = apply_north_mask(field)
        snapshots.append(field)

    return np.stack(snapshots, axis=0).astype(np.float32)   # (N, img_size, img_size)


# ---------------------------------------------------------------------------
# EOF computation
# ---------------------------------------------------------------------------

def compute_eofs(
    snapshots: np.ndarray,
    variance_threshold: float,
    n_components_max: int,
) -> dict:
    """Compute leading EOFs via randomised SVD and retain those explaining the threshold variance.

    Args:
        snapshots (np.ndarray): Shape (n_samples, H, W). Raw (possibly mean-subtracted)
            seasonal snapshots in normalized model space.
        variance_threshold (float): Cumulative variance fraction to explain (e.g. 0.9).
        n_components_max (int): Maximum number of SVD components to request.

    Returns:
        dict with keys:
            'eofs'   (np.ndarray): Shape (k, H, W) — retained EOF patterns.
            'pcstd'  (np.ndarray): Shape (k,) — per-mode PC standard deviation.
            'mean'   (np.ndarray): Shape (H, W) — sample mean subtracted before SVD.
            'var_explained' (np.ndarray): Shape (n_components_max,) — cumulative
                variance fraction.
            'n_modes' (int): Number of retained modes k.

    Example:
        >>> result = compute_eofs(snaps, 0.9, 200)
        >>> result['eofs'].shape
        (23, 224, 224)
    """
    n, H, W = snapshots.shape
    n_comp = min(n_components_max, n - 1)

    # Flatten to (n_samples, n_pixels)
    X = snapshots.reshape(n, -1)               # (n, H*W)

    # Subtract per-pixel seasonal mean
    season_mean = X.mean(axis=0)               # (H*W,)
    X = X - season_mean

    # Randomised SVD: X ≈ U S Vt,  Vt rows are EOF patterns
    U, s, Vt = randomized_svd(X, n_components=n_comp, random_state=42)

    # Variance explained per mode
    var_per_mode = s ** 2 / (n - 1)
    total_var = var_per_mode.sum()
    cum_var = np.cumsum(var_per_mode) / total_var

    # Find k: fewest modes to reach threshold
    k_idx = np.searchsorted(cum_var, variance_threshold)
    k = min(k_idx + 1, len(s))

    eofs = Vt[:k].reshape(k, H, W)            # (k, H, W)
    pcstd = s[:k] / np.sqrt(n - 1)            # std of PC scores in model space

    return {
        'eofs':          eofs.astype(np.float32),
        'pcstd':         pcstd.astype(np.float32),
        'mean':          season_mean.reshape(H, W).astype(np.float32),
        'var_explained': cum_var.astype(np.float32),
        'n_modes':       int(k),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Parse CLI arguments, load GLORYS data, compute seasonal EOFs, and save outputs.

    Args:
        None: All parameters sourced from sys.argv via argparse.

    Returns:
        None: Writes .npy files and metadata JSON to --output_dir.

    Example:
        >>> # python src/inference/compute_ocean_eofs.py \\
        >>> #     --data_dir data/1993_2020 \\
        >>> #     --mean_dir data/1993_2020/mean \\
        >>> #     --output_dir data/ocean_eofs
    """
    parser = argparse.ArgumentParser(
        description='Compute seasonal ocean EOFs from GLORYS training data'
    )
    parser.add_argument('--data_dir', default='data/1993_2020',
                        help='Directory containing ocean.nc')
    parser.add_argument('--mean_dir', default='data/1993_2020/mean',
                        help='Directory with mean_*.npy files')
    parser.add_argument('--output_dir', default='data/ocean_eofs',
                        help='Output directory for EOF files')
    parser.add_argument('--train_years', default='1993-2018',
                        help='Training year range YYYY-YYYY (default: 1993-2018)')
    parser.add_argument('--variance_threshold', type=float, default=0.9,
                        help='Cumulative variance fraction to retain (default: 0.9)')
    parser.add_argument('--n_components_max', type=int, default=200,
                        help='Max SVD components to compute (default: 200)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Model spatial resolution (default: 224)')
    args = parser.parse_args()

    year_start, year_end = [int(y) for y in args.train_years.split('-')]
    data_dir  = Path(args.data_dir)
    mean_dir  = Path(args.mean_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ocean_path = data_dir / 'ocean.nc'
    print(f'Loading ocean data from {ocean_path} ...', flush=True)
    ocean_ds = xr.open_dataset(ocean_path)

    metadata = {
        'created':            datetime.now().isoformat(),
        'train_years':        args.train_years,
        'variance_threshold': args.variance_threshold,
        'img_size':           args.img_size,
        'seasons':            {s: m for s, m in SEASONS.items()},
        'variables':          OCEAN_VARS,
        'modes':              {},
    }

    for season, months in SEASONS.items():
        print(f'\n{"="*60}')
        print(f'Season: {season}  (months {months})')
        print(f'{"="*60}')

        for var in OCEAN_VARS:
            mean = load_mean(mean_dir, var)

            print(f'  {var}: collecting snapshots ... ', end='', flush=True)
            snaps = collect_season_snapshots(
                ocean_ds, var, mean, months,
                year_start, year_end, args.img_size,
            )
            print(f'{snaps.shape[0]} samples  shape={snaps.shape}', flush=True)

            print(f'  {var}: computing EOFs (max {args.n_components_max}) ... ',
                  end='', flush=True)
            result = compute_eofs(snaps, args.variance_threshold,
                                  args.n_components_max)
            k = result['n_modes']
            cum_var_k = float(result['var_explained'][k - 1])
            print(f'{k} modes → {cum_var_k*100:.1f}% variance', flush=True)

            # Save
            tag = f'{season}_{var}'
            np.save(output_dir / f'{tag}_eofs.npy',  result['eofs'])
            np.save(output_dir / f'{tag}_pcstd.npy', result['pcstd'])
            np.save(output_dir / f'{tag}_mean.npy',  result['mean'])

            metadata['modes'][f'{season}/{var}'] = {
                'n_modes':         k,
                'var_explained':   cum_var_k,
                'n_samples':       snaps.shape[0],
            }

    (output_dir / 'eof_metadata.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8'
    )
    print(f'\nEOF files written to {output_dir}/')
    print(f'Metadata: {output_dir / "eof_metadata.json"}')


if __name__ == '__main__':
    main()

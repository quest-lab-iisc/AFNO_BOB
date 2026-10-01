"""Preprocess TIGGE ensemble GRIB files into normalised NumPy tensors.

Reads a TIGGE perturbed-forecast GRIB file (50 members, 9 lead-time steps,
one or more init days) and converts each member into a pre-normalised array
of shape (N_days, 9, 6, 224, 224) that can be fed directly into the
ocean-only AFNO model as atmospheric forcing.

The TIGGE ensemble GRIB uses an unstructured ECMWF reduced Gaussian grid
(~18 000 spatial points). This script regrids each field onto a regular
0.25° lat/lon grid matching the ERA5 training data, then applies the same
normalisation used during inference:
  - ssr, tp, msl : z-score  (x - mean) / sqrt(var)
  - u10, v10, tcc: no normalisation  (matches PreprocessTransform.forward())

ECMWF TIGGE GRIB files store SSR and TP as cumulative flux from T+0, not
daily increments. This script de-accumulates both variables before normalisation
so that each lead step carries only the 24-h increment, matching how ERA5
stores them in atm.nc (the training data).

Variable order is fixed: ['ssr', 'tp', 'u10', 'v10', 'msl', 'tcc']
matching config.data.atm_variable.

Inputs:
    grib_file (Path): TIGGE ensemble GRIB file, e.g. data/Tigge_Jan_2020_ens.grib.
    output_dir (Path): Directory where per-member .npy files are written.
    mean_dir (Path): Directory containing normalisation stats (.npy files).
        Required files:
            {var}_mean_1993_2018_all_months.npy  (ssr, tp, msl)
            {var}_var_1993_2018_all_months.npy   (ssr, tp, msl)

Outputs:
    {output_dir}/{stem}_m{nn:02d}.npy (float32, shape (N_days, 9, 6, 224, 224)):
        Pre-normalised atmospheric forcing for member nn (01–50).
        Axis meanings: [init_day, lead_step, variable, lat, lon]
        lead_step=0 → +24 h, lead_step=8 → +216 h.

Example:
    conda activate BoB_Surf_2
    python src/data_pipeline/prepare_tigge_ensemble.py \\
        --grib_file data/Tigge_Jan_2020_ens.grib \\
        --output_dir data/tigge_ensemble \\
        --mean_dir data/1993_2020/mean
"""

import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

try:
    import xarray as xr
except ImportError:
    print("ERROR: xarray is not installed.  pip install xarray")
    sys.exit(1)

try:
    import cfgrib  # noqa: F401 — registers cfgrib xarray engine
except ImportError:
    print("ERROR: cfgrib is not installed.  pip install cfgrib")
    sys.exit(1)

try:
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
except ImportError:
    print("ERROR: scipy is not installed.  pip install scipy")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Variable configuration
# ---------------------------------------------------------------------------

# Order must match config.data.atm_variable: ['ssr', 'tp', 'u10', 'v10', 'msl', 'tcc']
ATM_VARS = ['ssr', 'tp', 'u10', 'v10', 'msl', 'tcc']

PARAM_ID = {
    'ssr': 176,
    'tp' : 228228,
    'u10': 165,
    'v10': 166,
    'msl': 151,
    'tcc': 228164,
}

# Variables that receive z-score normalisation; others are passed through raw
ZSCORE_VARS = {'ssr', 'tp', 'msl'}

# ---------------------------------------------------------------------------
# Target regular 0.25° grid matching the ERA5 training data
# lat: 4.0 → 23.0 (S→N, 77 points), lon: 77.0 → 99.0 (W→E, 89 points)
# S→N orientation matches atm.nc and ocean.nc (lat index 0 = 4°N southernmost).
# ---------------------------------------------------------------------------
LAT_TARGET = np.arange(4.0, 23.01, 0.25).astype(np.float64)    # (77,)
LON_TARGET = np.arange(77.0, 99.01, 0.25).astype(np.float64)   # (89,)
_LON_GRID, _LAT_GRID = np.meshgrid(LON_TARGET, LAT_TARGET)      # each (77, 89)
_TARGET_PTS = np.column_stack([_LON_GRID.ravel(),
                                _LAT_GRID.ravel()])              # (6853, 2)
NLAT, NLON = len(LAT_TARGET), len(LON_TARGET)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_norm_stats(mean_dir: Path) -> tuple[dict, dict]:
    """Load normalisation mean and std arrays from .npy files.

    Args:
        mean_dir (Path): Directory containing the .npy statistics files.

    Returns:
        tuple[dict, dict]: (mean_dict, std_dict). mean_dict maps all ATM_VARS
            to their mean array (or None if file missing). std_dict maps
            only ssr/tp/msl to their std array (or None if file missing).

    Example:
        >>> mean_dict, std_dict = load_norm_stats(Path('/data/mean'))
        >>> mean_dict['ssr'].shape
        (1, 77, 89)
    """
    mean_dict = {}
    std_dict  = {}
    for var in ATM_VARS:
        mean_file = mean_dir / f"{var}_mean_1993_2018_all_months.npy"
        if mean_file.exists():
            mean_dict[var] = np.load(mean_file)
        else:
            warnings.warn(f"Mean file not found for {var}: {mean_file} — "
                          f"variable will not be mean-subtracted.")
            mean_dict[var] = None

    for var in ZSCORE_VARS:
        var_file = mean_dir / f"{var}_var_1993_2018_all_months.npy"
        if var_file.exists():
            std_dict[var] = np.sqrt(np.load(var_file))
        else:
            warnings.warn(f"Variance file not found for {var}: {var_file} — "
                          f"variable will not be std-scaled.")
            std_dict[var] = None

    return mean_dict, std_dict


def build_regrid_interpolators(lats: np.ndarray, lons: np.ndarray):
    """Build scipy interpolators for regriding from unstructured Gaussian grid.

    The TIGGE ECMWF Gaussian grid is unstructured; each field is stored as
    a 1-D 'values' array with associated lat/lon coordinate arrays. This
    function precomputes the Delaunay triangulation once so that all
    subsequent fields can be regrided cheaply.

    Args:
        lats (np.ndarray): 1-D source latitude array of shape (N_pts,).
        lons (np.ndarray): 1-D source longitude array of shape (N_pts,).

    Returns:
        tuple: (linear_interp, nearest_interp) where
            linear_interp  : LinearNDInterpolator — used for interior points.
            nearest_interp : NearestNDInterpolator — used to fill boundary NaNs.

    Example:
        >>> lin, nn = build_regrid_interpolators(lats, lons)
        >>> lin.values = field.reshape(-1, 1).astype(np.float64)
        >>> result = lin(_TARGET_PTS).reshape(NLAT, NLON)
    """
    print("  Building regrid interpolators (one-time Delaunay triangulation)...",
          flush=True)
    source_pts = np.column_stack([lons.astype(np.float64),
                                  lats.astype(np.float64)])
    dummy = np.zeros(len(lats), dtype=np.float64)
    linear_interp  = LinearNDInterpolator(source_pts, dummy)
    nearest_interp = NearestNDInterpolator(source_pts, dummy)
    return linear_interp, nearest_interp


def regrid_field(values: np.ndarray,
                 linear_interp,
                 nearest_interp) -> np.ndarray:
    """Regrid a single 1-D values field onto the target regular lat/lon grid.

    Uses linear interpolation for interior points and nearest-neighbour for
    boundary points that fall outside the Gaussian grid's convex hull.

    Args:
        values (np.ndarray): 1-D float32 field of shape (N_pts,).
        linear_interp  : LinearNDInterpolator built by build_regrid_interpolators.
        nearest_interp : NearestNDInterpolator built by build_regrid_interpolators.

    Returns:
        np.ndarray: float32 array of shape (NLAT, NLON) = (77, 89).

    Example:
        >>> grid = regrid_field(raw_vals, lin, nn)
        >>> grid.shape
        (77, 89)
    """
    v64 = values.astype(np.float64)
    linear_interp.values  = v64.reshape(-1, 1)
    nearest_interp.values = v64

    result = linear_interp(_TARGET_PTS).reshape(NLAT, NLON)
    nan_mask = np.isnan(result)
    if nan_mask.any():
        result[nan_mask] = nearest_interp(_TARGET_PTS[nan_mask.ravel()])

    return result.astype(np.float32)


def normalize_and_interp(data_2d: np.ndarray, var: str,
                          mean_dict: dict, std_dict: dict) -> np.ndarray:
    """Normalise a (NLAT, NLON) field and bilinearly interpolate to 224×224.

    Normalisation mirrors PreprocessTransform.forward() for type='atm':
      - ssr, tp, msl: (x - mean) / std
      - u10, v10, tcc: no normalisation (raw values, matching training)

    Args:
        data_2d (np.ndarray): float32 array of shape (NLAT, NLON) = (77, 89).
        var (str): Variable name, one of ATM_VARS.
        mean_dict (dict): Mean arrays keyed by variable name.
        std_dict (dict): Std arrays keyed by variable name (ssr/tp/msl only).

    Returns:
        np.ndarray: Normalised, interpolated float32 array of shape (224, 224).

    Example:
        >>> out = normalize_and_interp(grid, 'ssr', mean_dict, std_dict)
        >>> out.shape
        (224, 224)
    """
    arr = data_2d.astype(np.float32)

    # Unit conversions to match ERA5 atm.nc encoding:
    # - ssr: TIGGE is 24h accumulated J/m²; ERA5 stores hourly mean J/m² → divide by 24
    # - tp:  TIGGE is 24h accumulated kg/m² (mm); ERA5 stores hourly mean m → divide by 24000
    # - tcc: TIGGE is % (0-100); ERA5 is fraction (0-1) → divide by 100
    if var == 'ssr':
        arr /= 24.0
    elif var == 'tp':
        arr /= 24000.0
    elif var == 'tcc':
        arr /= 100.0

    if var in ZSCORE_VARS:
        mean = mean_dict.get(var)
        std  = std_dict.get(var)
        if mean is not None and std is not None:
            # Squeeze to (NLAT, NLON) to match arr shape before broadcasting
            arr = (arr - np.squeeze(mean)) / np.squeeze(std)

    # Bilinear interpolation to 224×224 (same as PreprocessTransform)
    t = torch.tensor(arr[np.newaxis, np.newaxis])  # (1, 1, NLAT, NLON)
    t = F.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)
    return t.squeeze().numpy()  # (224, 224)


def open_variable(grib_file: Path, param_id: int) -> xr.Dataset:
    """Open one variable from a TIGGE GRIB file using its GRIB paramId.

    Args:
        grib_file (Path): Path to the TIGGE GRIB file.
        param_id (int): GRIB paramId for the desired variable.

    Returns:
        xr.Dataset: Dataset containing a single data variable with dimensions
            that typically include 'number', 'step', and optionally 'time',
            plus a 'values' dimension for the spatial Gaussian grid points.

    Example:
        >>> ds = open_variable(Path('Tigge_Jan_2020_ens.grib'), 165)
        >>> list(ds.dims)
        ['number', 'step', 'values']
    """
    return xr.open_dataset(
        grib_file,
        engine='cfgrib',
        filter_by_keys={'paramId': param_id},
        indexpath=None,
        errors='ignore',
    )


def extract_array(ds: xr.Dataset) -> np.ndarray:
    """Extract data from cfgrib Dataset as (n_days, n_steps, n_members, n_pts).

    cfgrib returns TIGGE ensemble data with dimensions in a variable order
    depending on what was squeezed. This function uses named dimension access
    to reorder into a canonical (time, step, number, values) shape.

    Args:
        ds (xr.Dataset): Dataset returned by open_variable().

    Returns:
        np.ndarray: float32 array of shape (n_days, n_steps, n_members, n_pts).

    Example:
        >>> arr = extract_array(ds)
        >>> arr.shape  # e.g. (1, 9, 50, 18241)
        (1, 9, 50, 18241)
    """
    da = ds[list(ds.data_vars)[0]]
    dims = list(da.dims)

    # Reorder to canonical (time, step, number, values)
    order = [d for d in ('time', 'step', 'number', 'values') if d in dims]
    da = da.transpose(*order)
    arr = da.values.astype(np.float32)

    # Restore any squeezed singleton dims
    if 'time' not in dims:
        arr = arr[np.newaxis]       # → (1, ...)
    if 'number' not in dims:
        arr = np.expand_dims(arr, axis=2)  # → (..., 1, values)

    # Guarantee shape is (n_days, n_steps, n_members, n_pts)
    assert arr.ndim == 4, f"Expected 4-D array after reshape, got {arr.shape}"
    return arr


# ---------------------------------------------------------------------------
# Main preprocessing
# ---------------------------------------------------------------------------

def process_grib(grib_file: Path, output_dir: Path, mean_dir: Path) -> None:
    """Process all ensemble members from a TIGGE GRIB file and save .npy files.

    For each member, writes a float32 .npy file of shape
    (N_days, 9, 6, 224, 224) to output_dir. Existing files are skipped.

    Args:
        grib_file (Path): TIGGE ensemble GRIB (perturbed_forecast, 50 members).
        output_dir (Path): Directory for output .npy files.
        mean_dir (Path): Directory containing normalisation statistics.

    Returns:
        None: Writes per-member .npy files as side effects.

    Example:
        >>> process_grib(
        ...     Path('data/Tigge_Jan_2020_ens.grib'),
        ...     Path('data/tigge_ensemble'),
        ...     Path('data/1993_2020/mean'),
        ... )
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    mean_dict, std_dict = load_norm_stats(mean_dir)

    stem = grib_file.stem  # e.g. 'Tigge_Jan_2020_ens'

    print(f"\n{'='*60}")
    print(f"Processing: {grib_file.name}")
    print(f"Output dir: {output_dir}")
    print(f"Started   : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*60}")

    # -----------------------------------------------------------------------
    # Step 1: Load all variables → canonical (n_days, n_steps, n_members, n_pts)
    # -----------------------------------------------------------------------
    var_arrays = {}
    lats = lons = None
    linear_interp = nearest_interp = None
    n_days = n_steps = n_members = None

    for var in ATM_VARS:
        param_id = PARAM_ID[var]
        print(f"  Loading {var} (paramId={param_id}) ...", flush=True)

        ds  = open_variable(grib_file, param_id)
        arr = extract_array(ds)   # (n_days, n_steps, n_members, n_pts)

        if lats is None:
            lats = ds.latitude.values.astype(np.float64)
            lons = ds.longitude.values.astype(np.float64)
            linear_interp, nearest_interp = build_regrid_interpolators(lats, lons)

            n_days, n_steps, n_members, n_pts = arr.shape
            print(f"    GRIB shape : ({n_days} days, {n_steps} steps, "
                  f"{n_members} members, {n_pts} pts)")
            print(f"    Target grid: {NLAT} × {NLON} (0.25° regular)")

        var_arrays[var] = arr
        ds.close()

    # -----------------------------------------------------------------------
    # Step 1b: De-accumulate SSR (and TP).
    # TIGGE GRIB stores both as cumulative flux from T+0 (ECMWF convention).
    # ERA5 atm.nc stores daily (24-h) values, so the model expects daily
    # increments. Without de-accumulation, step+2 carries 48 h of radiation
    # instead of 24 h, causing the model input to grow ~4 σ per step and the
    # resulting forecast to drift catastrophically by lead+9.
    # -----------------------------------------------------------------------
    for var in ('ssr', 'tp'):
        if var in var_arrays:
            arr = var_arrays[var]                      # (n_days, n_steps, n_members, n_pts)
            arr_deaccum = arr.copy()
            arr_deaccum[:, 1:] = arr[:, 1:] - arr[:, :-1]   # daily difference; step 0 unchanged
            var_arrays[var] = arr_deaccum

    # -----------------------------------------------------------------------
    # Step 2: Regrid, normalise, interpolate → save per-member .npy files
    # -----------------------------------------------------------------------
    n_vars = len(ATM_VARS)

    for m in range(n_members):
        member_num = m + 1
        out_path = output_dir / f"{stem}_m{member_num:02d}.npy"

        if out_path.exists():
            print(f"  Skipping member {member_num:02d}: {out_path.name} already exists")
            continue

        print(f"  Member {member_num:02d}/{n_members} ...", flush=True)

        # Allocate output: (n_days, n_steps, n_vars, 224, 224)
        out = np.empty((n_days, n_steps, n_vars, 224, 224), dtype=np.float32)

        for d in range(n_days):
            for s in range(n_steps):
                for vi, var in enumerate(ATM_VARS):
                    raw  = var_arrays[var][d, s, m, :]       # (n_pts,)
                    grid = regrid_field(raw, linear_interp, nearest_interp)  # (77, 89)
                    out[d, s, vi] = normalize_and_interp(grid, var,
                                                          mean_dict, std_dict)

        np.save(out_path, out)
        print(f"    Saved {out_path.name}  shape={out.shape}  "
              f"({out.nbytes / 1e6:.1f} MB)")

    print(f"\nDone. {n_members} members written to {output_dir}/")
    print(f"Finished: {datetime.now():%Y-%m-%d %H:%M:%S}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Parse CLI arguments and run process_grib().

    Args:
        None: Reads --grib_file, --output_dir, --mean_dir from sys.argv.

    Returns:
        None: Delegates to process_grib() for side-effect file writes.

    Example:
        >>> # python src/data_pipeline/prepare_tigge_ensemble.py \\
        >>> #     --grib_file data/Tigge_Jan_2020_ens.grib \\
        >>> #     --output_dir data/tigge_ensemble \\
        >>> #     --mean_dir data/1993_2020/mean
    """
    parser = argparse.ArgumentParser(
        description="Preprocess TIGGE ensemble GRIB → per-member .npy tensors"
    )
    parser.add_argument("--grib_file", required=True,
                        help="Path to TIGGE ensemble GRIB file")
    parser.add_argument("--output_dir", default="data/tigge_ensemble",
                        help="Output directory for .npy files (default: data/tigge_ensemble)")
    parser.add_argument("--mean_dir", default="data/1993_2020/mean",
                        help="Directory with normalisation .npy files")
    args = parser.parse_args()

    process_grib(
        grib_file  = Path(args.grib_file),
        output_dir = Path(args.output_dir),
        mean_dir   = Path(args.mean_dir),
    )


if __name__ == "__main__":
    main()

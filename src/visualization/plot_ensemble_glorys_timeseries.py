"""Time-series comparison of ensemble predictions vs GLORYS (and OSTIA) for all ocean variables.

For each randomly selected ocean pixel, plots the 9-day ensemble mean ± 1σ
shaded band for all 5 ocean variables (SST, SSS, U-current, V-current, SSH),
overlaid with GLORYS L4 reanalysis values as filled black dots.  In the SST
row, OSTIA L4 SST is additionally shown as filled crimson dots and the ensemble
member with minimum MSE against OSTIA is drawn as a crimson dotted line
(distinct from the blue/orange dotted line for the GLORYS-closest member).

Produces two figures (one per season):
  fig11_glorys_timeseries_jan.{png,pdf}
  fig11_glorys_timeseries_oct.{png,pdf}

Each figure is a 5-row × N_loc-column grid:
  rows = variables [thetao, so, uo, vo, zos]
  cols = ocean locations (spatially spread, seed-reproducible)

Inputs:
    --jan_dir    (str): ensemble_ic_Jan_14012020 directory.
    --oct_dir    (str): ensemble_ic_Oct_14102020 directory.
    --mean_dir   (str): Directory with normalisation mean .npy files.
    --ocean_file (str): Path to ocean.nc (GLORYS daily reanalysis).
    --ostia_file (str): Path to ostia_2020.nc (optional; enables OSTIA overlay in SST row).
    --n_locs     (int): Number of random ocean locations (default 6).
    --seed       (int): Random seed for location selection (default 42).
    --output     (str): Output path stem (default results/figures/fig11_glorys_timeseries).
    --pdf        (flag): Also write PDF alongside PNG.
    --dpi        (int): Raster DPI (default 300).

Outputs:
    {output}_jan.png / {output}_jan.pdf
    {output}_oct.png / {output}_oct.pdf

Example:
    conda activate BoB_Surf_2
    python src/visualization/plot_ensemble_glorys_timeseries.py \\
        --jan_dir results/AFNO_BoB_Surf_E14/ensemble_ic_Jan_14012020 \\
        --oct_dir results/AFNO_BoB_Surf_E14/ensemble_ic_Oct_14102020 \\
        --mean_dir data/1993_2020/mean \\
        --ocean_file data/1993_2020/ocean.nc \\
        --ostia_file data/1993_2020/ostia_2020.nc \\
        --n_locs 6 --seed 42 --pdf
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import netCDF4 as nc
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GLORYS_LAT   = np.linspace(4.0, 23.0, 229)
GLORYS_LON   = np.linspace(77.0, 99.0, 265)
GLORYS_SHAPE = (229, 265)
MODEL_SHAPE  = (224, 224)

VARS       = ['thetao', 'so', 'uo', 'vo', 'zos']
VAR_LABELS = ['SST (°C)', 'SSS (PSU)', 'U (m/s)', 'V (m/s)', 'SSH (m)']

# Variables that need mean added back on denormalisation
_MEAN_VARS = {'thetao', 'so'}

# Fixed colour for OSTIA overlay (crimson — distinct from ensemble blue/orange and GLORYS black)
OSTIA_COLOR = '#C62828'
KELVIN      = 273.15


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_means(mean_dir: Path) -> dict:
    """Load normalisation means for thetao and so on the GLORYS 229×265 grid.

    Args:
        mean_dir (Path): Directory with mean_*_1993_2018_all_months.npy files.

    Returns:
        dict: Keys 'thetao' and 'so'; values are float32 arrays of shape (229, 265).
              Value is None if the file is missing.

    Example:
        >>> m = load_means(Path('data/1993_2020/mean'))
        >>> m['thetao'].shape
        (229, 265)
    """
    means = {}
    for var in ['thetao', 'so']:
        fpath = mean_dir / f'mean_{var}_1993_2018_all_months.npy'
        if fpath.exists():
            means[var] = np.squeeze(np.load(fpath).astype(np.float32))
        else:
            print(f'Warning: mean file not found: {fpath}', file=sys.stderr)
            means[var] = None
    return means


def load_glorys_slice(ocean_file: Path, start_day: int, n_days: int) -> np.ndarray:
    """Load a consecutive slice of GLORYS ocean data for all 5 variables.

    Args:
        ocean_file (Path): Path to ocean.nc.
        start_day  (int): 0-based day index (since 1993-01-01) of the first day to load.
        n_days     (int): Number of consecutive days to load.

    Returns:
        np.ndarray: Float32 array of shape (n_days, 5, 229, 265); NaN on land.
            Variable order: thetao (0), so (1), uo (2), vo (3), zos (4).

    Example:
        >>> gl = load_glorys_slice(Path('ocean.nc'), 9874, 10)
        >>> gl.shape
        (10, 5, 229, 265)
    """
    ds  = nc.Dataset(ocean_file)
    out = np.full((n_days, 5, 229, 265), np.nan, dtype=np.float32)

    for vi, vname in enumerate(VARS):
        raw = ds[vname][start_day:start_day + n_days]  # masked array
        arr = np.ma.filled(raw, fill_value=np.nan).astype(np.float32)
        if arr.ndim == 4:    # (time, depth, lat, lon) — thetao/so/uo/vo
            arr = arr[:, 0, :, :]
        # zos is 3-D (time, lat, lon) — arr already correct
        out[:, vi, :, :] = arr

    ds.close()
    return out


def load_ostia(ostia_file: Path) -> tuple:
    """Load OSTIA SST interpolated to the GLORYS 229×265 grid.

    Args:
        ostia_file (Path): Path to ostia_2020.nc.

    Returns:
        tuple:
            np.ndarray: SST in °C, shape (N_days, 229, 265).
            dict: Mapping from ISO date string ('YYYY-MM-DD') to day index.

    Example:
        >>> sst, idx = load_ostia(Path('ostia_2020.nc'))
        >>> sst.shape[1:]
        (229, 265)
    """
    ds   = xr.open_dataset(ostia_file)
    sst  = (ds['analysed_sst'] - KELVIN).interp(
        latitude=GLORYS_LAT, longitude=GLORYS_LON, method='linear'
    )
    dates = [str(t)[:10] for t in sst.time.values]
    arr   = sst.values.astype(np.float32)
    ds.close()
    return arr, {d: i for i, d in enumerate(dates)}


def pick_locations(thetao_mean: np.ndarray, n: int, seed: int) -> list:
    """Pick n random ocean pixels on the GLORYS grid, spatially spread.

    Divides the domain into n longitude bands and picks one random pixel per
    band so the selected points span the full longitudinal range.

    Args:
        thetao_mean (np.ndarray): Shape (229, 265); NaN = land.
        n    (int): Number of locations to select.
        seed (int): Random seed for reproducibility.

    Returns:
        list[tuple[int,int]]: List of (row, col) GLORYS grid indices.

    Example:
        >>> locs = pick_locations(mean, 6, 42)
        >>> len(locs)
        6
    """
    rng      = np.random.default_rng(seed)
    ocean_rc = np.argwhere(~np.isnan(thetao_mean))   # (N_ocean, 2)

    cols               = ocean_rc[:, 1]
    col_min, col_max   = cols.min(), cols.max()
    edges              = np.linspace(col_min, col_max + 1, n + 1)

    locs = []
    for i in range(n):
        band_mask = (cols >= edges[i]) & (cols < edges[i + 1])
        band_pts  = ocean_rc[band_mask]
        if len(band_pts) == 0:
            band_pts = ocean_rc
        idx = rng.integers(len(band_pts))
        locs.append((int(band_pts[idx, 0]), int(band_pts[idx, 1])))
    return locs


def sample_glorys_point(arr_224: np.ndarray, r_g: int, c_g: int) -> float:
    """Sample a 224×224 model-space field at a GLORYS grid point via bilinear.

    Maps GLORYS pixel (r_g, c_g) back to model coordinates and performs
    bilinear interpolation over the 2×2 neighbourhood.

    Args:
        arr_224 (np.ndarray): Float32 array of shape (224, 224).
        r_g (int): GLORYS row index (0-based, out of 229).
        c_g (int): GLORYS column index (0-based, out of 265).

    Returns:
        float: Bilinearly interpolated scalar value.

    Example:
        >>> v = sample_glorys_point(field, 100, 130)
    """
    r_m = (r_g + 0.5) * MODEL_SHAPE[0] / GLORYS_SHAPE[0] - 0.5
    c_m = (c_g + 0.5) * MODEL_SHAPE[1] / GLORYS_SHAPE[1] - 0.5
    r_m = np.clip(r_m, 0.0, MODEL_SHAPE[0] - 1.0)
    c_m = np.clip(c_m, 0.0, MODEL_SHAPE[1] - 1.0)

    r0, c0 = int(r_m), int(c_m)
    r1, c1 = min(r0 + 1, MODEL_SHAPE[0] - 1), min(c0 + 1, MODEL_SHAPE[1] - 1)
    dr, dc  = r_m - r0, c_m - c0

    return float(
        arr_224[r0, c0] * (1 - dr) * (1 - dc)
        + arr_224[r1, c0] * dr       * (1 - dc)
        + arr_224[r0, c1] * (1 - dr) * dc
        + arr_224[r1, c1] * dr       * dc
    )


def _denorm(norm_val: float, var_idx: int, means: dict, r_g: int, c_g: int) -> float:
    """Denormalise a single model output value to physical units.

    thetao and so were mean-subtracted; uo, vo, zos were not normalised.

    Args:
        norm_val (float): Normalised model output value.
        var_idx  (int):   Variable index (0=thetao, 1=so, 2=uo, 3=vo, 4=zos).
        means    (dict):  Mean arrays per variable (see load_means).
        r_g      (int):   GLORYS row index.
        c_g      (int):   GLORYS column index.

    Returns:
        float: Physical value in native units (°C / PSU / m s⁻¹ / m).

    Example:
        >>> _denorm(-0.5, 0, means, 100, 130)
        26.3
    """
    vname = VARS[var_idx]
    if vname in _MEAN_VARS and means.get(vname) is not None:
        return norm_val + float(means[vname][r_g, c_g])
    return norm_val


# ---------------------------------------------------------------------------
# Extract time series for all variables at one location
# ---------------------------------------------------------------------------

def extract_timeseries_all_vars(
    ens_mean:     np.ndarray,
    ens_std:      np.ndarray,
    ens_preds:    np.ndarray,
    means:        dict,
    glorys_slice: np.ndarray,
    r_g: int,
    c_g: int,
    ostia_sst:    np.ndarray = None,
    ostia_idx:    dict       = None,
    init_dt:      datetime   = None,
    lead_h:       list       = None,
) -> list:
    """Extract ensemble, GLORYS, and (optionally) OSTIA time series at one point.

    For each variable, identifies the ensemble member whose 9-day trajectory
    minimises MSE against GLORYS.  For thetao (SST) only, if OSTIA data is
    supplied, also identifies the OSTIA-closest member.

    Args:
        ens_mean     (np.ndarray): Shape (9, 5, 224, 224) normalised ensemble mean.
        ens_std      (np.ndarray): Shape (9, 5, 224, 224) normalised ensemble std.
        ens_preds    (np.ndarray): Shape (50, 9, 5, 224, 224) normalised member preds.
        means        (dict):       Normalisation means from load_means.
        glorys_slice (np.ndarray): Shape (10, 5, 229, 265); index 0 = IC day,
                                   indices 1-9 = lead days 1-9.
        r_g (int): GLORYS row index.
        c_g (int): GLORYS column index.
        ostia_sst  (np.ndarray): OSTIA SST in °C, shape (N_days, 229, 265). Optional.
        ostia_idx  (dict):       Date string → OSTIA day index. Optional.
        init_dt    (datetime):   IC datetime (needed to map lead hours to dates). Optional.
        lead_h     (list):       Lead times in hours, length 9. Optional.

    Returns:
        list[dict]: Length 5 (one per variable). Each dict has keys:
            'leads'              (np.ndarray): Lead day integers 1–9, shape (9,).
            'ens_mean'           (np.ndarray): Physical ensemble mean, shape (9,).
            'ens_lo'             (np.ndarray): Mean − 1σ in physical units, shape (9,).
            'ens_hi'             (np.ndarray): Mean + 1σ in physical units, shape (9,).
            'glorys'             (np.ndarray): GLORYS values, shape (9,); NaN=land.
            'best_member'        (np.ndarray): GLORYS-closest member trajectory, shape (9,).
            'best_idx'           (int):        0-based index of the GLORYS-closest member.
            'glorys_ic'          (float):      GLORYS value at IC day (lead 0).
            'ostia'              (np.ndarray): OSTIA SST in °C, shape (9,) — thetao only;
                                               None for other variables.
            'ostia_best_member'  (np.ndarray): OSTIA-closest member trajectory, shape (9,) —
                                               thetao only; None for other variables.
            'ostia_best_idx'     (int):        0-based index of the OSTIA-closest member;
                                               None for other variables.
            'ostia_ic'           (float):      OSTIA SST at IC date — thetao only; NaN if missing.

    Example:
        >>> ts_list = extract_timeseries_all_vars(em, es, ep, mns, gl, 100, 130)
        >>> len(ts_list)
        5
    """
    n_steps   = ens_mean.shape[0]     # 9
    n_members = ens_preds.shape[0]    # 50
    n_vars    = len(VARS)             # 5

    has_ostia = (ostia_sst is not None and ostia_idx is not None
                 and init_dt is not None and lead_h is not None)

    results = []
    for vi in range(n_vars):
        mu_arr, lo_arr, hi_arr, gl_arr = [], [], [], []

        for s in range(n_steps):
            mu_norm  = sample_glorys_point(ens_mean[s, vi], r_g, c_g)
            sig_norm = sample_glorys_point(ens_std[s,  vi], r_g, c_g)

            mu_phys  = _denorm(mu_norm,  vi, means, r_g, c_g)
            sig_phys = sig_norm   # std unchanged under mean subtraction

            mu_arr.append(mu_phys)
            lo_arr.append(mu_phys - sig_phys)
            hi_arr.append(mu_phys + sig_phys)

            gl_arr.append(float(glorys_slice[s + 1, vi, r_g, c_g]))

        glorys_arr = np.array(gl_arr)

        # Per-member physical trajectories (shared across GLORYS and OSTIA comparisons)
        member_ts = np.array([
            [_denorm(sample_glorys_point(ens_preds[m, s, vi], r_g, c_g),
                     vi, means, r_g, c_g)
             for s in range(n_steps)]
            for m in range(n_members)
        ])  # (50, 9)

        # Best member vs GLORYS
        valid_gl = np.isfinite(glorys_arr)
        if valid_gl.sum() >= 1:
            mse_gl   = np.mean((member_ts[:, valid_gl] - glorys_arr[valid_gl]) ** 2, axis=1)
            best_idx = int(np.argmin(mse_gl))
        else:
            best_idx = 0

        glorys_ic = float(glorys_slice[0, vi, r_g, c_g])

        # OSTIA overlay — thetao (SST) only
        ostia_arr          = None
        ostia_best_member  = None
        ostia_best_idx     = None
        ostia_ic_val       = np.nan

        if has_ostia and vi == 0:
            ost_vals = []
            for s in range(n_steps):
                valid_dt = init_dt + timedelta(hours=int(lead_h[s]))
                dk       = valid_dt.strftime('%Y-%m-%d')
                ost_vals.append(
                    float(ostia_sst[ostia_idx[dk], r_g, c_g]) if dk in ostia_idx else np.nan
                )
            ostia_arr  = np.array(ost_vals)

            valid_ost = np.isfinite(ostia_arr)
            if valid_ost.sum() >= 1:
                mse_ost        = np.mean((member_ts[:, valid_ost] - ostia_arr[valid_ost]) ** 2, axis=1)
                ostia_best_idx = int(np.argmin(mse_ost))
            else:
                ostia_best_idx = 0
            ostia_best_member = member_ts[ostia_best_idx]

            ic_dk        = init_dt.strftime('%Y-%m-%d')
            ostia_ic_val = float(ostia_sst[ostia_idx[ic_dk], r_g, c_g]) if ic_dk in ostia_idx else np.nan

        results.append({
            'leads':             np.arange(1, 10),
            'ens_mean':          np.array(mu_arr),
            'ens_lo':            np.array(lo_arr),
            'ens_hi':            np.array(hi_arr),
            'glorys':            glorys_arr,
            'best_member':       member_ts[best_idx],
            'best_idx':          best_idx,
            'glorys_ic':         glorys_ic,
            'ostia':             ostia_arr,
            'ostia_best_member': ostia_best_member,
            'ostia_best_idx':    ostia_best_idx,
            'ostia_ic':          ostia_ic_val,
        })

    return results


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def build_figure(
    all_locs_vars: list,
    locations:     list,
    season_label:  str,
    color:         str,
    has_ostia:     bool = False,
) -> plt.Figure:
    """Build the 5-variable × N_loc multi-panel time-series figure.

    Layout: rows = variables (SST/SSS/U/V/SSH), columns = ocean locations.
    When has_ostia is True, the SST row (vi=0) additionally shows OSTIA
    observations as crimson dots and the OSTIA-closest member as a crimson
    dotted line.

    Args:
        all_locs_vars (list[list[dict]]): Outer list over locations; inner list
            over variables. all_locs_vars[i][vi] is the timeseries dict for
            location i and variable vi.
        locations (list[tuple[int,int]]): (row, col) GLORYS indices per location.
        season_label (str): Season string for the figure super-title.
        color (str): Hex colour for the ensemble band and lines.
        has_ostia (bool): Whether OSTIA data is present in the timeseries dicts.

    Returns:
        plt.Figure: Assembled matplotlib figure.

    Example:
        >>> fig = build_figure(all_ts, locs, 'Winter (IC: 14 Jan 2020)', '#1565C0', has_ostia=True)
        >>> fig.savefig('out.png', dpi=300, bbox_inches='tight')
    """
    n_locs = len(locations)
    n_vars = len(VARS)

    plt.rcParams.update({'font.size': 12})
    band_alpha = 0.22

    fig, axes = plt.subplots(
        n_vars, n_locs,
        figsize=(3.0 * n_locs, 2.6 * n_vars),
        sharex=True,
        squeeze=False,
    )

    for vi in range(n_vars):
        for i, (r_g, c_g) in enumerate(locations):
            ax    = axes[vi, i]
            ts    = all_locs_vars[i][vi]
            leads = ts['leads']

            # Ensemble band and mean
            ax.fill_between(leads, ts['ens_lo'], ts['ens_hi'],
                            color=color, alpha=band_alpha, linewidth=0)
            ax.plot(leads, ts['ens_mean'], color=color, lw=1.4)

            # GLORYS-closest member (dotted, ensemble colour)
            ax.plot(leads, ts['best_member'], color=color, lw=0.9, ls=':')

            # GLORYS truth dots (black)
            valid_gl = np.isfinite(ts['glorys'])
            if valid_gl.any():
                ax.scatter(leads[valid_gl], ts['glorys'][valid_gl],
                           color='k', s=18, zorder=5, marker='o')

            # GLORYS IC hollow dot at lead 0
            if np.isfinite(ts['glorys_ic']):
                ax.scatter([0], [ts['glorys_ic']],
                           facecolors='none', edgecolors=color,
                           s=24, lw=1.2, zorder=6)

            # OSTIA overlay — SST row only
            if has_ostia and vi == 0:
                if ts['ostia'] is not None:
                    valid_ost = np.isfinite(ts['ostia'])
                    if valid_ost.any():
                        ax.scatter(leads[valid_ost], ts['ostia'][valid_ost],
                                   color=OSTIA_COLOR, s=18, zorder=5, marker='o')
                if ts['ostia_best_member'] is not None:
                    ax.plot(leads, ts['ostia_best_member'],
                            color=OSTIA_COLOR, lw=0.9, ls=':')
                # OSTIA IC hollow crimson dot at lead 0
                if np.isfinite(ts['ostia_ic']):
                    ax.scatter([0], [ts['ostia_ic']],
                               facecolors='none', edgecolors=OSTIA_COLOR,
                               s=24, lw=1.2, zorder=6)

            ax.set_xlim(-0.4, 9.4)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            ax.grid(axis='y', alpha=0.25, lw=0.4)
            ax.tick_params(labelsize=10)

            # Column header: lat/lon (top row only)
            if vi == 0:
                lat = GLORYS_LAT[r_g]
                lon = GLORYS_LON[c_g]
                ax.set_title(f'{lat:.1f}°N\n{lon:.1f}°E',
                             fontsize=12, fontweight='bold', pad=3)

            # Row label: variable unit (left column only)
            if i == 0:
                ax.set_ylabel(VAR_LABELS[vi], fontsize=12, labelpad=3)

            # x-label (bottom row only)
            if vi == n_vars - 1:
                ax.set_xlabel('Lead day', fontsize=10)

    # Shared legend — includes OSTIA entries only when present
    legend_elements = [
        Patch(facecolor=color, alpha=band_alpha + 0.15, label='Ens. mean ± 1σ'),
        Line2D([0], [0], color=color, lw=1.4,         label='Ens. mean'),
        Line2D([0], [0], color='k',   lw=0, marker='o', markersize=5, label='GLORYS'),
        Line2D([0], [0], color=color, lw=0.9, ls=':',
               label='Best member (min MSE vs GLORYS)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor=color, markeredgewidth=1.2,
               markersize=5, label='GLORYS IC'),
    ]
    if has_ostia:
        legend_elements += [
            Line2D([0], [0], color=OSTIA_COLOR, lw=0, marker='o',
                   markersize=5, label='OSTIA (SST row)'),
            Line2D([0], [0], color=OSTIA_COLOR, lw=0.9, ls=':',
                   label='Best member (min MSE vs OSTIA, SST row)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor=OSTIA_COLOR, markeredgewidth=1.2,
                   markersize=5, label='OSTIA IC (SST row)'),
        ]

    ncol = 4 if not has_ostia else 4
    fig.legend(handles=legend_elements, loc='lower center', ncol=ncol,
               fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'Ensemble vs GLORYS — {season_label}',
                 fontsize=15, fontweight='bold', y=1.005)
    fig.tight_layout(rect=[0, 0.05, 1, 1.0])
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments.

    Example:
        >>> args = _parse_args()
    """
    p = argparse.ArgumentParser(
        description='Ensemble vs GLORYS time series for all ocean variables'
    )
    p.add_argument('--jan_dir',    required=True,
                   help='ensemble_ic_Jan_14012020 directory')
    p.add_argument('--oct_dir',    required=True,
                   help='ensemble_ic_Oct_14102020 directory')
    p.add_argument('--apr_dir',    default=None,
                   help='ensemble_ic_Apr_14042020 directory (optional)')
    p.add_argument('--jun_dir',    default=None,
                   help='ensemble_ic_Jun_14062020 directory (optional)')
    p.add_argument('--mean_dir',   default='data/1993_2020/mean')
    p.add_argument('--ocean_file', default='data/1993_2020/ocean.nc')
    p.add_argument('--ostia_file', default='data/1993_2020/ostia_2020.nc',
                   help='Path to ostia_2020.nc for SST overlay (optional)')
    p.add_argument('--n_locs',  type=int, default=6)
    p.add_argument('--seed',    type=int, default=42)
    p.add_argument('--output',  default='results/figures/fig11_glorys_timeseries')
    p.add_argument('--pdf',     action='store_true')
    p.add_argument('--dpi',     type=int, default=300)
    return p.parse_args()


def main():
    """Load ensembles and GLORYS reanalysis, pick locations, build and save figures.

    Args:
        None: All settings from sys.argv (see module docstring).

    Returns:
        None: Writes PNG (and optional PDF) for Jan and Oct seasons.

    Example:
        >>> # python src/visualization/plot_ensemble_glorys_timeseries.py \\
        >>> #     --jan_dir results/.../ensemble_ic_Jan_14012020 \\
        >>> #     --oct_dir results/.../ensemble_ic_Oct_14102020 \\
        >>> #     --pdf
    """
    args = _parse_args()

    # Load normalisation means and pick shared locations
    means       = load_means(Path(args.mean_dir))
    thetao_mean = means['thetao']

    locs = pick_locations(thetao_mean, args.n_locs, args.seed)
    print(f'Selected {len(locs)} locations:')
    for r_g, c_g in locs:
        print(f'  ({GLORYS_LAT[r_g]:.2f}°N, {GLORYS_LON[c_g]:.2f}°E)')

    # Load OSTIA if available
    ostia_sst, ostia_idx = None, None
    ostia_path = Path(args.ostia_file)
    if ostia_path.exists():
        print(f'Loading OSTIA from {ostia_path} ...', flush=True)
        ostia_sst, ostia_idx = load_ostia(ostia_path)
    else:
        print(f'OSTIA file not found ({ostia_path}); skipping OSTIA overlay.', flush=True)

    has_ostia = ostia_sst is not None

    def day_index(dt: datetime) -> int:
        """Days since 1993-01-01."""
        return (dt - datetime(1993, 1, 1)).days

    seasons = [
        dict(
            label='Winter (IC: 14 Jan 2020)',
            dir=args.jan_dir,
            color='#1565C0',
            suffix='jan',
        ),
        dict(
            label='Post-monsoon (IC: 14 Oct 2020)',
            dir=args.oct_dir,
            color='#E65100',
            suffix='oct',
        ),
    ]
    if args.apr_dir:
        seasons.append(dict(
            label='Pre-monsoon (IC: 14 Apr 2020)',
            dir=args.apr_dir,
            color='#1B5E20',
            suffix='apr',
        ))
    if args.jun_dir:
        seasons.append(dict(
            label='Monsoon (IC: 14 Jun 2020)',
            dir=args.jun_dir,
            color='#B71C1C',
            suffix='jun',
        ))

    for s in seasons:
        ens_dir = Path(s['dir'])
        meta    = json.load(open(ens_dir / 'ensemble_metadata.json'))
        init_dt = datetime.strptime(meta['input_date'], '%d-%m-%Y')
        lead_h  = meta['lead_times_h']

        print(f"\nProcessing {s['label']} ...", flush=True)

        ens_mean  = np.load(ens_dir / 'ensemble_mean.npy')          # (9, 5, 224, 224)
        ens_std   = np.load(ens_dir / 'ensemble_std.npy')           # (9, 5, 224, 224)
        ens_preds = np.load(ens_dir / 'ensemble_predictions.npy')   # (50, 9, 5, 224, 224)
        print(f'  Loaded {ens_preds.shape[0]} members', flush=True)

        ic_day       = day_index(init_dt)
        print(f'  Loading GLORYS slice: day {ic_day} + 9 leads ...', flush=True)
        glorys_slice = load_glorys_slice(Path(args.ocean_file), ic_day, 10)

        print('  Extracting time series ...', flush=True)
        all_locs_vars = [
            extract_timeseries_all_vars(
                ens_mean, ens_std, ens_preds, means, glorys_slice, r_g, c_g,
                ostia_sst=ostia_sst, ostia_idx=ostia_idx,
                init_dt=init_dt, lead_h=lead_h,
            )
            for r_g, c_g in locs
        ]

        fig = build_figure(all_locs_vars, locs, s['label'], s['color'],
                           has_ostia=has_ostia)

        out_stem = f"{args.output}_{s['suffix']}"
        for ext in (['png', 'pdf'] if args.pdf else ['png']):
            out_path = f'{out_stem}.{ext}'
            fig.savefig(out_path,
                        dpi=args.dpi if ext == 'png' else None,
                        bbox_inches='tight')
            print(f'Saved: {out_path}')
        plt.close(fig)


if __name__ == '__main__':
    main()

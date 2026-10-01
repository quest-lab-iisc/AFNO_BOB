"""Time-series comparison of ensemble SST vs OSTIA at random ocean locations.

For each randomly selected ocean pixel, plots the 9-day ensemble mean ± 1 std
as a shaded band, overlaid with OSTIA L4 SST observations as filled dots.
A hollow dot at lead day 0 shows the GLORYS initial condition.

Produces two figures (one per season):
  fig10_ostia_timeseries_jan.{png,pdf}
  fig10_ostia_timeseries_oct.{png,pdf}

Each figure is an N_loc × 1 column of panels (one per location), or arranged
in a 2-column grid if N_loc > 4.

Inputs:
    --jan_dir  (str): ensemble_ic_Jan_14012020 directory.
    --oct_dir  (str): ensemble_ic_Oct_14102020 directory.
    --mean_dir (str): Directory with normalisation mean .npy files.
    --ostia_file (str): Path to ostia_2020.nc.
    --ocean_file (str): Path to ocean.nc (for GLORYS IC SST at lead 0).
    --n_locs   (int): Number of random ocean locations (default 6).
    --seed     (int): Random seed for location selection (default 42).
    --output   (str): Output path stem (default results/figures/fig10_ostia_timeseries).
    --pdf      (flag): Also write PDF alongside PNG.
    --dpi      (int): Raster DPI (default 300).

Outputs:
    {output}_jan.png / {output}_jan.pdf
    {output}_oct.png / {output}_oct.pdf

Example:
    conda activate BoB_Surf_2
    python src/visualization/plot_ensemble_ostia_timeseries.py \\
        --jan_dir results/AFNO_BoB_Surf_E14/ensemble_ic_Jan_14012020 \\
        --oct_dir results/AFNO_BoB_Surf_E14/ensemble_ic_Oct_14102020 \\
        --mean_dir data/1993_2020/mean \\
        --ostia_file data/1993_2020/ostia_2020.nc \\
        --ocean_file data/1993_2020/ocean.nc \\
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
import netCDF4 as nc
import xarray as xr
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GLORYS_LAT   = np.linspace(4.0, 23.0, 229)
GLORYS_LON   = np.linspace(77.0, 99.0, 265)
GLORYS_SHAPE = (229, 265)
MODEL_SHAPE  = (224, 224)
KELVIN       = 273.15


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_thetao_mean(mean_dir: Path) -> np.ndarray:
    """Load climatological thetao mean on GLORYS grid.

    Args:
        mean_dir (Path): Directory with mean_thetao_1993_2018_all_months.npy.

    Returns:
        np.ndarray: Float32 array of shape (229, 265); NaN on land.

    Example:
        >>> m = load_thetao_mean(Path('data/1993_2020/mean'))
        >>> m.shape
        (229, 265)
    """
    return np.squeeze(
        np.load(mean_dir / 'mean_thetao_1993_2018_all_months.npy').astype(np.float32)
    )


def load_ostia(ostia_file: Path) -> tuple[np.ndarray, dict]:
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


def load_glorys_sst(ocean_file: Path, day_index: int) -> np.ndarray:
    """Load GLORYS thetao for one day on the native 229×265 grid.

    Args:
        ocean_file (Path): Path to ocean.nc.
        day_index (int): 0-based day index since 1993-01-01.

    Returns:
        np.ndarray: Float32 array of shape (229, 265); NaN on land.

    Example:
        >>> sst = load_glorys_sst(Path('ocean.nc'), 9874)
        >>> sst.shape
        (229, 265)
    """
    ds  = nc.Dataset(ocean_file)
    arr = np.ma.filled(
        np.ma.array(ds['thetao'][day_index, 0, :, :]), fill_value=np.nan
    ).astype(np.float32)
    ds.close()
    return arr


def sample_glorys_point(arr_224: np.ndarray, r_g: int, c_g: int) -> float:
    """Sample a (224×224) model-space field at a GLORYS grid point via bilinear.

    Maps GLORYS pixel (r_g, c_g) back to model coords and samples bilinearly.

    Args:
        arr_224 (np.ndarray): Float32 array of shape (224, 224).
        r_g (int): GLORYS row index (0-based, out of 229).
        c_g (int): GLORYS column index (0-based, out of 265).

    Returns:
        float: Interpolated scalar value.

    Example:
        >>> v = sample_glorys_point(field, 100, 130)
    """
    # GLORYS row r_g maps to model coordinate via the bilinear interp formula
    r_m = (r_g + 0.5) * MODEL_SHAPE[0] / GLORYS_SHAPE[0] - 0.5
    c_m = (c_g + 0.5) * MODEL_SHAPE[1] / GLORYS_SHAPE[1] - 0.5

    # Clamp to valid range
    r_m = np.clip(r_m, 0.0, MODEL_SHAPE[0] - 1.0)
    c_m = np.clip(c_m, 0.0, MODEL_SHAPE[1] - 1.0)

    r0, c0 = int(r_m), int(c_m)
    r1, c1 = min(r0 + 1, MODEL_SHAPE[0] - 1), min(c0 + 1, MODEL_SHAPE[1] - 1)
    dr, dc  = r_m - r0, c_m - c0

    return float(
        arr_224[r0, c0] * (1 - dr) * (1 - dc)
        + arr_224[r1, c0] * dr * (1 - dc)
        + arr_224[r0, c1] * (1 - dr) * dc
        + arr_224[r1, c1] * dr * dc
    )


def pick_locations(thetao_mean: np.ndarray, n: int, seed: int) -> list[tuple[int,int]]:
    """Pick n random ocean pixels on the GLORYS grid, spatially spread.

    Divides the domain into n vertical bands and picks one random pixel per
    band so locations span the full longitudinal range.

    Args:
        thetao_mean (np.ndarray): Shape (229, 265); NaN = land.
        n (int): Number of locations to select.
        seed (int): Random seed.

    Returns:
        list[tuple[int,int]]: List of (row, col) GLORYS grid indices.

    Example:
        >>> locs = pick_locations(mean, 6, 42)
        >>> len(locs)
        6
    """
    rng = np.random.default_rng(seed)
    ocean_rc = np.argwhere(~np.isnan(thetao_mean))   # (N_ocean, 2)

    # Divide ocean pixels into n longitude bands
    cols  = ocean_rc[:, 1]
    col_min, col_max = cols.min(), cols.max()
    edges = np.linspace(col_min, col_max + 1, n + 1)

    locs = []
    for i in range(n):
        band_mask = (cols >= edges[i]) & (cols < edges[i + 1])
        band_pts  = ocean_rc[band_mask]
        if len(band_pts) == 0:
            band_pts = ocean_rc
        idx = rng.integers(len(band_pts))
        locs.append((int(band_pts[idx, 0]), int(band_pts[idx, 1])))
    return locs


# ---------------------------------------------------------------------------
# Extract time series at one location
# ---------------------------------------------------------------------------

def extract_timeseries(
    ens_mean:    np.ndarray,
    ens_std:     np.ndarray,
    ens_preds:   np.ndarray,
    thetao_mean: np.ndarray,
    ostia_sst:   np.ndarray,
    ostia_idx:   dict,
    init_dt:     datetime,
    lead_h:      list,
    r_g: int,
    c_g: int,
) -> dict:
    """Extract ensemble and OSTIA SST time series at one GLORYS grid point.

    Also finds the ensemble member whose 9-day trajectory minimises MSE
    against the OSTIA observations at this location.

    Args:
        ens_mean  (np.ndarray): Shape (9, 5, 224, 224) normalised ensemble mean.
        ens_std   (np.ndarray): Shape (9, 5, 224, 224) normalised ensemble std.
        ens_preds (np.ndarray): Shape (50, 9, 5, 224, 224) normalised member preds.
        thetao_mean (np.ndarray): Climatological mean, shape (229, 265).
        ostia_sst (np.ndarray): OSTIA SST in °C, shape (N_days, 229, 265).
        ostia_idx (dict): Date string → day index.
        init_dt (datetime): IC datetime.
        lead_h (list): Lead times in hours, length 9.
        r_g (int): GLORYS row index.
        c_g (int): GLORYS column index.

    Returns:
        dict: Keys:
            'leads'       (np.ndarray): Lead day integers 1–9.
            'ens_mean'    (np.ndarray): Ensemble mean SST in °C, shape (9,).
            'ens_lo'      (np.ndarray): Mean - 1 std in °C, shape (9,).
            'ens_hi'      (np.ndarray): Mean + 1 std in °C, shape (9,).
            'ostia'       (np.ndarray): OSTIA SST in °C, shape (9,); NaN if missing.
            'best_member' (np.ndarray): Best-MSE member trajectory in °C, shape (9,).
            'best_idx'    (int): 0-based index of the best member.

    Example:
        >>> ts = extract_timeseries(em, es, ep, tm, osst, oidx, dt, lh, 100, 130)
        >>> ts['best_member'].shape
        (9,)
    """
    clim = float(thetao_mean[r_g, c_g])
    n_steps = len(lead_h)
    n_members = ens_preds.shape[0]

    means, los, his, ostias = [], [], [], []
    for s in range(n_steps):
        mu_norm  = sample_glorys_point(ens_mean[s, 0], r_g, c_g)
        sig_norm = sample_glorys_point(ens_std[s,  0], r_g, c_g)

        means.append(mu_norm + clim)
        los.append(mu_norm - sig_norm + clim)
        his.append(mu_norm + sig_norm + clim)

        valid_dt = init_dt + timedelta(hours=int(lead_h[s]))
        dk = valid_dt.strftime('%Y-%m-%d')
        ostias.append(
            float(ostia_sst[ostia_idx[dk], r_g, c_g]) if dk in ostia_idx else np.nan
        )

    ostia_arr = np.array(ostias)

    # --- Per-member trajectories at this location ---
    # Shape (n_members, n_steps): sample each member's normalized thetao, then denorm
    member_sst = np.array([
        [sample_glorys_point(ens_preds[m, s, 0], r_g, c_g) + clim
         for s in range(n_steps)]
        for m in range(n_members)
    ])  # (50, 9)

    # Best member: minimise MSE over steps where OSTIA is available
    valid = np.isfinite(ostia_arr)
    if valid.sum() >= 1:
        mse = np.mean((member_sst[:, valid] - ostia_arr[valid]) ** 2, axis=1)
        best_idx = int(np.argmin(mse))
    else:
        best_idx = 0   # fallback: no OSTIA data, just pick member 0

    return {
        'leads':       np.arange(1, 10),
        'ens_mean':    np.array(means),
        'ens_lo':      np.array(los),
        'ens_hi':      np.array(his),
        'ostia':       ostia_arr,
        'best_member': member_sst[best_idx],
        'best_idx':    best_idx,
    }


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def build_figure(
    timeseries_list: list[dict],
    locations:       list[tuple[int,int]],
    ic_ssts:         list[float],
    ic_ostias:       list[float],
    season_label:    str,
    ic_date_str:     str,
    color:           str,
) -> plt.Figure:
    """Build the multi-panel time-series figure for one season.

    Args:
        timeseries_list (list[dict]): One dict per location (from extract_timeseries).
        locations (list[tuple[int,int]]): (row, col) GLORYS indices for each location.
        ic_ssts (list[float]): GLORYS IC SST at each location (°C).
        ic_ostias (list[float]): OSTIA SST at IC date for each location (°C).
        season_label (str): Label for the figure title, e.g. 'Winter (IC: 14 Jan 2020)'.
        ic_date_str (str): IC date as 'YYYY-MM-DD' for subtitle annotation.
        color (str): Hex colour for ensemble band.

    Returns:
        plt.Figure: Assembled matplotlib figure.

    Example:
        >>> fig = build_figure(ts_list, locs, ic_ssts, ic_osts, 'Winter', '2020-01-14', '#1565C0')
        >>> fig.savefig('out.png', dpi=300)
    """
    n = len(timeseries_list)
    ncols = 2 if n > 3 else 1
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 2.8 * nrows),
                             sharex=True)
    axes = np.atleast_1d(axes).flatten()

    plt.rcParams.update({'font.size': 8})
    band_alpha = 0.25

    for i, (ts, (r_g, c_g)) in enumerate(zip(timeseries_list, locations)):
        ax = axes[i]
        lat = GLORYS_LAT[r_g]
        lon = GLORYS_LON[c_g]

        leads = ts['leads']

        # --- Ensemble band ---
        ax.fill_between(leads, ts['ens_lo'], ts['ens_hi'],
                        color=color, alpha=band_alpha, linewidth=0)
        ax.plot(leads, ts['ens_mean'], color=color, lw=1.5, label='Ens. mean')

        # --- Best member trajectory ---
        ax.plot(leads, ts['best_member'], color=color, lw=1.0,
                ls=':', label=f"Best member (#{ts['best_idx']+1})")

        # --- OSTIA observations ---
        ost = ts['ostia']
        valid = np.isfinite(ost)
        if valid.any():
            ax.scatter(leads[valid], ost[valid],
                       color='k', s=22, zorder=5, label='OSTIA', marker='o')

        # --- IC anchor at lead 0 (GLORYS IC, hollow; OSTIA IC, filled grey) ---
        if np.isfinite(ic_ssts[i]):
            ax.scatter([0], [ic_ssts[i]], facecolors='none', edgecolors=color,
                       s=28, lw=1.2, zorder=6)
        if np.isfinite(ic_ostias[i]):
            ax.scatter([0], [ic_ostias[i]], color='grey', s=22, zorder=5,
                       marker='o')

        ax.set_xlim(-0.3, 9.3)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.grid(axis='y', alpha=0.3, lw=0.5)
        ax.tick_params(labelsize=7)
        ax.set_ylabel('SST (°C)', fontsize=7)
        ax.set_title(f'{lat:.1f}°N, {lon:.1f}°E', fontsize=8, fontweight='bold')

    # x-label on bottom row
    for ax in axes[-(ncols):]:
        ax.set_xlabel('Lead day', fontsize=7)

    # Hide unused panels
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, alpha=band_alpha + 0.15, label='Ens. mean ± 1σ'),
        Line2D([0], [0], color=color, lw=1.5, label='Ens. mean'),
        Line2D([0], [0], color=color, lw=1.0, ls=':', label='Best member (min MSE vs OSTIA)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k',
               markersize=5, label='OSTIA'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
               markersize=5, label='OSTIA (IC day)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor=color, markeredgewidth=1.2,
               markersize=6, label='GLORYS IC'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               fontsize=7, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(f'Ensemble SST vs OSTIA — {season_label}',
                 fontsize=9, fontweight='bold')
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
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
        description='Ensemble SST vs OSTIA time series at random ocean locations'
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
    p.add_argument('--ostia_file', default='data/1993_2020/ostia_2020.nc')
    p.add_argument('--ocean_file', default='data/1993_2020/ocean.nc')
    p.add_argument('--n_locs',  type=int, default=6)
    p.add_argument('--seed',    type=int, default=42)
    p.add_argument('--output',  default='results/figures/fig10_ostia_timeseries')
    p.add_argument('--pdf',     action='store_true')
    p.add_argument('--dpi',     type=int, default=300)
    return p.parse_args()


def main():
    """Load ensembles and OSTIA, pick random locations, build and save figures.

    Args:
        None: All settings from sys.argv (see module docstring).

    Returns:
        None: Writes PNG (and optional PDF) for Jan and Oct seasons.

    Example:
        >>> # python src/visualization/plot_ensemble_ostia_timeseries.py \\
        >>> #     --jan_dir results/.../ensemble_ic_Jan_14012020 \\
        >>> #     --oct_dir results/.../ensemble_ic_Oct_14102020 \\
        >>> #     --pdf
    """
    args = _parse_args()

    thetao_mean = load_thetao_mean(Path(args.mean_dir))
    ocean_mask  = ~np.isnan(thetao_mean)

    print('Loading OSTIA ...', flush=True)
    ostia_sst, ostia_idx = load_ostia(Path(args.ostia_file))

    # Pick shared locations for both seasons
    locs = pick_locations(thetao_mean, args.n_locs, args.seed)
    print(f'Selected {len(locs)} locations:')
    for r_g, c_g in locs:
        print(f'  ({GLORYS_LAT[r_g]:.2f}°N, {GLORYS_LON[c_g]:.2f}°E)')

    # Helper: day index from 1993-01-01
    def day_index(dt: datetime) -> int:
        """Compute days since 1993-01-01."""
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

        ens_mean  = np.load(ens_dir / 'ensemble_mean.npy')         # (9, 5, 224, 224)
        ens_std   = np.load(ens_dir / 'ensemble_std.npy')          # (9, 5, 224, 224)
        ens_preds = np.load(ens_dir / 'ensemble_predictions.npy')  # (50, 9, 5, 224, 224)
        print(f'  Loaded ensemble: {ens_preds.shape[0]} members', flush=True)

        # GLORYS IC SST at each location
        ic_day   = day_index(init_dt)
        glorys_ic = load_glorys_sst(Path(args.ocean_file), ic_day)

        ic_ssts   = [float(glorys_ic[r_g, c_g]) for r_g, c_g in locs]

        # OSTIA on IC date
        ic_date_str = init_dt.strftime('%Y-%m-%d')
        if ic_date_str in ostia_idx:
            ic_ostias = [float(ostia_sst[ostia_idx[ic_date_str], r_g, c_g])
                         for r_g, c_g in locs]
        else:
            ic_ostias = [np.nan] * len(locs)

        # Extract time series at each location
        ts_list = [
            extract_timeseries(
                ens_mean, ens_std, ens_preds, thetao_mean,
                ostia_sst, ostia_idx,
                init_dt, lead_h, r_g, c_g
            )
            for r_g, c_g in locs
        ]

        fig = build_figure(ts_list, locs, ic_ssts, ic_ostias,
                           s['label'], ic_date_str, s['color'])

        out = f"{args.output}_{s['suffix']}"
        for ext in (['png', 'pdf'] if args.pdf else ['png']):
            p = f'{out}.{ext}'
            fig.savefig(p, dpi=args.dpi if ext == 'png' else None,
                        bbox_inches='tight')
            print(f'Saved: {p}')
        plt.close(fig)


if __name__ == '__main__':
    main()

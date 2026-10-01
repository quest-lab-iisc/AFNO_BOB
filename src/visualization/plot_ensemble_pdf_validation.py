"""Ensemble PDF validation against ARGO float observations (fig7).

For each ARGO float that profiles within the 9-day forecast window (lead days 1–9),
the 50-member ensemble is extracted at the float's nearest model grid point and
displayed as a horizontal violin (PDF).  The actual ARGO observation is overlaid
as a red star.

Layout (one figure per season):
  • 9 rows — lead days 1–9.  Row height is proportional to the number of floats
    that profiled on that day so space is not wasted on sparse days.
  • 2 columns — SST (°C) and SSS (PSU).
  • Each panel: one horizontal violin per float, y-axis labelled with the last
    four digits of the platform ID and its latitude.
  • Red star = ARGO observation; no star = NaN / not available for that variable.
  • Ensemble median shown as a coloured tick inside each violin.

Outputs two figures:
  {output}_jan.png / .pdf  — Winter 2020 (IC 14 Jan 2020)
  {output}_oct.png / .pdf  — Post-monsoon 2020 (IC 14 Oct 2020)

Inputs:
    --jan_dir (str)    : Directory with Winter ensemble_predictions.npy.
    --oct_dir (str)    : Directory with PostMonsoon ensemble_predictions.npy.
    --argo_jan (str)   : Directory with per-day ARGO _prof.nc files, Jan 2020.
    --argo_oct (str)   : Directory with per-day ARGO _prof.nc files, Oct 2020.
    --mean_dir (str)   : Directory with mean_{var}_1993_2018_all_months.npy.
    --ocean_file (str) : GLORYS ocean.nc (for 224×224 lat/lon grid).
    --output (str)     : Output file stem; '_jan' / '_oct' appended automatically.
    --surf_pres (float): Max pressure (dbar) for surface layer (default 10).
    --pdf              : Also save PDF versions.

Outputs:
    {output}_jan.png / .pdf
    {output}_oct.png / .pdf

Example:
    conda activate BoB_Surf_2
    python src/visualization/plot_ensemble_pdf_validation.py \\
        --jan_dir results/AFNO_BoB_Surf_E14/ensemble_ic_Jan_14012020 \\
        --oct_dir results/AFNO_BoB_Surf_E14/ensemble_ic_Oct_14102020 \\
        --argo_jan data/argo/jan2020 --argo_oct data/argo/oct2020 \\
        --mean_dir data/1993_2020/mean \\
        --ocean_file data/1993_2020/ocean.nc \\
        --output results/figures/fig7_ensemble_pdf --pdf
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

_BOB_LAT    = (4.0, 23.0)
_BOB_LON    = (77.0, 99.0)
_MODEL_H, _MODEL_W = 224, 224
_OCEAN_VARS = ['thetao', 'so', 'uo', 'vo', 'zos']
_VAR_IDX    = {v: i for i, v in enumerate(_OCEAN_VARS)}

_COL_JAN = '#2166ac'
_COL_OCT = '#d6604d'
_COL_APR = '#1B5E20'
_COL_JUN = '#B71C1C'

# Variable → (argo key in profile dict, x-axis label)
_VAR_SPEC = [
    ('thetao', 'sst', 'SST (°C)'),
    ('so',     'sss', 'SSS (PSU)'),
]


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def load_model_grid(ocean_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Build 224×224 model lat/lon arrays by bilinear interpolation from GLORYS.

    Args:
        ocean_file (Path): GLORYS ocean.nc with latitude/longitude coordinates.

    Returns:
        tuple: lats (224,), lons (224,) float32.

    Example:
        >>> lats, lons = load_model_grid(Path('ocean.nc'))
    """
    ds = xr.open_dataset(ocean_file)
    lats_g = ds['latitude'].values
    lons_g = ds['longitude'].values
    ds.close()
    row_pos = (np.arange(_MODEL_H) + 0.5) * (len(lats_g) / _MODEL_H) - 0.5
    col_pos = (np.arange(_MODEL_W) + 0.5) * (len(lons_g) / _MODEL_W) - 0.5
    return (np.interp(row_pos, np.arange(len(lats_g)), lats_g).astype(np.float32),
            np.interp(col_pos, np.arange(len(lons_g)), lons_g).astype(np.float32))


def nearest_grid_point(lat: float, lon: float,
                       model_lats: np.ndarray,
                       model_lons: np.ndarray) -> tuple[int, int]:
    """Return the (row, col) of the nearest 224×224 model grid cell.

    Args:
        lat (float): Target latitude °N.
        lon (float): Target longitude °E.
        model_lats (np.ndarray): 1-D model latitude array.
        model_lons (np.ndarray): 1-D model longitude array.

    Returns:
        tuple: (row, col) integer indices.

    Example:
        >>> row, col = nearest_grid_point(12.5, 85.0, lats, lons)
    """
    return (int(np.argmin(np.abs(model_lats - lat))),
            int(np.argmin(np.abs(model_lons - lon))))


# ---------------------------------------------------------------------------
# Climatological mean
# ---------------------------------------------------------------------------

def load_clim_mean_224(mean_dir: Path, var: str) -> np.ndarray:
    """Load and resize climatological mean to 224×224 (NaN on land).

    Args:
        mean_dir (Path): Directory with mean_{var}_1993_2018_all_months.npy.
        var (str): Variable name ('thetao' or 'so').

    Returns:
        np.ndarray: (224, 224) float32, NaN on land pixels.

    Example:
        >>> cm = load_clim_mean_224(Path('/data/mean'), 'thetao')
    """
    arr = np.squeeze(np.load(mean_dir / f'mean_{var}_1993_2018_all_months.npy')).astype(np.float32)
    t   = torch.tensor(arr[np.newaxis, np.newaxis])
    t   = F.interpolate(t, size=(_MODEL_H, _MODEL_W), mode='bilinear', align_corners=False)
    return t.squeeze().numpy()


# ---------------------------------------------------------------------------
# Ensemble loading
# ---------------------------------------------------------------------------

def load_ensemble_physical(ensemble_dir: Path, clim_means: dict) -> np.ndarray:
    """Load ensemble predictions and denormalize to physical units.

    Args:
        ensemble_dir (Path): Directory with ensemble_predictions.npy,
            shape (N_members, 9, 5, 224, 224) in normalized space.
        clim_means (dict): {'thetao': (224,224), 'so': (224,224)}, NaN on land.

    Returns:
        np.ndarray: (N_members, 9, 5, 224, 224) float32, physical units.

    Example:
        >>> preds = load_ensemble_physical(Path('ensemble_ic_Jan_14012020'), clim)
    """
    preds = np.load(ensemble_dir / 'ensemble_predictions.npy').astype(np.float32)
    for var, ch in [('thetao', 0), ('so', 1)]:
        preds[:, :, ch] += clim_means[var][np.newaxis, np.newaxis]
    return preds


# ---------------------------------------------------------------------------
# ARGO scanning
# ---------------------------------------------------------------------------

def _decode_pid(raw) -> str:
    """Decode a raw ARGO PLATFORM_NUMBER bytes value to a stripped string.

    Args:
        raw: bytes, bytearray, or str from the NetCDF PLATFORM_NUMBER variable.

    Returns:
        str: Stripped platform ID.

    Example:
        >>> _decode_pid(b'2902234 ')
        '2902234'
    """
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode('ascii').strip()
    return str(raw).strip()


def scan_all_floats(
    argo_dir: Path,
    ic_date: datetime,
    surf_pres_max: float = 10.0,
) -> dict[str, list[dict]]:
    """Find all ARGO floats with BoB surface profiles in the 9-day window.

    Args:
        argo_dir (Path): Directory with daily ARGO *_prof.nc files.
        ic_date (datetime): Forecast initialisation date (lead day 0).
        surf_pres_max (float): Max pressure (dbar) defining the surface layer.

    Returns:
        dict: {platform_id: [profile_dict, ...]} where each dict has
            keys lead_day, lat, lon, sst, sss.

    Example:
        >>> floats = scan_all_floats(Path('data/argo/jan2020'), datetime(2020,1,14))
    """
    seen: dict[str, list[dict]] = {}
    for d in range(0, 10):
        nc_path = argo_dir / f'{(ic_date + timedelta(days=d)).strftime("%Y%m%d")}_prof.nc'
        if not nc_path.exists():
            continue
        ds   = xr.open_dataset(nc_path)
        lat  = ds['LATITUDE'].values
        lon  = ds['LONGITUDE'].values
        pres = ds['PRES'].values
        temp = ds['TEMP_ADJUSTED'].values
        psal = ds['PSAL_ADJUSTED'].values
        plat = ds['PLATFORM_NUMBER'].values
        ds.close()
        for i in range(len(lat)):
            pid = _decode_pid(plat[i])
            if not (_BOB_LAT[0] <= lat[i] <= _BOB_LAT[1] and
                    _BOB_LON[0] <= lon[i] <= _BOB_LON[1]):
                continue
            surf = (pres[i] <= surf_pres_max) & np.isfinite(temp[i])
            if not surf.any():
                continue
            sst   = float(temp[i][surf].mean())
            sss_v = psal[i][surf]
            sss   = float(sss_v[np.isfinite(sss_v)].mean()) if np.isfinite(sss_v).any() else np.nan
            seen.setdefault(pid, []).append(
                dict(lead_day=d, lat=float(lat[i]), lon=float(lon[i]),
                     sst=sst, sss=sss))
    return seen


def group_by_leadday(
    all_profiles: dict[str, list[dict]],
) -> dict[int, list[tuple[str, dict]]]:
    """Group (float_id, profile) pairs by lead day 1–9.

    Args:
        all_profiles (dict): Output of scan_all_floats.

    Returns:
        dict: {lead_day: [(float_id, profile_dict), ...]} sorted by float_id.

    Example:
        >>> by_day = group_by_leadday(scan_all_floats(...))
    """
    by_day: dict[int, list] = {d: [] for d in range(1, 10)}
    for fid, profiles in all_profiles.items():
        for p in profiles:
            if 1 <= p['lead_day'] <= 9:
                by_day[p['lead_day']].append((fid, p))
    for d in by_day:
        by_day[d].sort(key=lambda x: x[0])   # alphabetical / numeric order
    return by_day


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def _style_violin(vp: dict, color: str) -> None:
    """Apply consistent styling to a matplotlib violinplot output dict.

    Args:
        vp (dict): Return value of ax.violinplot().
        color (str): Hex colour string for the violin body and median tick.

    Returns:
        None

    Example:
        >>> vp = ax.violinplot([data], vert=False)
        >>> _style_violin(vp, '#2166ac')
    """
    for body in vp['bodies']:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.35)
    vp['cmedians'].set_color(color)
    vp['cmedians'].set_linewidth(2.0)


def build_pdf_figure(
    season_label: str,
    ic_label: str,
    all_profiles: dict[str, list[dict]],
    preds: np.ndarray,
    model_lats: np.ndarray,
    model_lons: np.ndarray,
    color: str,
    output_path: str,
    save_pdf: bool = False,
) -> None:
    """Build and save the ensemble PDF validation figure for one season.

    Layout: 5 rows × 4 columns.  Lead days are paired into rows:
    (1,2), (3,4), (5,6), (7,8), (9,—).  Within each row the four columns are
    [lead_a SST | lead_a SSS | lead_b SST | lead_b SSS].  Row height is
    proportional to the maximum float count across the two lead days in the pair.
    Y-axis labels (float ID + latitude) appear only on SST columns to avoid
    duplication.  ARGO observations are marked as red stars.

    Args:
        season_label (str): e.g. 'Winter' or 'Post-monsoon'.
        ic_label (str): IC date string, e.g. '14 Jan 2020'.
        all_profiles (dict): {float_id: [profile_dict, ...]} from scan_all_floats.
        preds (np.ndarray): (N_members, 9, 5, 224, 224) physical-unit predictions.
        model_lats (np.ndarray): 1-D model latitude array (224,).
        model_lons (np.ndarray): 1-D model longitude array (224,).
        color (str): Violin body colour (season colour).
        output_path (str): File stem without extension.
        save_pdf (bool): Also save a PDF.

    Returns:
        None

    Example:
        >>> build_pdf_figure('Winter', '14 Jan 2020', profiles, preds,
        ...     lats, lons, '#2166ac', 'results/figures/fig7_ensemble_pdf_jan')
    """
    by_day = group_by_leadday(all_profiles)

    # Pre-extract (50, 9, 5) ensemble at each float's first-profile grid point.
    float_ens: dict[str, np.ndarray] = {}
    for fid, profiles in all_profiles.items():
        ref = profiles[0]
        row, col = nearest_grid_point(ref['lat'], ref['lon'], model_lats, model_lons)
        float_ens[fid] = preds[:, :, :, row, col]   # (50, 9, 5)

    # Pair lead days into 5 rows; last pair has only one lead day.
    lead_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, None)]

    n_per_day = {d: len(by_day[d]) for d in range(1, 10)}
    # Row height driven by the larger of the two paired lead days.
    n_per_row = [
        max(n_per_day.get(a, 1), n_per_day.get(b, 0) if b else 0)
        for a, b in lead_pairs
    ]

    row_h_in = 0.72
    total_h  = sum(n_per_row) * row_h_in + 2.8

    fig = plt.figure(figsize=(14, total_h), facecolor='white')
    # 5 columns: SST1, SSS1, [spacer], SST2, SSS2.
    # The narrow spacer (col 2) separates the two lead-day column pairs so that
    # the float-ID y-tick labels on SST2 (col 3) do not overlap SSS1 (col 1).
    gs = gridspec.GridSpec(
        5, 5,
        height_ratios=n_per_row,
        width_ratios=[1.35, 1.0, 0.25, 1.35, 1.0],
        hspace=0.85,
        wspace=0.22,
        top=1 - 1.4 / total_h,
        bottom=1.5 / total_h,
        left=0.13,
        right=0.98,
    )

    is_last_row = len(lead_pairs) - 1

    for row_idx, (lead_a, lead_b) in enumerate(lead_pairs):
        n_row = n_per_row[row_idx]   # max float count for height alignment

        for col_pair, lead_day in enumerate([lead_a, lead_b]):
            floats_d = by_day[lead_day] if lead_day is not None else []
            n_floats = len(floats_d)

            for var_idx, (var, obs_key, col_label) in enumerate(_VAR_SPEC):
                col = col_pair * 3 + var_idx
                ax  = fig.add_subplot(gs[row_idx, col])

                if lead_day is None or not floats_d:
                    ax.set_visible(False)
                    continue

                # ---- Per-float ensemble and observation ----
                ens_data:    list[np.ndarray] = []
                obs_vals:    list[float]      = []
                tick_labels: list[str]        = []

                for fid, profile in floats_d:
                    raw_ens = float_ens[fid][:, lead_day - 1, _VAR_IDX[var]]
                    valid   = raw_ens[np.isfinite(raw_ens)]
                    if len(np.unique(valid)) < 2:
                        valid = np.concatenate([valid, valid + 1e-6])
                    ens_data.append(valid)
                    obs_vals.append(profile[obs_key])
                    tick_labels.append(
                        f'{fid[-4:]}  {profile["lat"]:.1f}°N')

                # ---- Violin ----
                vp = ax.violinplot(
                    ens_data,
                    positions=list(range(n_floats)),
                    vert=False,
                    showmedians=True,
                    showextrema=False,
                    widths=0.65,
                )
                _style_violin(vp, color)

                # ---- ARGO stars ----
                for k, (_, profile) in enumerate(floats_d):
                    obs = obs_vals[k]
                    if np.isfinite(obs):
                        ax.scatter([obs], [k], marker='*', color='crimson',
                                   s=55, zorder=7, linewidths=0.4,
                                   edgecolors='darkred')

                # ---- Y-axis: labels only on SST columns ----
                ax.set_yticks(range(n_floats))
                if var_idx == 0:
                    ax.set_yticklabels(tick_labels, fontsize=10)
                else:
                    ax.set_yticklabels([])
                ax.set_ylim(-0.5, n_row - 0.5)
                ax.tick_params(axis='y', length=0)

                # ---- X-axis ----
                ax.tick_params(axis='x', labelsize=11)
                if row_idx == is_last_row:
                    ax.set_xlabel(col_label, fontsize=13)

                # ---- Grid and spines ----
                ax.grid(axis='x', color='#e0e0e0', linewidth=0.5)
                ax.set_axisbelow(True)
                for sp in ax.spines.values():
                    sp.set_color('#cccccc')
                    sp.set_linewidth(0.6)

                # ---- Panel title: variable name + lead day ----
                var_short = col_label.split('(')[0].strip()   # 'SST' or 'SSS'
                unit      = col_label.split('(')[1].rstrip(')')  # '°C' or 'PSU'
                ax.set_title(f'{var_short}  +{lead_day}d  ({unit})',
                             fontsize=14, fontweight='bold', pad=3)

    # ---- Legend ----
    handles = [
        Patch(facecolor=color, alpha=0.35, label='Ensemble PDF'),
        Line2D([0], [0], color=color, lw=2.0, label='Ensemble median'),
        Line2D([0], [0], marker='*', color='crimson', markersize=9,
               linestyle='None', label='ARGO observation'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=15,
               frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        f'AFNO RT  Ensemble PDF vs ARGO — {season_label} 2020  (IC: {ic_label})',
        fontsize=19, fontweight='bold',
    )

    _save(fig, output_path, save_pdf)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, output_path: str, save_pdf: bool) -> None:
    """Save figure to PNG and optionally PDF, then close.

    Args:
        fig (plt.Figure): Figure to save.
        output_path (str): File stem without extension.
        save_pdf (bool): Also write a PDF.

    Returns:
        None

    Example:
        >>> _save(fig, 'results/figures/fig7_jan', True)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    png = f'{output_path}.png'
    fig.savefig(png, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved: {png}')
    if save_pdf:
        pdf = f'{output_path}.pdf'
        fig.savefig(pdf, bbox_inches='tight', facecolor='white')
        print(f'Saved: {pdf}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the PDF validation figure script.

    Returns:
        argparse.Namespace: Parsed arguments.

    Example:
        >>> args = parse_args()
    """
    p = argparse.ArgumentParser(
        description='Ensemble PDF vs ARGO float observations (fig7)')
    p.add_argument('--jan_dir',    required=True)
    p.add_argument('--oct_dir',    required=True)
    p.add_argument('--argo_jan',   required=True)
    p.add_argument('--argo_oct',   required=True)
    p.add_argument('--apr_dir',    default=None)
    p.add_argument('--jun_dir',    default=None)
    p.add_argument('--argo_apr',   default=None)
    p.add_argument('--argo_jun',   default=None)
    p.add_argument('--mean_dir',   required=True)
    p.add_argument('--ocean_file', required=True)
    p.add_argument('--output',     default='results/figures/fig7_ensemble_pdf')
    p.add_argument('--surf_pres',  type=float, default=10.0)
    p.add_argument('--pdf',        action='store_true')
    return p.parse_args()


def main() -> None:
    """Generate fig7 PDF validation figures for Winter and Post-monsoon seasons.

    Example:
        >>> # python src/visualization/plot_ensemble_pdf_validation.py \\
        >>> #     --jan_dir results/AFNO_BoB_Surf_E14/ensemble_ic_Jan_14012020 \\
        >>> #     --oct_dir results/AFNO_BoB_Surf_E14/ensemble_ic_Oct_14102020 \\
        >>> #     --argo_jan data/argo/jan2020 --argo_oct data/argo/oct2020 \\
        >>> #     --mean_dir data/1993_2020/mean \\
        >>> #     --ocean_file data/1993_2020/ocean.nc \\
        >>> #     --output results/figures/fig7_ensemble_pdf --pdf
    """
    args     = parse_args()
    mean_dir = Path(args.mean_dir)

    print('Loading climatological means ...')
    clim_means = {
        'thetao': load_clim_mean_224(mean_dir, 'thetao'),
        'so':     load_clim_mean_224(mean_dir, 'so'),
    }

    print('Loading model grid ...')
    model_lats, model_lons = load_model_grid(Path(args.ocean_file))

    ic_jan = datetime(2020,  1, 14)
    ic_oct = datetime(2020, 10, 14)

    # ---- Winter ----
    print(f'\nScanning ARGO floats — Winter (IC {ic_jan.date()}) ...')
    profiles_jan = scan_all_floats(Path(args.argo_jan), ic_jan, args.surf_pres)
    by_day_jan   = group_by_leadday(profiles_jan)
    print('  Obs per lead day:', {d: len(v) for d, v in by_day_jan.items()})
    print('Loading Winter ensemble ...')
    preds_jan = load_ensemble_physical(Path(args.jan_dir), clim_means)
    build_pdf_figure(
        'Winter', '14 Jan 2020',
        profiles_jan, preds_jan, model_lats, model_lons,
        _COL_JAN, f'{args.output}_jan', save_pdf=args.pdf,
    )

    # ---- Post-monsoon ----
    print(f'\nScanning ARGO floats — PostMonsoon (IC {ic_oct.date()}) ...')
    profiles_oct = scan_all_floats(Path(args.argo_oct), ic_oct, args.surf_pres)
    by_day_oct   = group_by_leadday(profiles_oct)
    print('  Obs per lead day:', {d: len(v) for d, v in by_day_oct.items()})
    print('Loading PostMonsoon ensemble ...')
    preds_oct = load_ensemble_physical(Path(args.oct_dir), clim_means)
    build_pdf_figure(
        'Post-monsoon', '14 Oct 2020',
        profiles_oct, preds_oct, model_lats, model_lons,
        _COL_OCT, f'{args.output}_oct', save_pdf=args.pdf,
    )

    # ---- Pre-monsoon (optional) ----
    if args.apr_dir and args.argo_apr:
        ic_apr = datetime(2020, 4, 14)
        print(f'\nScanning ARGO floats — PreMonsoon (IC {ic_apr.date()}) ...')
        profiles_apr = scan_all_floats(Path(args.argo_apr), ic_apr, args.surf_pres)
        by_day_apr   = group_by_leadday(profiles_apr)
        print('  Obs per lead day:', {d: len(v) for d, v in by_day_apr.items()})
        print('Loading PreMonsoon ensemble ...')
        preds_apr = load_ensemble_physical(Path(args.apr_dir), clim_means)
        build_pdf_figure(
            'Pre-monsoon', '14 Apr 2020',
            profiles_apr, preds_apr, model_lats, model_lons,
            _COL_APR, f'{args.output}_apr', save_pdf=args.pdf,
        )

    # ---- Monsoon (optional) ----
    if args.jun_dir and args.argo_jun:
        ic_jun = datetime(2020, 6, 14)
        print(f'\nScanning ARGO floats — Monsoon (IC {ic_jun.date()}) ...')
        profiles_jun = scan_all_floats(Path(args.argo_jun), ic_jun, args.surf_pres)
        by_day_jun   = group_by_leadday(profiles_jun)
        print('  Obs per lead day:', {d: len(v) for d, v in by_day_jun.items()})
        print('Loading Monsoon ensemble ...')
        preds_jun = load_ensemble_physical(Path(args.jun_dir), clim_means)
        build_pdf_figure(
            'Monsoon', '14 Jun 2020',
            profiles_jun, preds_jun, model_lats, model_lons,
            _COL_JUN, f'{args.output}_jun', save_pdf=args.pdf,
        )


if __name__ == '__main__':
    main()

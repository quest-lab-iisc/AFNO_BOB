"""Ensemble mean ± std time series at ARGO float locations versus ARGO truth.

Produces three figures:
  (1) fig6_ensemble_timeseries — 2×2 summary (selected float per season × SST/SSS)
  (2) fig6_all_floats_jan     — all Jan-2020 floats, one row per float (SST, SSS)
  (3) fig6_all_floats_oct     — all Oct-2020 floats, one row per float (SST, SSS)

For each float the ensemble (50 members) is extracted at that float's first-profile
location for all lead days 1–9.  ARGO observations are overlaid as dots: a hollow
circle at lead 0 (IC anchor) and filled circles at lead ≥ 1 (verification).

Inputs:
    --jan_dir (str)    : Directory with Winter ensemble_predictions.npy.
    --oct_dir (str)    : Directory with PostMonsoon ensemble_predictions.npy.
    --argo_jan (str)   : Directory with per-day ARGO _prof.nc files, Jan 2020.
    --argo_oct (str)   : Directory with per-day ARGO _prof.nc files, Oct 2020.
    --mean_dir (str)   : Directory with mean_{var}_1993_2018_all_months.npy files.
    --ocean_file (str) : GLORYS ocean.nc (supplies the lat/lon grid).
    --float_jan (str)  : Platform number for the 2×2 Winter float (default 2902234).
    --float_oct (str)  : Platform number for the 2×2 PostMonsoon float (default 2902230).
    --output (str)     : Output path stem; '_jan' / '_oct' suffixes are appended for
                         the all-floats figures.
    --surf_pres (float): Max pressure (dbar) for surface layer (default 10).
    --pdf              : Also save PDF versions.

Outputs:
    {output}.png / .pdf          — 2×2 summary figure.
    {output}_all_floats_jan.png/.pdf  — all Jan floats.
    {output}_all_floats_oct.png/.pdf  — all Oct floats.

Example:
    conda activate BoB_Surf_2
    python src/visualization/plot_ensemble_timeseries.py \\
        --jan_dir results/AFNO_BoB_Surf_E14/ensemble_ic_Jan_14012020 \\
        --oct_dir results/AFNO_BoB_Surf_E14/ensemble_ic_Oct_14102020 \\
        --argo_jan data/argo/jan2020 --argo_oct data/argo/oct2020 \\
        --mean_dir data/1993_2020/mean \\
        --ocean_file data/1993_2020/ocean.nc \\
        --output results/figures/fig6_ensemble_timeseries --pdf
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

_BOB_LAT   = (4.0, 23.0)
_BOB_LON   = (77.0, 99.0)
_MODEL_H, _MODEL_W = 224, 224
_OCEAN_VARS = ['thetao', 'so', 'uo', 'vo', 'zos']
_VAR_IDX    = {v: i for i, v in enumerate(_OCEAN_VARS)}

_COL_JAN = '#2166ac'
_COL_OCT = '#d6604d'


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def load_model_grid(ocean_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Build 224×224 model lat/lon arrays from GLORYS ocean.nc.

    Args:
        ocean_file (Path): GLORYS NetCDF with latitude/longitude coordinates.

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
    lats = np.interp(row_pos, np.arange(len(lats_g)), lats_g).astype(np.float32)
    lons = np.interp(col_pos, np.arange(len(lons_g)), lons_g).astype(np.float32)
    return lats, lons


def nearest_grid_point(lat: float, lon: float,
                       model_lats: np.ndarray,
                       model_lons: np.ndarray) -> tuple[int, int]:
    """Return (row, col) of the nearest model grid cell.

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
    """Load and bilinearly resize climatological mean to 224×224 (NaN on land).

    Args:
        mean_dir (Path): Directory containing mean_{var}_1993_2018_all_months.npy.
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

    Adds the climatological mean (NaN on land) to thetao and so channels so that
    land pixels become NaN in all members, preventing artificially low spread.

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
# ARGO float discovery and profile loading
# ---------------------------------------------------------------------------

def _decode_platform_id(raw) -> str:
    """Decode a raw ARGO PLATFORM_NUMBER value to a stripped string.

    Args:
        raw: Raw platform value (bytes, bytearray, or str).

    Returns:
        str: Stripped platform ID string.

    Example:
        >>> _decode_platform_id(b'2902234 ')
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
    """Find all ARGO floats with BoB surface profiles in the 9-day forecast window.

    Args:
        argo_dir (Path): Directory with daily ARGO *_prof.nc files.
        ic_date (datetime): Forecast initialisation date (lead day 0).
        surf_pres_max (float): Max pressure (dbar) for the surface layer.

    Returns:
        dict: {platform_id: [profile_dict, ...]} where each profile_dict has
            keys lead_day, lat, lon, sst, sss.  Profiles are sorted by lead_day.

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
            pid = _decode_platform_id(plat[i])
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
                dict(lead_day=d, lat=float(lat[i]), lon=float(lon[i]), sst=sst, sss=sss)
            )
    return seen


def collect_float_profiles(
    argo_dir: Path,
    ic_date: datetime,
    float_id: str,
    surf_pres_max: float = 10.0,
) -> list[dict]:
    """Collect all surface profiles from one ARGO float within the 9-day window.

    Args:
        argo_dir (Path): Directory with daily ARGO *_prof.nc files.
        ic_date (datetime): Forecast initialisation date.
        float_id (str): ARGO platform number string (e.g. '2902234').
        surf_pres_max (float): Max pressure (dbar) for surface layer.

    Returns:
        list[dict]: Profiles with keys lead_day, lat, lon, sst, sss.
            Includes lead 0 (IC date) if the float profiled then.

    Example:
        >>> profiles = collect_float_profiles(Path('data/argo/jan2020'),
        ...                                   datetime(2020,1,14), '2902234')
    """
    all_floats = scan_all_floats(argo_dir, ic_date, surf_pres_max)
    return all_floats.get(float_id, [])


# ---------------------------------------------------------------------------
# Ensemble extraction at a float location
# ---------------------------------------------------------------------------

def extract_float_timeseries(
    preds: np.ndarray,
    profiles: list[dict],
    model_lats: np.ndarray,
    model_lons: np.ndarray,
) -> dict:
    """Extract 9-day ensemble forecast at a float's first-profile location.

    The extraction point is fixed at the float's earliest profile for all 9 lead
    days, giving a continuous band.  Returns separate IC (lead 0) and verification
    (lead ≥ 1) observation lists.

    Args:
        preds (np.ndarray): (N_members, 9, 5, 224, 224) physical-unit predictions.
        profiles (list[dict]): Float profiles from collect_float_profiles / scan_all_floats.
        model_lats (np.ndarray): 1-D model latitude array (224,).
        model_lons (np.ndarray): 1-D model longitude array (224,).

    Returns:
        dict: {
            'lead_days': list [1..9],
            'ens_mean_sst': (9,), 'ens_std_sst': (9,),
            'ens_mean_sss': (9,), 'ens_std_sss': (9,),
            'ic_sst': float or None,   # ARGO obs at lead 0
            'ic_sss': float or None,
            'argo_lead': list[int],    # lead days ≥ 1 with ARGO obs
            'argo_sst':  list[float],
            'argo_sss':  list[float],
            'ref_lat': float, 'ref_lon': float,  # extraction point
        }

    Example:
        >>> ts = extract_float_timeseries(preds, profiles, lats, lons)
    """
    ref = profiles[0]
    row, col = nearest_grid_point(ref['lat'], ref['lon'], model_lats, model_lons)

    lead_days = list(range(1, 10))
    ens_mean_sst, ens_std_sst = [], []
    ens_mean_sss, ens_std_sss = [], []

    for d in lead_days:
        sst_ens = preds[:, d - 1, _VAR_IDX['thetao'], row, col]
        sss_ens = preds[:, d - 1, _VAR_IDX['so'],     row, col]
        ens_mean_sst.append(float(np.nanmean(sst_ens)))
        ens_std_sst.append(float(np.nanstd(sst_ens)))
        ens_mean_sss.append(float(np.nanmean(sss_ens)))
        ens_std_sss.append(float(np.nanstd(sss_ens)))

    ic_sst = ic_sss = None
    argo_lead, argo_sst, argo_sss = [], [], []
    for p in profiles:
        if p['lead_day'] == 0:
            ic_sst = p['sst']
            ic_sss = p['sss'] if np.isfinite(p['sss']) else None
        elif p['lead_day'] >= 1:
            argo_lead.append(p['lead_day'])
            argo_sst.append(p['sst'])
            argo_sss.append(p['sss'])

    return dict(
        lead_days=lead_days,
        ens_mean_sst=np.array(ens_mean_sst),
        ens_std_sst=np.array(ens_std_sst),
        ens_mean_sss=np.array(ens_mean_sss),
        ens_std_sss=np.array(ens_std_sss),
        ic_sst=ic_sst, ic_sss=ic_sss,
        argo_lead=argo_lead, argo_sst=argo_sst, argo_sss=argo_sss,
        ref_lat=float(model_lats[row]), ref_lon=float(model_lons[col]),
    )


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------

def _plot_panel(
    ax: plt.Axes,
    ts: dict,
    var: str,
    color: str,
    title: str,
    ylabel: str,
    show_xlabel: bool,
    float_id: str,
    ref_profile: dict,
    compact: bool = False,
) -> None:
    """Draw one ensemble band + ARGO dots panel.

    Hollow circle at lead 0 = IC anchor (ARGO observation on the IC date).
    Filled circle at lead ≥ 1 = ARGO verification observation.

    Args:
        ax (plt.Axes): Target axes.
        ts (dict): Output of extract_float_timeseries.
        var (str): 'sst' or 'sss'.
        color (str): Band and line colour (hex string).
        title (str): Panel title (shown only when not compact).
        ylabel (str): Y-axis label.
        show_xlabel (bool): Draw x-axis 'Lead day' label.
        float_id (str): Float platform ID for the annotation text.
        ref_profile (dict): First profile dict (for location annotation).
        compact (bool): Use smaller fonts and suppress title (for all-float panels).

    Returns:
        None

    Example:
        >>> _plot_panel(ax, ts, 'sst', '#2166ac', 'Winter SST', 'SST (°C)',
        ...             False, '2902234', profiles[0])
    """
    days = np.array(ts['lead_days'])    # [1..9]
    mu   = ts[f'ens_mean_{var}']
    sig  = ts[f'ens_std_{var}']
    ic_val   = ts[f'ic_{var}']          # float or None
    obs_days = ts['argo_lead']
    obs_vals = ts[f'argo_{var}']

    fsize = 7.5 if compact else 10

    ax.fill_between(days, mu - sig, mu + sig,
                    color=color, alpha=0.25, label='Ensemble mean ± 1σ')
    ax.plot(days, mu, color=color, lw=1.8 if compact else 2.0,
            label='Ensemble mean')

    # IC anchor dot (hollow circle)
    if ic_val is not None and np.isfinite(ic_val):
        ax.scatter([0], [ic_val],
                   facecolors='none', edgecolors='k',
                   s=35 if compact else 50, linewidths=1.2, zorder=5,
                   marker='o', label='ARGO (IC anchor)')

    # Verification dots (filled circle)
    valid_obs = [(d, v) for d, v in zip(obs_days, obs_vals) if np.isfinite(v)]
    if valid_obs:
        ox, oy = zip(*valid_obs)
        ax.scatter(ox, oy, color='k',
                   s=35 if compact else 50, zorder=5,
                   marker='o', label='ARGO observation')

    if not compact:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=5)
    ax.set_ylabel(ylabel, fontsize=fsize)
    if show_xlabel:
        ax.set_xlabel('Lead day', fontsize=fsize)
    ax.set_xticks(range(0, 10))
    ax.set_xlim(-0.5, 9.5)
    ax.grid(True, color='#e0e0e0', linewidth=0.5 if compact else 0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=fsize - 0.5)
    for sp in ax.spines.values():
        sp.set_color('#bbbbbb')
        sp.set_linewidth(0.7)

    if not compact:
        loc_str = (f'Float {float_id}  ({ref_profile["lat"]:.2f}°N,'
                   f' {ref_profile["lon"]:.2f}°E)')
        ax.annotate(loc_str, xy=(0.02, 0.04), xycoords='axes fraction',
                    fontsize=7.5, color='#444444')


# ---------------------------------------------------------------------------
# Figure 1: 2×2 summary
# ---------------------------------------------------------------------------

def build_summary_figure(
    ts_jan: dict, ts_oct: dict,
    prof_jan: list, prof_oct: list,
    float_jan: str, float_oct: str,
    output_path: str,
    ic_jan: str, ic_oct: str,
    save_pdf: bool = False,
) -> None:
    """Build and save the 2×2 summary ensemble time series figure.

    Args:
        ts_jan (dict): Timeseries dict for Winter.
        ts_oct (dict): Timeseries dict for PostMonsoon.
        prof_jan (list): Float profiles for Winter (for annotation).
        prof_oct (list): Float profiles for PostMonsoon (for annotation).
        float_jan (str): Float ID label for Winter.
        float_oct (str): Float ID label for PostMonsoon.
        output_path (str): File stem without extension.
        ic_jan (str): IC date label for Winter.
        ic_oct (str): IC date label for PostMonsoon.
        save_pdf (bool): Also save a PDF.

    Returns:
        None

    Example:
        >>> build_summary_figure(ts_jan, ts_oct, p_jan, p_oct,
        ...     '2902234', '2902230', 'results/figures/fig6',
        ...     '14 Jan 2020', '14 Oct 2020')
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex='col')
    fig.patch.set_facecolor('white')

    specs = [
        (0, 0, ts_jan, 'sst', _COL_JAN,
         f'Winter SST  (IC: {ic_jan})', 'SST (°C)', False, float_jan, prof_jan),
        (0, 1, ts_oct, 'sst', _COL_OCT,
         f'Post-monsoon SST  (IC: {ic_oct})', 'SST (°C)', False, float_oct, prof_oct),
        (1, 0, ts_jan, 'sss', _COL_JAN,
         f'Winter SSS  (IC: {ic_jan})', 'SSS (PSU)', True, float_jan, prof_jan),
        (1, 1, ts_oct, 'sss', _COL_OCT,
         f'Post-monsoon SSS  (IC: {ic_oct})', 'SSS (PSU)', True, float_oct, prof_oct),
    ]

    for row, col, ts, var, color, title, ylabel, show_xl, fid, profs in specs:
        _plot_panel(axes[row, col], ts, var, color, title, ylabel, show_xl,
                    fid, profs[0], compact=False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center', ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        'AFNO RT Ensemble Forecast vs ARGO Observations — Bay of Bengal 2020',
        fontsize=12, fontweight='bold', y=1.01)
    fig.tight_layout(h_pad=1.5, w_pad=1.5)
    _save(fig, output_path, save_pdf)


# ---------------------------------------------------------------------------
# Figure 2/3: all floats for one season
# ---------------------------------------------------------------------------

def build_all_floats_figure(
    season_label: str,
    ic_label: str,
    all_profiles: dict,
    preds: np.ndarray,
    model_lats: np.ndarray,
    model_lons: np.ndarray,
    color: str,
    output_path: str,
    save_pdf: bool = False,
) -> None:
    """Build a tall figure with one row per ARGO float for a single season.

    Each row shows two panels: SST (left) and SSS (right).  Floats are sorted
    numerically by platform ID.  Row labels show float ID and first-profile location.

    Args:
        season_label (str): Season name for the figure title, e.g. 'Winter'.
        ic_label (str): IC date string, e.g. '14 Jan 2020'.
        all_profiles (dict): {float_id: [profile_dict,...]} from scan_all_floats.
        preds (np.ndarray): (N_members, 9, 5, 224, 224) physical-unit predictions.
        model_lats (np.ndarray): 1-D model latitude array (224,).
        model_lons (np.ndarray): 1-D model longitude array (224,).
        color (str): Band colour (hex string).
        output_path (str): File stem without extension.
        save_pdf (bool): Also save a PDF.

    Returns:
        None

    Example:
        >>> build_all_floats_figure('Winter', '14 Jan 2020', profiles, preds,
        ...     lats, lons, '#2166ac', 'results/figures/fig6_all_floats_jan')
    """
    float_ids = sorted(all_profiles.keys(), key=lambda x: int(x))
    n = len(float_ids)
    row_h = 1.6
    fig, axes = plt.subplots(n, 2, figsize=(12, row_h * n),
                             sharex=True, squeeze=False)
    fig.patch.set_facecolor('white')

    for i, fid in enumerate(float_ids):
        profiles = all_profiles[fid]
        ts = extract_float_timeseries(preds, profiles, model_lats, model_lons)
        ref = profiles[0]

        for j, (var, ylabel) in enumerate([('sst', 'SST (°C)'), ('sss', 'SSS (PSU)')]):
            ax = axes[i, j]
            show_xl = (i == n - 1)
            _plot_panel(ax, ts, var, color, title='', ylabel='',
                        show_xlabel=show_xl, float_id=fid, ref_profile=ref,
                        compact=True)
            # minimal y label only on SST column
            if j == 0:
                ax.set_ylabel(f'{fid}\n({ref["lat"]:.1f}°N,{ref["lon"]:.1f}°E)',
                              fontsize=6.5, rotation=0, ha='right', va='center',
                              labelpad=60)

        # Column headers on first row
        if i == 0:
            axes[0, 0].set_title(f'{season_label} SST  (IC: {ic_label})',
                                 fontsize=9, fontweight='bold', pad=4)
            axes[0, 1].set_title(f'{season_label} SSS  (IC: {ic_label})',
                                 fontsize=9, fontweight='bold', pad=4)

    # Legend from first row
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center', ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        f'AFNO RT Ensemble vs ARGO — {season_label} 2020  ({n} floats)',
        fontsize=11, fontweight='bold')
    fig.tight_layout(h_pad=0.3, w_pad=0.5)
    _save(fig, output_path, save_pdf)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, output_path: str, save_pdf: bool) -> None:
    """Save figure to PNG (and optionally PDF), then close.

    Args:
        fig (plt.Figure): Matplotlib figure to save.
        output_path (str): File stem without extension.
        save_pdf (bool): Also write a PDF.

    Returns:
        None

    Example:
        >>> _save(fig, 'results/figures/fig6', True)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    png = f'{output_path}.png'
    fig.savefig(png, dpi=300, bbox_inches='tight', facecolor='white')
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
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.

    Example:
        >>> args = parse_args()
    """
    p = argparse.ArgumentParser(
        description='Ensemble timeseries vs ARGO floats (fig6 + all-float panels)')
    p.add_argument('--jan_dir',    required=True)
    p.add_argument('--oct_dir',    required=True)
    p.add_argument('--argo_jan',   required=True)
    p.add_argument('--argo_oct',   required=True)
    p.add_argument('--mean_dir',   required=True)
    p.add_argument('--ocean_file', required=True)
    p.add_argument('--float_jan',  default='2902234',
                   help='Platform number for Winter 2×2 float (default 2902234)')
    p.add_argument('--float_oct',  default='2902230',
                   help='Platform number for PostMonsoon 2×2 float (default 2902230)')
    p.add_argument('--output',     default='results/figures/fig6_ensemble_timeseries')
    p.add_argument('--surf_pres',  type=float, default=10.0)
    p.add_argument('--pdf',        action='store_true')
    return p.parse_args()


def main() -> None:
    """Entry point: generate the 2×2 summary and both all-floats figures.

    Example:
        >>> # python src/visualization/plot_ensemble_timeseries.py \\
        >>> #     --jan_dir results/AFNO_BoB_Surf_E14/ensemble_ic_Jan_14012020 \\
        >>> #     --oct_dir results/AFNO_BoB_Surf_E14/ensemble_ic_Oct_14102020 \\
        >>> #     --argo_jan data/argo/jan2020 --argo_oct data/argo/oct2020 \\
        >>> #     --mean_dir data/1993_2020/mean \\
        >>> #     --ocean_file data/1993_2020/ocean.nc \\
        >>> #     --output results/figures/fig6_ensemble_timeseries --pdf
    """
    args = parse_args()
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

    # ---- Jan data --------------------------------------------------------
    print(f'\nScanning ARGO floats — Winter (IC {ic_jan.date()}) ...')
    all_profiles_jan = scan_all_floats(Path(args.argo_jan), ic_jan, args.surf_pres)
    print(f'  Found {len(all_profiles_jan)} floats')
    print('Loading Winter ensemble predictions ...')
    preds_jan = load_ensemble_physical(Path(args.jan_dir), clim_means)

    # ---- Oct data --------------------------------------------------------
    print(f'\nScanning ARGO floats — PostMonsoon (IC {ic_oct.date()}) ...')
    all_profiles_oct = scan_all_floats(Path(args.argo_oct), ic_oct, args.surf_pres)
    print(f'  Found {len(all_profiles_oct)} floats')
    print('Loading PostMonsoon ensemble predictions ...')
    preds_oct = load_ensemble_physical(Path(args.oct_dir), clim_means)

    # ---- 2×2 summary figure ---------------------------------------------
    print(f'\n--- 2×2 summary: float {args.float_jan} (Jan) / {args.float_oct} (Oct) ---')
    if args.float_jan not in all_profiles_jan:
        raise ValueError(f'Float {args.float_jan} not found in Jan ARGO data')
    if args.float_oct not in all_profiles_oct:
        raise ValueError(f'Float {args.float_oct} not found in Oct ARGO data')

    prof_jan = all_profiles_jan[args.float_jan]
    prof_oct = all_profiles_oct[args.float_oct]
    for p in prof_jan:
        print(f'  Jan {args.float_jan} lead+{p["lead_day"]}d '
              f'lat={p["lat"]:.2f} lon={p["lon"]:.2f} SST={p["sst"]:.2f} SSS={p["sss"]:.2f}')
    for p in prof_oct:
        print(f'  Oct {args.float_oct} lead+{p["lead_day"]}d '
              f'lat={p["lat"]:.2f} lon={p["lon"]:.2f} SST={p["sst"]:.2f} SSS={p["sss"]:.2f}')

    ts_jan = extract_float_timeseries(preds_jan, prof_jan, model_lats, model_lons)
    ts_oct = extract_float_timeseries(preds_oct, prof_oct, model_lats, model_lons)

    build_summary_figure(
        ts_jan, ts_oct, prof_jan, prof_oct,
        args.float_jan, args.float_oct,
        args.output, '14 Jan 2020', '14 Oct 2020',
        save_pdf=args.pdf,
    )

    # ---- All-floats figures ----------------------------------------------
    stem = args.output.replace('_timeseries', '')
    jan_all_path = f'{stem}_all_floats_jan'
    oct_all_path = f'{stem}_all_floats_oct'

    print(f'\n--- All Jan floats ({len(all_profiles_jan)}) ---')
    build_all_floats_figure(
        'Winter', '14 Jan 2020',
        all_profiles_jan, preds_jan,
        model_lats, model_lons, _COL_JAN,
        jan_all_path, save_pdf=args.pdf,
    )

    print(f'\n--- All Oct floats ({len(all_profiles_oct)}) ---')
    build_all_floats_figure(
        'Post-monsoon', '14 Oct 2020',
        all_profiles_oct, preds_oct,
        model_lats, model_lons, _COL_OCT,
        oct_all_path, save_pdf=args.pdf,
    )


if __name__ == '__main__':
    main()

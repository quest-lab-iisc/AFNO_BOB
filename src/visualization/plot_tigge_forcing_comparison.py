"""Manuscript figure: TIGGE ensemble mean and spread of atmospheric forcing.

Produces a 12-row × 4-column panel figure showing the ensemble mean and spread
(std) of all six TIGGE atmospheric forcing channels for two Bay of Bengal seasons
(Winter and Post-monsoon) at two lead times (+3 d and +5 d).

Row order: ssr mean, ssr std, tp mean, tp std, u10 mean, u10 std,
           v10 mean, v10 std, msl mean, msl std, tcc mean, tcc std

Columns: Winter +Xd, Winter +Yd, Post-monsoon +Xd, Post-monsoon +Yd

Normalization notes:
    ssr, tp, msl  — z-score (x − mean) / std stored in mean_dir.
    u10, v10, tcc — raw physical values (no normalization applied).

Inputs:
    --tigge_dir (str): Directory with Tigge_{month}_2020_ens_m{nn:02d}.npy files.
    --jan_month (str): Month label used in Jan filenames (default: Jan).
    --oct_month (str): Month label used in Oct filenames (default: Oct).
    --mean_dir (str): Directory with normalization mean/var .npy files.
    --ocean_file (str): GLORYS ocean.nc for grid coordinates.
    --output (str): Output base path without extension.
    --leads (int ...): Two lead days to plot (default: 3 5).
    --n_members (int): Number of TIGGE members per season (default: 50).
    --levels (int): Number of contourf levels (default: 10).
    --pdf (flag): Also save vector PDF alongside the PNG.

Outputs:
    {output}.png               — 300 dpi raster (pcolormesh)
    {output}_contourf.png      — 300 dpi raster (contourf)
    {output}.pdf               — vector PDF (pcolormesh), only with --pdf
    {output}_contourf.pdf      — vector PDF (contourf), only with --pdf

Example:
    conda activate BoB_Surf_2
    python src/visualization/plot_tigge_forcing_comparison.py \\
        --tigge_dir data/tigge_ensemble \\
        --mean_dir data/1993_2020/mean \\
        --ocean_file data/1993_2020/ocean.nc \\
        --output results/figures/tigge_forcing_comparison \\
        --leads 3 5 --levels 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator
from scipy.interpolate import griddata

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

sys.path.append(str(Path(__file__).parent.parent))

from inference.run_ensemble_verification import interp_to_glorys

try:
    import cmocean
    _SOLAR_CMAP = cmocean.cm.solar
    _RAIN_CMAP  = cmocean.cm.rain
    _DIV_CMAP   = cmocean.cm.balance
    _STD_CMAP   = cmocean.cm.amp
except ImportError:
    _SOLAR_CMAP = 'YlOrRd'
    _RAIN_CMAP  = 'Blues'
    _DIV_CMAP   = 'RdBu_r'
    _STD_CMAP   = 'YlOrRd'

_GREY_CMAP  = 'Greys_r'
_SEQ_CMAP   = 'RdYlBu_r'

_LON_RANGE = (77, 99)
_LAT_RANGE = (4, 23)

# Channel order within TIGGE files: [ssr, tp, u10, v10, msl, tcc]
_ATM_VARS = ['ssr', 'tp', 'u10', 'v10', 'msl', 'tcc']
_ATM_CH   = {v: i for i, v in enumerate(_ATM_VARS)}

# Variables that carry z-score normalization; others are raw physical.
_ZSCORE_VARS = {'ssr', 'tp', 'msl'}


# ---------------------------------------------------------------------------
# Normalization stat helpers
# ---------------------------------------------------------------------------

def load_atm_norm_stats(mean_dir: Path) -> dict:
    """Load atmospheric normalization means and stds from disk.

    Only ssr, tp, msl have stored stats. u10, v10, tcc are unmodified physical
    values and return None for both mean and std.

    Args:
        mean_dir (Path): Directory containing {var}_mean_*.npy and {var}_var_*.npy.

    Returns:
        dict: Keys are variable names; values are dicts with 'mean' and 'std'
            (both np.ndarray of shape (1, 77, 89) or None).

    Example:
        >>> stats = load_atm_norm_stats(Path('data/1993_2020/mean'))
        >>> stats['ssr']['mean'].shape
        (1, 77, 89)
    """
    stats = {}
    for var in _ATM_VARS:
        if var in _ZSCORE_VARS:
            mean_f = mean_dir / f'{var}_mean_1993_2018_all_months.npy'
            var_f  = mean_dir / f'{var}_var_1993_2018_all_months.npy'
            stats[var] = {
                'mean': np.load(mean_f).astype(np.float32),
                'std':  np.sqrt(np.load(var_f)).astype(np.float32),
            }
        else:
            stats[var] = {'mean': None, 'std': None}
    return stats


def interp_stat_to_224(stat_77x89: np.ndarray) -> np.ndarray:
    """Bilinearly interpolate a (1, 77, 89) norm stat array to (224, 224).

    Args:
        stat_77x89 (np.ndarray): shape (1, 77, 89) float32 norm stat.

    Returns:
        np.ndarray: shape (224, 224) float32 interpolated stat.

    Example:
        >>> mean_224 = interp_stat_to_224(stats['ssr']['mean'])
    """
    t = torch.tensor(stat_77x89[np.newaxis], dtype=torch.float32)   # (1, 1, 77, 89)
    t = F.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)
    return t.squeeze().numpy()                                        # (224, 224)


def denormalize_atm(arr_224: np.ndarray, var: str, stats: dict) -> np.ndarray:
    """Denormalize a 224×224 atmospheric field to physical units.

    Args:
        arr_224 (np.ndarray): Normalized field, shape (224, 224).
        var (str): Variable name, e.g. 'ssr'.
        stats (dict): Output of load_atm_norm_stats().

    Returns:
        np.ndarray: Physical-unit field, shape (224, 224).

    Example:
        >>> ssr_phys = denormalize_atm(ssr_norm, 'ssr', stats)
    """
    if var not in _ZSCORE_VARS:
        return arr_224.copy()
    mean_224 = interp_stat_to_224(stats[var]['mean'])
    std_224  = interp_stat_to_224(stats[var]['std'])
    return arr_224 * std_224 + mean_224


def to_display_units(arr: np.ndarray, var: str) -> np.ndarray:
    """Convert a physical-unit field to display units suitable for a manuscript.

    Conversions applied:
        ssr:  J m⁻² → W m⁻²   (÷ 86400)
        tp:   m     → mm d⁻¹  (× 1000)
        msl:  Pa    → hPa      (÷ 100)
        tcc:  0-1   → %        (× 100)
        u10, v10: no change (already m s⁻¹).

    Args:
        arr (np.ndarray): Field in raw physical units.
        var (str): Variable name.

    Returns:
        np.ndarray: Field in display units.

    Example:
        >>> ssr_wm2 = to_display_units(ssr_phys, 'ssr')
    """
    if var == 'ssr':
        return arr / 86400.0
    if var == 'tp':
        return arr * 1000.0
    if var == 'msl':
        return arr / 100.0
    if var == 'tcc':
        return arr * 100.0
    return arr.copy()


# ---------------------------------------------------------------------------
# TIGGE loading
# ---------------------------------------------------------------------------

def load_tigge_ensemble(tigge_dir: Path, month: str, n_members: int) -> np.ndarray:
    """Load and stack all TIGGE member files for a given month.

    Each file has shape (1, 9, 6, 224, 224); the leading day-dimension is
    always 1 for the preprocessed ensemble files, so index 0 is taken directly.

    Args:
        tigge_dir (Path): Directory containing Tigge_{month}_2020_ens_m{nn:02d}.npy.
        month (str): Month label, e.g. 'Jan' or 'Oct'.
        n_members (int): Number of members to load (1-based, default 50).

    Returns:
        np.ndarray: shape (n_members, 9, 6, 224, 224) float32 normalized TIGGE fields.

    Example:
        >>> tigge = load_tigge_ensemble(Path('data/tigge_ensemble'), 'Jan', 50)
        >>> tigge.shape
        (50, 9, 6, 224, 224)
    """
    members = []
    for m in range(1, n_members + 1):
        path = tigge_dir / f'Tigge_{month}_2020_ens_m{m:02d}.npy'
        arr  = np.load(path).astype(np.float32)    # (1, 9, 6, 224, 224)
        members.append(arr[0])                      # (9, 6, 224, 224)
    return np.stack(members, axis=0)                # (N, 9, 6, 224, 224)


# ---------------------------------------------------------------------------
# Field processing
# ---------------------------------------------------------------------------

def process_atm_field(
    arr_224: np.ndarray,
    var: str,
    stats: dict,
    is_std: bool = False,
) -> np.ndarray:
    """Denormalize, convert units, and interpolate one 224×224 atmospheric field.

    For std fields the denormalization multiplies only by the std (no mean added),
    then unit conversion is applied consistently.

    Args:
        arr_224 (np.ndarray): shape (224, 224) field in normalized space.
        var (str): Variable name.
        stats (dict): Output of load_atm_norm_stats().
        is_std (bool): True when processing a spread (std) field; mean is not
            added, only the std scale factor is applied.

    Returns:
        np.ndarray: shape (229, 265) field in display units.

    Example:
        >>> field = process_atm_field(ens_mean[2, 0], 'ssr', stats, is_std=False)
    """
    if var in _ZSCORE_VARS:
        std_224 = interp_stat_to_224(stats[var]['std'])
        if is_std:
            phys = arr_224 * std_224          # spread in physical units
        else:
            mean_224 = interp_stat_to_224(stats[var]['mean'])
            phys = arr_224 * std_224 + mean_224
    else:
        phys = arr_224.copy()                 # already physical

    phys = to_display_units(phys, var)
    return interp_to_glorys(phys)             # (229, 265)


# ---------------------------------------------------------------------------
# Colour range helpers
# ---------------------------------------------------------------------------

def compute_ranges(panels: list[np.ndarray]) -> tuple[float, float]:
    """Return the [p2, p98] range of valid pixels across a list of fields.

    Args:
        panels (list[np.ndarray]): List of (229, 265) arrays, possibly with NaN.

    Returns:
        tuple: (vmin, vmax) float.

    Example:
        >>> vmin, vmax = compute_ranges([field1, field2])
    """
    pooled = np.concatenate([p[np.isfinite(p)].ravel() for p in panels])
    return float(np.percentile(pooled, 2)), float(np.percentile(pooled, 98))


def compute_std_range(panels: list[np.ndarray]) -> float:
    """Return the 98th percentile of spread fields (lower bound fixed at 0).

    Args:
        panels (list[np.ndarray]): List of std (229, 265) arrays.

    Returns:
        float: vmax value for the spread colorbar.

    Example:
        >>> vmax = compute_std_range([std1, std2])
    """
    pooled = np.concatenate([p[np.isfinite(p)].ravel() for p in panels])
    return float(np.percentile(pooled, 98))


# ---------------------------------------------------------------------------
# Map helpers
# ---------------------------------------------------------------------------

def setup_map(ax, left_labels: bool = False, bottom_labels: bool = False) -> None:
    """Configure a cartopy axes for the Bay of Bengal atmospheric domain.

    Args:
        ax: Cartopy GeoAxes with PlateCarree projection.
        left_labels (bool): Draw latitude labels on the left axis.
        bottom_labels (bool): Draw longitude labels on the bottom axis.

    Example:
        >>> setup_map(ax, left_labels=True, bottom_labels=True)
    """
    ax.set_extent([_LON_RANGE[0], _LON_RANGE[1], _LAT_RANGE[0], _LAT_RANGE[1]],
                  crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      facecolor='#d3d3d3', edgecolor='#555555',
                   linewidth=0.4, zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      alpha=0.6, linestyle='--')
    gl.xlocator     = FixedLocator([80, 85, 90, 95, 100])
    gl.ylocator     = FixedLocator([5, 10, 15, 20, 25])
    gl.top_labels    = False
    gl.right_labels  = False
    gl.left_labels   = left_labels
    gl.bottom_labels = bottom_labels
    gl.xlabel_style  = {'size': 6}
    gl.ylabel_style  = {'size': 6}


def draw_panel(ax, data: np.ndarray, lons: np.ndarray, lats: np.ndarray,
               cmap, norm) -> object:
    """Render one map panel with pcolormesh.

    Args:
        ax: Cartopy GeoAxes already configured by setup_map().
        data (np.ndarray): (229, 265) field.
        lons (np.ndarray): 1-D longitude array of length 265.
        lats (np.ndarray): 1-D latitude array of length 229.
        cmap: Matplotlib colormap.
        norm (mcolors.Normalize): Colour normalisation.

    Returns:
        QuadMesh: The pcolormesh object (used for colorbar).

    Example:
        >>> mesh = draw_panel(ax, field, lons, lats, cmap, norm)
    """
    lon2d, lat2d = np.meshgrid(lons, lats)
    return ax.pcolormesh(lon2d, lat2d, data, cmap=cmap, norm=norm,
                         transform=ccrs.PlateCarree(), zorder=1, shading='auto')


def draw_panel_contourf(ax, data: np.ndarray, lons: np.ndarray, lats: np.ndarray,
                        cmap, norm, n_levels: int = 10) -> object:
    """Render one map panel with smooth contourf filling.

    Args:
        ax: Cartopy GeoAxes already configured by setup_map().
        data (np.ndarray): (229, 265) field.
        lons (np.ndarray): 1-D longitude array of length 265.
        lats (np.ndarray): 1-D latitude array of length 229.
        cmap: Matplotlib colormap.
        norm (mcolors.Normalize): Colour normalisation.
        n_levels (int): Number of contour levels (default 10).

    Returns:
        QuadContourSet: The contourf object (used for colorbar).

    Example:
        >>> cf = draw_panel_contourf(ax, field, lons, lats, cmap, norm)
    """
    lon2d, lat2d = np.meshgrid(lons, lats)
    masked = np.ma.masked_invalid(data)
    levels = np.linspace(norm.vmin, norm.vmax, n_levels)
    return ax.contourf(lon2d, lat2d, masked, levels=levels,
                       cmap=cmap, norm=norm,
                       transform=ccrs.PlateCarree(), zorder=1, extend='both')


def get_coords(ocean_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract 1-D longitude and latitude arrays from the GLORYS ocean file.

    Args:
        ocean_file (Path): Path to GLORYS ocean.nc.

    Returns:
        tuple: lons (np.ndarray) length 265, lats (np.ndarray) length 229.

    Example:
        >>> lons, lats = get_coords(Path('ocean.nc'))
    """
    ds = xr.open_dataset(ocean_file)
    da = ds['thetao']
    lons = lats = None
    for name in ('longitude', 'lon', 'x', 'nav_lon'):
        if name in da.coords:
            arr = da.coords[name].values
            lons = arr[0] if arr.ndim > 1 else arr
            break
    for name in ('latitude', 'lat', 'y', 'nav_lat'):
        if name in da.coords:
            arr = da.coords[name].values
            lats = arr[:, 0] if arr.ndim > 1 else arr
            break
    ds.close()
    if lons is None:
        lons = np.linspace(_LON_RANGE[0], _LON_RANGE[1], 265)
    if lats is None:
        lats = np.linspace(_LAT_RANGE[0], _LAT_RANGE[1], 229)
    return lons, lats


# ---------------------------------------------------------------------------
# Figure layout spec
# ---------------------------------------------------------------------------

# (var, is_std, row_label, unit_label, cmap, diverging)
_ROW_SPECS = [
    ('ssr', False, 'SSR mean',  'W m⁻²',  _SOLAR_CMAP, False),
    ('ssr', True,  'SSR spread','W m⁻²',  _STD_CMAP,   False),
    ('tp',  False, 'TP mean',   'mm d⁻¹', _RAIN_CMAP,  False),
    ('tp',  True,  'TP spread', 'mm d⁻¹', _STD_CMAP,   False),
    ('u10', False, 'U10 mean',  'm s⁻¹',  _DIV_CMAP,   True),
    ('u10', True,  'U10 spread','m s⁻¹',  _STD_CMAP,   False),
    ('v10', False, 'V10 mean',  'm s⁻¹',  _DIV_CMAP,   True),
    ('v10', True,  'V10 spread','m s⁻¹',  _STD_CMAP,   False),
    ('msl', False, 'MSL mean',  'hPa',    _SEQ_CMAP,   False),
    ('msl', True,  'MSL spread','hPa',    _STD_CMAP,   False),
    ('tcc', False, 'TCC mean',  '%',      _GREY_CMAP,  False),
    ('tcc', True,  'TCC spread','%',      _STD_CMAP,   False),
]


# ---------------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------------

def build_figure(
    data: dict,
    lons: np.ndarray,
    lats: np.ndarray,
    leads: list[int],
    output_path: str,
    method: str = 'pcolormesh',
    n_levels: int = 10,
    save_pdf: bool = False,
) -> None:
    """Build and save the 12-row × 4-col TIGGE atmospheric forcing figure.

    Layout:
        Cols 0-3: Winter +Xd, Winter +Yd, Post-monsoon +Xd, Post-monsoon +Yd
        Col  4:   vertical colorbar for that row
        Rows 0-11: interleaved mean/spread for ssr, tp, u10, v10, msl, tcc

    Args:
        data (dict): data[season][var]['mean'|'std'][lead] → (229, 265) array.
        lons (np.ndarray): 1-D longitude array.
        lats (np.ndarray): 1-D latitude array.
        leads (list[int]): Two lead days, e.g. [3, 5].
        output_path (str): Base path without extension.
        method (str): 'pcolormesh' or 'contourf'.
        n_levels (int): Contourf levels (ignored for pcolormesh).
        save_pdf (bool): Also save vector PDF.

    Returns:
        None: Saves PNG (and optionally PDF) to output_path.

    Example:
        >>> build_figure(data, lons, lats, [3, 5], 'results/figures/tigge_forcing_comparison')
    """
    seasons = ['Winter', 'PostMonsoon']
    col_labels = [
        f'Winter  +{leads[0]} d', f'Winter  +{leads[1]} d',
        f'Post-monsoon  +{leads[0]} d', f'Post-monsoon  +{leads[1]} d',
    ]
    col_order = [
        ('Winter',      leads[0]),
        ('Winter',      leads[1]),
        ('PostMonsoon', leads[0]),
        ('PostMonsoon', leads[1]),
    ]

    n_rows = len(_ROW_SPECS)  # 12
    n_cols = 4

    # --- Compute per-row colour norms ---
    norms = []
    for var, is_std, row_label, unit_label, cmap, diverging in _ROW_SPECS:
        panels = [data[s][var]['std' if is_std else 'mean'][l]
                  for s in seasons for l in leads]
        if is_std:
            vmax = compute_std_range(panels)
            norm = mcolors.Normalize(vmin=0, vmax=vmax)
        elif diverging:
            vmin, vmax = compute_ranges(panels)
            bound = max(abs(vmin), abs(vmax))
            norm = mcolors.Normalize(vmin=-bound, vmax=bound)
        else:
            vmin, vmax = compute_ranges(panels)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        norms.append(norm)

    # --- Figure ---
    fig = plt.figure(figsize=(15, 2.5 * n_rows))
    gs  = GridSpec(n_rows, n_cols + 1,
                   width_ratios=[1, 1, 1, 1, 0.045],
                   hspace=0.08, wspace=0.05,
                   top=0.96, bottom=0.03,
                   left=0.08, right=0.97)

    axes = {}

    for r, (var, is_std, row_label, unit_label, cmap, _) in enumerate(_ROW_SPECS):
        norm     = norms[r]
        row_mesh = None

        for c, (season, lead) in enumerate(col_order):
            is_left   = (c == 0)
            is_bottom = (r == n_rows - 1)
            ax = fig.add_subplot(gs[r, c], projection=ccrs.PlateCarree())
            axes[(r, c)] = ax
            setup_map(ax, left_labels=is_left, bottom_labels=is_bottom)
            field = data[season][var]['std' if is_std else 'mean'][lead]
            if method == 'contourf':
                row_mesh = draw_panel_contourf(ax, field, lons, lats, cmap, norm, n_levels)
            else:
                row_mesh = draw_panel(ax, field, lons, lats, cmap, norm)
            if r == 0:
                ax.set_title(col_labels[c], fontsize=9, fontweight='bold', pad=4)

        # Row label on the far left
        pos   = axes[(r, 0)].get_position()
        mid_y = (pos.y0 + pos.y1) / 2
        fig.text(pos.x0 - 0.05, mid_y, f'{row_label}\n({unit_label})',
                 ha='center', va='center', fontsize=7,
                 rotation=90, transform=fig.transFigure)

        # Vertical colorbar in the 5th column
        cax = fig.add_subplot(gs[r, n_cols])
        cb  = fig.colorbar(row_mesh, cax=cax, orientation='vertical')
        cb.set_label(unit_label, fontsize=6)
        cb.ax.tick_params(labelsize=5)

    # --- Save ---
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    png_path = f'{output_path}.png'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {png_path}')
    if save_pdf:
        pdf_path = f'{output_path}.pdf'
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f'Saved: {pdf_path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments, load TIGGE ensemble data, and produce the figure.

    Args:
        None: All parameters read from sys.argv via argparse.

    Returns:
        None: Saves PNG (and optionally PDF) to --output.

    Example:
        >>> # python src/visualization/plot_tigge_forcing_comparison.py \\
        >>> #     --tigge_dir data/tigge_ensemble \\
        >>> #     --mean_dir data/1993_2020/mean \\
        >>> #     --ocean_file data/1993_2020/ocean.nc \\
        >>> #     --output results/figures/tigge_forcing_comparison \\
        >>> #     --leads 3 5 --levels 10 --pdf
    """
    parser = argparse.ArgumentParser(
        description='Manuscript figure: TIGGE atmospheric forcing mean and spread'
    )
    parser.add_argument('--tigge_dir',  required=True,
                        help='Directory with Tigge_*_2020_ens_m*.npy files')
    parser.add_argument('--jan_month',  default='Jan',
                        help='Month label for Jan files (default: Jan)')
    parser.add_argument('--oct_month',  default='Oct',
                        help='Month label for Oct files (default: Oct)')
    parser.add_argument('--mean_dir',   default='data/1993_2020/mean',
                        help='Directory with normalization mean/var .npy files')
    parser.add_argument('--ocean_file', default='data/1993_2020/ocean.nc',
                        help='GLORYS ocean.nc for grid coordinates')
    parser.add_argument('--output',     default='results/figures/tigge_forcing_comparison',
                        help='Output path without extension')
    parser.add_argument('--leads',     nargs='+', type=int, default=[3, 5],
                        help='Two lead days to plot (default: 3 5)')
    parser.add_argument('--n_members', type=int, default=50,
                        help='Number of TIGGE members per season (default: 50)')
    parser.add_argument('--levels',    type=int, default=10,
                        help='Number of contourf levels (default: 10)')
    parser.add_argument('--pdf',       action='store_true',
                        help='Also save a vector PDF alongside the PNG')
    args = parser.parse_args()

    if len(args.leads) != 2:
        parser.error('--leads must specify exactly two lead days, e.g. --leads 3 5')

    tigge_dir  = Path(args.tigge_dir)
    mean_dir   = Path(args.mean_dir)
    ocean_file = Path(args.ocean_file)
    leads      = args.leads
    lead_idx   = {l: l - 1 for l in leads}

    # --- Load TIGGE ensembles ---
    print('Loading TIGGE ensemble (Jan) ...')
    jan_all = load_tigge_ensemble(tigge_dir, args.jan_month, args.n_members)
    print('Loading TIGGE ensemble (Oct) ...')
    oct_all = load_tigge_ensemble(tigge_dir, args.oct_month, args.n_members)

    jan_mean_raw = jan_all.mean(axis=0)   # (9, 6, 224, 224)
    jan_std_raw  = jan_all.std(axis=0)
    oct_mean_raw = oct_all.mean(axis=0)
    oct_std_raw  = oct_all.std(axis=0)

    # --- Load norm stats ---
    print('Loading normalization stats ...')
    stats = load_atm_norm_stats(mean_dir)

    # --- Grid coordinates ---
    print('Loading grid coordinates ...')
    lons, lats = get_coords(ocean_file)

    # --- Process all panels ---
    print('Processing fields ...')
    data = {}
    season_configs = [
        ('Winter',      jan_mean_raw, jan_std_raw),
        ('PostMonsoon', oct_mean_raw, oct_std_raw),
    ]

    for season, mean_raw, std_raw in season_configs:
        data[season] = {}
        for var in _ATM_VARS:
            ch = _ATM_CH[var]
            data[season][var] = {'mean': {}, 'std': {}}
            for lead in leads:
                li = lead_idx[lead]
                mean_field = process_atm_field(mean_raw[li, ch], var, stats, is_std=False)
                std_field  = process_atm_field(std_raw[li,  ch], var, stats, is_std=True)
                data[season][var]['mean'][lead] = mean_field
                data[season][var]['std'][lead]  = std_field
                print(f'  {season:12s} {var:4s} +{lead}d: '
                      f'mean={np.nanmean(mean_field):.3f}  '
                      f'spread={np.nanmean(std_field):.4f}')

    # --- Build figures ---
    print('Building pcolormesh figure ...')
    build_figure(data, lons, lats, leads, args.output,
                 method='pcolormesh', n_levels=args.levels, save_pdf=args.pdf)

    print('Building contourf figure ...')
    build_figure(data, lons, lats, leads, f'{args.output}_contourf',
                 method='contourf', n_levels=args.levels, save_pdf=args.pdf)


if __name__ == '__main__':
    main()

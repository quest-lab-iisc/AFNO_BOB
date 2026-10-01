"""Manuscript figure: ensemble mean and spread for SST and SSS across seasons and lead times.

Produces a 4-row × 4-column panel figure comparing the IC-perturbed ensemble forecast
for two Bay of Bengal seasons (Winter and Post-monsoon) at two lead times (+3 d and +5 d).
Rows show ensemble mean and ensemble spread (std) for SST (thetao) and SSS (so).
Both a pcolormesh (discrete cells) and a contourf (smooth) version are produced.

Inputs:
    --jan_dir (str): Directory containing Winter ensemble_mean.npy and ensemble_std.npy.
    --oct_dir (str): Directory containing Post-monsoon ensemble_mean.npy and ensemble_std.npy.
    --mean_dir (str): Directory with mean_{var}_1993_2018_all_months.npy normalization files.
    --ocean_file (str): Path to GLORYS ocean.nc (used for land mask and coordinates).
    --output (str): Output path without extension.
    --leads (int ...): Lead days to plot (default: 3 5).
    --levels (int): Number of contourf levels (default: 10).
    --pdf (flag): Also save PDF in addition to the default PNG.

Outputs:
    {output}.png               — 300 dpi raster (pcolormesh)
    {output}_contourf.png      — 300 dpi raster (contourf)
    {output}.pdf               — vector PDF (pcolormesh), only with --pdf
    {output}_contourf.pdf      — vector PDF (contourf), only with --pdf

Example:
    conda activate BoB_Surf_2
    python src/visualization/plot_ensemble_season_comparison.py \\
        --jan_dir results/AFNO_BoB_Surf_E11p1/ensemble_ic_Jan_14012020 \\
        --oct_dir results/AFNO_BoB_Surf_E11p1/ensemble_ic_Oct_14102020 \\
        --mean_dir data/1993_2020/mean \\
        --ocean_file data/1993_2020/ocean.nc \\
        --output results/figures/ensemble_season_comparison \\
        --levels 10 --pdf
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator
from scipy.interpolate import griddata

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

sys.path.append(str(Path(__file__).parent.parent))

from inference.run_ensemble_verification import (
    load_norm_stats,
    interp_to_glorys,
    denormalize,
)

try:
    import cmocean
    _SST_CMAP  = cmocean.cm.thermal
    _SSS_CMAP  = cmocean.cm.haline
    _STD_CMAP  = cmocean.cm.amp
except ImportError:
    _SST_CMAP  = 'RdYlBu_r'
    _SSS_CMAP  = 'viridis'
    _STD_CMAP  = 'YlOrRd'

_LON_RANGE = (77, 99)
_LAT_RANGE = (4, 23)

# Fixed channel and variable order matching training
_THETAO_IDX = 0
_SO_IDX     = 1


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_ensemble(ensemble_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load ensemble mean, std, and metadata from a directory.

    Args:
        ensemble_dir (Path): Directory produced by run_ensemble_inference_ic.py.

    Returns:
        tuple:
            mean (np.ndarray): shape (9, 5, 224, 224), normalized.
            std  (np.ndarray): shape (9, 5, 224, 224), normalized units.
            meta (dict): ensemble_metadata.json contents.

    Example:
        >>> mean, std, meta = load_ensemble(Path('results/.../ensemble_ic_Jan_14012020'))
    """
    mean = np.load(ensemble_dir / 'ensemble_mean.npy')
    std  = np.load(ensemble_dir / 'ensemble_std.npy')
    meta = json.loads((ensemble_dir / 'ensemble_metadata.json').read_text())
    return mean, std, meta


def get_land_mask(mean_dict: dict) -> np.ndarray:
    """Return a boolean mask (True = invalid) from the thetao climatological mean.

    Uses the climatological mean NaN pattern: pixels that are NaN in the thetao
    climatological mean are permanently land in GLORYS and should not be plotted.
    No explicit north-row masking is applied — the model's zeroed north rows correspond
    to the Bangladesh/Myanmar delta region which is already NaN in the climatological mean.

    Args:
        mean_dict (dict): Output of load_norm_stats(); contains 'thetao' key with
            shape (1, 229, 265) float32 array (NaN on land).

    Returns:
        np.ndarray: Boolean array of shape (229, 265); True where data is invalid.

    Example:
        >>> mask = get_land_mask(mean_dict)
        >>> mask.shape
        (229, 265)
    """
    return np.isnan(np.squeeze(mean_dict['thetao']))   # (229, 265)


def get_coords(ocean_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract 1-D longitude and latitude arrays from the GLORYS ocean file.

    Args:
        ocean_file (Path): Path to GLORYS ocean.nc.

    Returns:
        tuple: lons (np.ndarray) of length 265, lats (np.ndarray) of length 229.

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


_NORTH_FILL_ROWS = 20  # northernmost rows in GLORYS (229×265) to fill by interpolation


def fill_north_rows(field: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    """Fill the northernmost rows of field using nearest-neighbour interpolation.

    The model forces pred[:, -20:, :] to zero during training and inference.
    After bilinear interpolation to 229×265 those rows carry near-zero artefacts.
    This function replaces ocean pixels in the fill zone with values extrapolated
    from valid ocean pixels immediately to the south.

    Args:
        field (np.ndarray): shape (229, 265); may contain zero-artefacts in north rows.
        land_mask (np.ndarray): boolean (229, 265); True = land.

    Returns:
        np.ndarray: field with north fill-zone ocean pixels replaced by interpolated
            values; land pixels and southern pixels are unchanged.

    Example:
        >>> filled = fill_north_rows(interped_field, land_mask)
    """
    out = field.copy()

    # Source pixels: valid ocean pixels south of the fill zone
    src = ~land_mask & ~np.isnan(out)
    src[-_NORTH_FILL_ROWS:, :] = False

    # Target pixels: ocean pixels inside the fill zone
    tgt = np.zeros_like(land_mask)
    tgt[-_NORTH_FILL_ROWS:, :] = True
    tgt &= ~land_mask

    if src.any() and tgt.any():
        ys, xs = np.where(src)
        src_pts = np.stack([ys, xs], axis=1)
        src_vals = out[ys, xs]

        fy, fx = np.where(tgt)
        tgt_pts = np.stack([fy, fx], axis=1)

        filled = griddata(src_pts, src_vals, tgt_pts, method='cubic')

        # griddata cubic returns NaN outside the convex hull; fall back to nearest
        nan_idx = np.isnan(filled)
        if nan_idx.any():
            filled[nan_idx] = griddata(src_pts, src_vals, tgt_pts[nan_idx], method='nearest')

        out[fy, fx] = filled

    return out


def process_field(
    arr_224: np.ndarray,
    var: str,
    mean_dict: dict,
    land_mask: np.ndarray,
    denorm: bool = True,
    apply_north_fill: bool = True,
    mask_north_rows: bool = False,
) -> np.ndarray:
    """Interpolate, optionally denormalize, fill or mask north rows, and apply land mask.

    Args:
        arr_224 (np.ndarray): Normalized model output, shape (224, 224).
        var (str): Variable name, e.g. 'thetao'.
        mean_dict (dict): Climatological means from load_norm_stats().
        land_mask (np.ndarray): Boolean mask (229, 265); True = land.
        denorm (bool): If True, add climatological mean (for mean panels).
            Set False for std panels where normalization is mean-only.
        apply_north_fill (bool): If True, replace the northernmost GLORYS rows with
            values interpolated from the south (for models with north_mask_rows > 0).
        mask_north_rows (bool): If True, set the northernmost _NORTH_FILL_ROWS GLORYS
            rows to NaN. Use for spread fields when north_mask_rows=0 — those rows
            are dominated by land-filled inputs that collapse member spread to near-zero,
            producing a spurious low-spread band at the top of the domain.

    Returns:
        np.ndarray: Float32 field of shape (229, 265); NaN on land.

    Example:
        >>> field = process_field(arr, 'thetao', mean_dict, mask, denorm=True)
    """
    out = interp_to_glorys(arr_224)          # (229, 265)
    if denorm:
        out = denormalize(out, var, mean_dict)
    if apply_north_fill:
        out = fill_north_rows(out, land_mask)
    if mask_north_rows:
        out[-_NORTH_FILL_ROWS:, :] = np.nan
    out[land_mask] = np.nan
    return out


# ---------------------------------------------------------------------------
# Color range helpers
# ---------------------------------------------------------------------------

def compute_ranges(panels: list[np.ndarray]) -> tuple[float, float]:
    """Compute the 2nd and 98th percentile range over a list of 2-D fields.

    Args:
        panels (list[np.ndarray]): List of (229, 265) arrays, possibly with NaN.

    Returns:
        tuple: (vmin, vmax) derived from pooled valid pixels.

    Example:
        >>> vmin, vmax = compute_ranges([field1, field2])
    """
    pooled = np.concatenate([p[~np.isnan(p)].ravel() for p in panels])
    return float(np.percentile(pooled, 2)), float(np.percentile(pooled, 98))


def compute_std_range(panels: list[np.ndarray]) -> float:
    """Compute the 98th percentile of spread fields (lower bound fixed at 0).

    Args:
        panels (list[np.ndarray]): List of std (229, 265) arrays.

    Returns:
        float: vmax value for the spread colorbar.

    Example:
        >>> vmax = compute_std_range([std1, std2])
    """
    pooled = np.concatenate([p[~np.isnan(p)].ravel() for p in panels])
    return float(np.percentile(pooled, 98))


# ---------------------------------------------------------------------------
# Map helpers
# ---------------------------------------------------------------------------

def setup_map(ax, left_labels: bool = False, bottom_labels: bool = False) -> None:
    """Configure a cartopy axes for the Bay of Bengal domain.

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
        data (np.ndarray): (229, 265) field; NaN pixels are transparent.
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

    NaN pixels (land) are masked before contouring so they appear transparent,
    letting the LAND cartopy feature show through.

    Args:
        ax: Cartopy GeoAxes already configured by setup_map().
        data (np.ndarray): (229, 265) field; NaN pixels are land.
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
    """Build and save the 4×4 manuscript comparison figure with right-side colorbars.

    Layout (rows × cols):
        Cols 0-3: map panels — Winter +Xd, Winter +Yd, PostMon +Xd, PostMon +Yd
        Col  4:   vertical colourbar for that row
        Row  0:   SST ensemble mean
        Row  1:   SST ensemble spread
        Row  2:   SSS ensemble mean
        Row  3:   SSS ensemble spread

    Args:
        data (dict): Nested dict data[season][field][var][lead] → (229, 265) array.
            season: 'Winter' | 'PostMonsoon'
            field:  'mean' | 'std'
            var:    'thetao' | 'so'
            lead:   int (lead day, e.g. 3)
        lons (np.ndarray): 1-D longitude array.
        lats (np.ndarray): 1-D latitude array.
        leads (list[int]): Two lead days to plot, e.g. [3, 5].
        output_path (str): Base path without extension.
        method (str): Rendering method — 'pcolormesh' (discrete cells) or
            'contourf' (smooth filled contours). Default 'pcolormesh'.
        n_levels (int): Number of contour levels for contourf. Default 10.
        save_pdf (bool): Also save a vector PDF in addition to the PNG. Default False.

    Returns:
        None: Always saves {output_path}.png at 300 dpi; optionally {output_path}.pdf.

    Example:
        >>> build_figure(data, lons, lats, [3, 5], 'results/figures/ensemble_season_comparison')
        >>> build_figure(data, lons, lats, [3, 5], 'results/figures/ensemble_season_comparison_contourf', method='contourf', n_levels=10, save_pdf=True)
    """
    seasons  = ['Winter', 'PostMonsoon']
    col_labels = [
        f'Winter  +{leads[0]} d', f'Winter  +{leads[1]} d',
        f'Post-monsoon  +{leads[0]} d', f'Post-monsoon  +{leads[1]} d',
    ]
    row_labels  = ['SST mean (°C)', 'SST spread (°C)', 'SSS mean (PSU)', 'SSS spread (PSU)']
    cbar_labels = ['°C', '°C', 'PSU', 'PSU']

    # --- Compute shared color ranges (pooled across all panels in each row) ---
    sst_mean_panels = [data[s]['mean']['thetao'][l] for s in seasons for l in leads]
    sst_std_panels  = [data[s]['std']['thetao'][l]  for s in seasons for l in leads]
    sss_mean_panels = [data[s]['mean']['so'][l]      for s in seasons for l in leads]
    sss_std_panels  = [data[s]['std']['so'][l]       for s in seasons for l in leads]

    sst_vmin, sst_vmax = compute_ranges(sst_mean_panels)
    sss_vmin, sss_vmax = compute_ranges(sss_mean_panels)
    sst_std_vmax = compute_std_range(sst_std_panels)
    sss_std_vmax = compute_std_range(sss_std_panels)

    sst_norm  = mcolors.Normalize(vmin=sst_vmin,  vmax=sst_vmax)
    sss_norm  = mcolors.Normalize(vmin=sss_vmin,  vmax=sss_vmax)
    sst_snorm = mcolors.Normalize(vmin=0,          vmax=sst_std_vmax)
    sss_snorm = mcolors.Normalize(vmin=0,          vmax=sss_std_vmax)

    row_specs = [
        ('thetao', 'mean', _SST_CMAP, sst_norm),
        ('thetao', 'std',  _STD_CMAP, sst_snorm),
        ('so',     'mean', _SSS_CMAP, sss_norm),
        ('so',     'std',  _STD_CMAP, sss_snorm),
    ]

    # --- Figure and GridSpec ---
    # 4 equal map columns + 1 narrow colorbar column
    n_rows = 4
    n_cols = 4
    fig = plt.figure(figsize=(15, 3.5 * n_rows))
    gs  = GridSpec(n_rows, n_cols + 1,
                   width_ratios=[1, 1, 1, 1, 0.045],
                   hspace=0.12, wspace=0.05,
                   top=0.93, bottom=0.06,
                   left=0.07, right=0.97)

    # --- Map panels ---
    col_order = [
        ('Winter',      leads[0]),
        ('Winter',      leads[1]),
        ('PostMonsoon', leads[0]),
        ('PostMonsoon', leads[1]),
    ]

    axes = {}   # (r, c) → axes object

    for r, (var, field, cmap, norm) in enumerate(row_specs):
        row_mesh = None
        for c, (season, lead) in enumerate(col_order):
            is_left   = (c == 0)
            is_bottom = (r == n_rows - 1)
            ax = fig.add_subplot(gs[r, c], projection=ccrs.PlateCarree())
            axes[(r, c)] = ax
            setup_map(ax, left_labels=is_left, bottom_labels=is_bottom)
            panel_data = data[season][field][var][lead]
            if method == 'contourf':
                row_mesh = draw_panel_contourf(ax, panel_data, lons, lats, cmap, norm, n_levels)
            else:
                row_mesh = draw_panel(ax, panel_data, lons, lats, cmap, norm)
            if r == 0:
                ax.set_title(col_labels[c], fontsize=9, fontweight='bold', pad=4)

        # Row label on the far left
        pos = axes[(r, 0)].get_position()
        mid_y = (pos.y0 + pos.y1) / 2
        fig.text(pos.x0 - 0.045, mid_y, row_labels[r],
                 ha='center', va='center', fontsize=8,
                 rotation=90, transform=fig.transFigure)

        # Vertical colorbar in the 5th column, aligned to this row
        cax = fig.add_subplot(gs[r, n_cols])
        cb  = fig.colorbar(row_mesh, cax=cax, orientation='vertical')
        cb.set_label(cbar_labels[r], fontsize=7)
        cb.ax.tick_params(labelsize=6)

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
    """Parse CLI arguments, load ensemble data, and produce the manuscript figure.

    Args:
        None: All parameters read from sys.argv via argparse.

    Returns:
        None: Always saves PNG at 300 dpi; also PDF if --pdf is passed.

    Example:
        >>> # python src/visualization/plot_ensemble_season_comparison.py \\
        >>> #     --jan_dir results/AFNO_BoB_Surf_E11p1/ensemble_ic_Jan_14012020 \\
        >>> #     --oct_dir results/AFNO_BoB_Surf_E11p1/ensemble_ic_Oct_14102020 \\
        >>> #     --mean_dir data/1993_2020/mean \\
        >>> #     --ocean_file data/1993_2020/ocean.nc \\
        >>> #     --output results/figures/ensemble_season_comparison \\
        >>> #     --levels 10 --pdf
    """
    parser = argparse.ArgumentParser(
        description='Manuscript figure: ensemble SST/SSS mean and spread by season and lead'
    )
    parser.add_argument('--jan_dir',    required=True,
                        help='Winter ensemble directory (Jan IC)')
    parser.add_argument('--oct_dir',    required=True,
                        help='Post-monsoon ensemble directory (Oct IC)')
    parser.add_argument('--mean_dir',   default='data/1993_2020/mean',
                        help='Directory with normalization mean .npy files')
    parser.add_argument('--ocean_file', default='data/1993_2020/ocean.nc',
                        help='GLORYS ocean.nc for land mask and coordinates')
    parser.add_argument('--output',     default='results/figures/ensemble_season_comparison',
                        help='Output path without extension')
    parser.add_argument('--leads',  nargs='+', type=int, default=[3, 5],
                        help='Two lead days to plot (default: 3 5)')
    parser.add_argument('--levels', type=int, default=10,
                        help='Number of contourf levels (default: 10)')
    parser.add_argument('--pdf',    action='store_true',
                        help='Also save a vector PDF alongside the PNG')
    args = parser.parse_args()

    if len(args.leads) != 2:
        parser.error('--leads must specify exactly two lead days, e.g. --leads 3 5')

    jan_dir  = Path(args.jan_dir)
    oct_dir  = Path(args.oct_dir)
    mean_dir = Path(args.mean_dir)
    ocean_file = Path(args.ocean_file)
    leads    = args.leads

    # --- Load ensemble arrays and metadata ---
    print('Loading ensemble arrays ...')
    jan_mean, jan_std, jan_meta = load_ensemble(jan_dir)
    oct_mean, oct_std, oct_meta = load_ensemble(oct_dir)

    # north_mask_rows=0 means the model produced valid output in north rows,
    # so fill_north_rows must NOT be applied (it would overwrite genuine values).
    jan_north_fill = jan_meta.get('north_mask_rows', 20) > 0
    oct_north_fill = oct_meta.get('north_mask_rows', 20) > 0
    print(f'North-row fill: Winter={jan_north_fill}  PostMonsoon={oct_north_fill}')

    # --- Lead indices (0-based): lead day 3 → index 2, lead day 5 → index 4 ---
    lead_indices = {l: l - 1 for l in leads}

    # --- Load normalization stats ---
    print('Loading normalization stats ...')
    mean_dict = load_norm_stats(mean_dir)

    # --- Land mask from climatological mean NaN pattern + north boundary rows ---
    print('Loading land mask ...')
    land_mask = get_land_mask(mean_dict)
    jan_mask = land_mask
    oct_mask = land_mask

    # --- Grid coordinates ---
    print('Loading grid coordinates ...')
    lons, lats = get_coords(ocean_file)

    # --- Process all panels ---
    # data[season][field][var][lead] → (229, 265) physical-unit field
    season_configs = [
        ('Winter',      jan_mean, jan_std, jan_mask, jan_north_fill),
        ('PostMonsoon', oct_mean, oct_std, oct_mask, oct_north_fill),
    ]
    var_configs = [
        ('thetao', _THETAO_IDX),
        ('so',     _SO_IDX),
    ]

    print('Processing fields ...')
    data = {}
    for season, ens_mean, ens_std, mask, north_fill in season_configs:
        data[season] = {'mean': {}, 'std': {}}
        for var, ch in var_configs:
            data[season]['mean'][var] = {}
            data[season]['std'][var]  = {}
            for lead in leads:
                li = lead_indices[lead]
                mean_field = process_field(ens_mean[li, ch], var, mean_dict, mask,
                                           denorm=True,  apply_north_fill=north_fill)
                std_field  = process_field(ens_std[li,  ch], var, mean_dict, mask,
                                           denorm=False, apply_north_fill=north_fill,
                                           mask_north_rows=not north_fill)
                data[season]['mean'][var][lead] = mean_field
                data[season]['std'][var][lead]  = std_field
                print(f'  {season:12s} {var} +{lead}d: '
                      f'mean={np.nanmean(mean_field):.2f}  '
                      f'spread_mean={np.nanmean(std_field):.4f}')

    # --- Build figures ---
    print('Building pcolormesh figure ...')
    build_figure(data, lons, lats, leads, args.output,
                 method='pcolormesh', n_levels=args.levels, save_pdf=args.pdf)

    print('Building contourf figure ...')
    build_figure(data, lons, lats, leads, f'{args.output}_contourf',
                 method='contourf', n_levels=args.levels, save_pdf=args.pdf)


if __name__ == '__main__':
    main()

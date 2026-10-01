"""Ensemble mean and spread comparison figure for publication (fig6).

Loads ensemble_predictions.npy for two seasons, computes mean and standard
deviation across members in-memory, denormalizes (thetao, so: add climatological
mean), applies a land mask, and plots a 4-row × 4-column panel figure.

No special north-row handling — the raw model predictions are used as-is.

Inputs:
    --jan_dir (str)     : Directory with Winter ensemble_predictions.npy.
    --oct_dir (str)     : Directory with Post-monsoon ensemble_predictions.npy.
    --mean_dir (str)    : Directory with mean_{var}_1993_2018_all_months.npy files.
    --ocean_file (str)  : Path to GLORYS ocean.nc (for lat/lon grid).
    --output (str)      : Output path stem without extension.
    --leads (int int)   : Two lead days to plot (default: 3 5).
    --pdf               : Also save a PDF.

Outputs:
    {output}.png  — 300 dpi raster figure.
    {output}.pdf  — vector PDF (if --pdf is set).

Example:
    conda activate BoB_Surf_2
    python src/visualization/plot_ensemble_comparison.py \\
        --jan_dir results/AFNO_BoB_Surf_E14/ensemble_ic_Jan_14012020 \\
        --oct_dir results/AFNO_BoB_Surf_E14/ensemble_ic_Oct_14102020 \\
        --mean_dir data/1993_2020/mean \\
        --ocean_file data/1993_2020/ocean.nc \\
        --output results/figures/fig6_ensemble_season_comparison \\
        --leads 3 5 --pdf
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

try:
    import cmocean
    _SST_CMAP = cmocean.cm.thermal
    _SSS_CMAP = cmocean.cm.haline
    _STD_CMAP = cmocean.cm.amp
except ImportError:
    _SST_CMAP = 'RdYlBu_r'
    _SSS_CMAP = 'viridis'
    _STD_CMAP = 'YlOrRd'

_LON_RANGE = (77, 99)
_LAT_RANGE = (4, 23)
_MODEL_H, _MODEL_W = 224, 224

# Channel indices in the 5-channel model output (thetao, so, uo, vo, zos)
_VAR_IDX = {'thetao': 0, 'so': 1, 'uo': 2, 'vo': 3, 'zos': 4}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_predictions(
    ensemble_dir: Path,
    clim_means: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Load ensemble_predictions.npy, denormalize to physical units, then compute mean and std.

    Denormalization is applied per-member before aggregation so that the std is
    computed in physical space.  For thetao and so the normalization was mean-
    subtraction, so the clim_mean (shape 224×224, NaN on land) is added back to
    each member.  NaN propagates from land pixels in the clim_mean through the
    addition, making land pixels NaN in every member — std is therefore naturally
    NaN on land without any explicit masking.  uo, vo, and zos were not
    normalized and are returned unchanged.

    Args:
        ensemble_dir (Path): Directory containing ensemble_predictions.npy with
            shape (N_members, 9, 5, 224, 224) in normalized model units.
        clim_means (dict): Maps variable name → np.ndarray (224, 224) with NaN
            on land.  Must contain keys 'thetao' and 'so'.

    Returns:
        tuple:
            mean (np.ndarray): shape (9, 5, 224, 224), physical units, NaN on land.
            std  (np.ndarray): shape (9, 5, 224, 224), physical units, NaN on land.

    Example:
        >>> mean, std = load_predictions(
        ...     Path('results/AFNO_BoB_Surf_E14/ensemble_ic_Jan_14012020'),
        ...     clim_means={'thetao': mean_thetao_224, 'so': mean_so_224})
        >>> mean.shape
        (9, 5, 224, 224)
    """
    preds = np.load(ensemble_dir / 'ensemble_predictions.npy').astype(np.float32)
    print(f'  Loaded {preds.shape[0]} members, shape {preds.shape}')

    # Denormalize in-place per variable: add clim_mean (NaN on land) to each member.
    # NaN propagates → land pixels become NaN across all members → std = NaN on land.
    for var, ch in [('thetao', 0), ('so', 1)]:
        cm = clim_means[var].astype(np.float32)          # (224, 224), NaN on land
        preds[:, :, ch, :, :] += cm[np.newaxis, np.newaxis]

    mean = np.nanmean(preds, axis=0)   # (9, 5, 224, 224)
    std  = np.nanstd(preds, axis=0)    # (9, 5, 224, 224), NaN where all members are NaN
    return mean, std


def load_glorys_coords(ocean_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load 1-D latitude and longitude arrays from a GLORYS ocean.nc file.

    Args:
        ocean_file (Path): Path to GLORYS NetCDF file with thetao variable.

    Returns:
        tuple: lons (np.ndarray) shape (265,), lats (np.ndarray) shape (229,).

    Example:
        >>> lons, lats = load_glorys_coords(Path('data/1993_2020/ocean.nc'))
    """
    ds = xr.open_dataset(ocean_file)
    lons = ds['longitude'].values  # (265,)
    lats = ds['latitude'].values   # (229,)
    ds.close()
    return lons, lats


def resize_to_model(arr_glorys: np.ndarray) -> np.ndarray:
    """Bilinearly resize a (229, 265) GLORYS field to (224, 224) model grid.

    Args:
        arr_glorys (np.ndarray): shape (229, 265), may contain NaN.

    Returns:
        np.ndarray: shape (224, 224) float32.

    Example:
        >>> arr_224 = resize_to_model(arr_229)
    """
    t = torch.tensor(arr_glorys[np.newaxis, np.newaxis], dtype=torch.float32)
    t = F.interpolate(t, size=(_MODEL_H, _MODEL_W), mode='bilinear', align_corners=False)
    return t.squeeze().numpy()


def load_clim_mean_224(mean_dir: Path, var: str) -> np.ndarray:
    """Load climatological mean for a variable and resize to 224×224.

    Args:
        mean_dir (Path): Directory with mean_{var}_1993_2018_all_months.npy files.
        var (str): Variable name (e.g. 'thetao', 'so').

    Returns:
        np.ndarray: shape (224, 224) float32 climatological mean; NaN where land.

    Example:
        >>> mean_224 = load_clim_mean_224(Path('/data/mean'), 'thetao')
    """
    path = mean_dir / f'mean_{var}_1993_2018_all_months.npy'
    arr = np.squeeze(np.load(path)).astype(np.float32)  # (229, 265)
    return resize_to_model(arr)


def get_land_mask_224(mean_dir: Path) -> np.ndarray:
    """Return a boolean land mask at 224×224 from the thetao climatological mean NaN pattern.

    Args:
        mean_dir (Path): Directory with mean_thetao_1993_2018_all_months.npy.

    Returns:
        np.ndarray: Boolean (224, 224); True where land.

    Example:
        >>> mask = get_land_mask_224(Path('/data/mean'))
    """
    mean_thetao = load_clim_mean_224(mean_dir, 'thetao')  # NaN on land
    return np.isnan(mean_thetao)


def denormalize(field_224: np.ndarray, var: str, clim_mean_224: np.ndarray) -> np.ndarray:
    """Convert normalized model output to physical units by adding climatological mean.

    Only thetao and so were normalized by mean subtraction; uo, vo, zos are returned
    unchanged.

    Args:
        field_224 (np.ndarray): shape (224, 224) normalized model output.
        var (str): Variable name.
        clim_mean_224 (np.ndarray): shape (224, 224) climatological mean.

    Returns:
        np.ndarray: shape (224, 224) in physical units.

    Example:
        >>> sst = denormalize(pred_norm, 'thetao', clim_mean)
    """
    if var in ('thetao', 'so'):
        return field_224 + clim_mean_224
    return field_224.copy()


# ---------------------------------------------------------------------------
# Color range helpers
# ---------------------------------------------------------------------------

def pooled_range(panels: list[np.ndarray]) -> tuple[float, float]:
    """2nd–98th percentile range over a list of masked 2-D fields.

    Args:
        panels (list[np.ndarray]): List of (224, 224) arrays, NaN on land.

    Returns:
        tuple: (vmin, vmax).

    Example:
        >>> vmin, vmax = pooled_range([sst_panel1, sst_panel2])
    """
    pooled = np.concatenate([p[~np.isnan(p)].ravel() for p in panels])
    return float(np.percentile(pooled, 2)), float(np.percentile(pooled, 98))


def pooled_vmax(panels: list[np.ndarray]) -> float:
    """98th percentile over a list of spread fields (vmin always 0).

    Args:
        panels (list[np.ndarray]): List of (224, 224) std arrays, NaN on land.

    Returns:
        float: vmax for spread colorbar.

    Example:
        >>> vmax = pooled_vmax([std_panel1, std_panel2])
    """
    pooled = np.concatenate([p[~np.isnan(p)].ravel() for p in panels])
    return float(np.percentile(pooled, 98))


# ---------------------------------------------------------------------------
# Map helpers
# ---------------------------------------------------------------------------

def setup_map(ax: plt.Axes, left_labels: bool = False,
              bottom_labels: bool = False) -> None:
    """Configure a cartopy GeoAxes for the Bay of Bengal domain.

    Args:
        ax: Cartopy GeoAxes with PlateCarree projection.
        left_labels (bool): Draw latitude labels on the left.
        bottom_labels (bool): Draw longitude labels on the bottom.

    Returns:
        None

    Example:
        >>> setup_map(ax, left_labels=True)
    """
    ax.set_extent([_LON_RANGE[0], _LON_RANGE[1], _LAT_RANGE[0], _LAT_RANGE[1]],
                  crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#d3d3d3', edgecolor='#555555',
                   linewidth=0.4, zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      alpha=0.6, linestyle='--')
    gl.xlocator     = FixedLocator([80, 85, 90, 95])
    gl.ylocator     = FixedLocator([5, 10, 15, 20])
    gl.top_labels    = False
    gl.right_labels  = False
    gl.left_labels   = left_labels
    gl.bottom_labels = bottom_labels
    gl.xlabel_style  = {'size': 6}
    gl.ylabel_style  = {'size': 6}


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def build_figure(
    panels: dict,
    lons_model: np.ndarray,
    lats_model: np.ndarray,
    leads: list[int],
    output_path: str,
    save_pdf: bool = False,
) -> None:
    """Build and save the 4-row × 4-column ensemble comparison figure.

    Layout:
        Rows: SST mean, SST spread, SSS mean, SSS spread
        Cols: Winter +lead[0], Winter +lead[1], PostMonsoon +lead[0], PostMonsoon +lead[1]
        Col 4 (narrow): per-row colorbar

    Args:
        panels (dict): Nested dict panels[season][field][var][lead] → (224, 224).
            season: 'Winter' | 'PostMonsoon'
            field:  'mean' | 'std'
            var:    'thetao' | 'so'
            lead:   int
        lons_model (np.ndarray): 1-D longitude array of length 224 (model grid).
        lats_model (np.ndarray): 1-D latitude array of length 224 (model grid).
        leads (list[int]): Two lead days to plot.
        output_path (str): File stem without extension.
        save_pdf (bool): Also save a PDF.

    Returns:
        None

    Example:
        >>> build_figure(panels, lons, lats, [3, 5],
        ...              'results/figures/fig6_ensemble_season_comparison', save_pdf=True)
    """
    seasons = ['Winter', 'PostMonsoon']
    col_labels = [
        f'Winter  +{leads[0]} d', f'Winter  +{leads[1]} d',
        f'Post-monsoon  +{leads[0]} d', f'Post-monsoon  +{leads[1]} d',
    ]
    row_specs = [
        ('thetao', 'mean', _SST_CMAP, 'SST mean (°C)'),
        ('thetao', 'std',  _STD_CMAP, 'SST spread (°C)'),
        ('so',     'mean', _SSS_CMAP, 'SSS mean (PSU)'),
        ('so',     'std',  _STD_CMAP, 'SSS spread (PSU)'),
    ]

    # Compute shared color ranges per row
    norms = {}
    for var, field, _, _ in row_specs:
        all_panels = [panels[s][field][var][l] for s in seasons for l in leads]
        key = (var, field)
        if field == 'mean':
            vmin, vmax = pooled_range(all_panels)
            norms[key] = mcolors.Normalize(vmin=vmin, vmax=vmax)
        else:
            norms[key] = mcolors.Normalize(vmin=0, vmax=pooled_vmax(all_panels))

    lon2d, lat2d = np.meshgrid(lons_model, lats_model)

    fig = plt.figure(figsize=(15, 3.5 * 4))
    gs  = GridSpec(4, 5, width_ratios=[1, 1, 1, 1, 0.045],
                   hspace=0.12, wspace=0.05, figure=fig)

    for row, (var, field, cmap, row_label) in enumerate(row_specs):
        norm = norms[(var, field)]
        first_mesh = None
        for col, (season, lead) in enumerate(
                [(s, l) for s in seasons for l in leads]):
            ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
            setup_map(ax,
                      left_labels=(col == 0),
                      bottom_labels=(row == len(row_specs) - 1))

            data = panels[season][field][var][lead]
            mesh = ax.pcolormesh(lon2d, lat2d, data,
                                 cmap=cmap, norm=norm,
                                 transform=ccrs.PlateCarree(),
                                 zorder=1, shading='auto')
            if first_mesh is None:
                first_mesh = mesh

            if row == 0:
                ax.set_title(col_labels[col], fontsize=8, pad=4)
            if col == 0:
                ax.text(-0.20, 0.5, row_label,
                        transform=ax.transAxes,
                        fontsize=7, rotation=90, va='center', ha='right')

        cax = fig.add_subplot(gs[row, 4])
        fig.colorbar(first_mesh, cax=cax)
        cax.tick_params(labelsize=6)

    fig.suptitle(
        'AFNO RT Ensemble — Winter (14 Jan 2020) & Post-monsoon (14 Oct 2020)',
        fontsize=9, y=1.01)

    png_path = f'{output_path}.png'
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {png_path}')

    if save_pdf:
        pdf_path = f'{output_path}.pdf'
        fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f'Saved: {pdf_path}')

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
        description='Plot ensemble mean and spread for two seasons (fig6)')
    p.add_argument('--jan_dir',    required=True,
                   help='Directory with Winter ensemble_predictions.npy')
    p.add_argument('--oct_dir',    required=True,
                   help='Directory with Post-monsoon ensemble_predictions.npy')
    p.add_argument('--mean_dir',   required=True,
                   help='Directory with climatological mean .npy files')
    p.add_argument('--ocean_file', required=True,
                   help='Path to GLORYS ocean.nc (for lat/lon coordinates)')
    p.add_argument('--output',     default='results/figures/fig6_ensemble_season_comparison',
                   help='Output path stem (no extension)')
    p.add_argument('--leads',      type=int, nargs=2, default=[3, 5],
                   metavar=('LEAD1', 'LEAD2'),
                   help='Two lead days to plot (default: 3 5)')
    p.add_argument('--pdf',        action='store_true',
                   help='Also save a PDF')
    return p.parse_args()


def main() -> None:
    """Entry point: load predictions, process fields, build figure.

    Example:
        >>> # python src/visualization/plot_ensemble_comparison.py \\
        >>> #     --jan_dir results/AFNO_BoB_Surf_E14/ensemble_ic_Jan_14012020 \\
        >>> #     --oct_dir results/AFNO_BoB_Surf_E14/ensemble_ic_Oct_14102020 \\
        >>> #     --mean_dir data/1993_2020/mean \\
        >>> #     --ocean_file data/1993_2020/ocean.nc \\
        >>> #     --output results/figures/fig6_ensemble_season_comparison --pdf
    """
    args = parse_args()
    jan_dir  = Path(args.jan_dir)
    oct_dir  = Path(args.oct_dir)
    mean_dir = Path(args.mean_dir)
    leads    = args.leads

    # --- Load GLORYS coordinates and derive model-grid lats/lons ---
    print('Loading grid coordinates ...')
    lons_glorys, lats_glorys = load_glorys_coords(Path(args.ocean_file))
    # Model grid is bilinearly resampled from GLORYS; use the same domain bounds
    lons_model = np.linspace(lons_glorys[0], lons_glorys[-1], _MODEL_W)
    lats_model = np.linspace(lats_glorys[0], lats_glorys[-1], _MODEL_H)

    # --- Climatological means at 224×224 (for denormalization inside load_predictions) ---
    print('Loading climatological means ...')
    clim_mean = {
        'thetao': load_clim_mean_224(mean_dir, 'thetao'),
        'so':     load_clim_mean_224(mean_dir, 'so'),
    }

    # --- Load ensemble predictions for each season ---
    season_dirs = [
        ('Winter',      jan_dir),
        ('PostMonsoon', oct_dir),
    ]
    vars_to_plot = ['thetao', 'so']
    lead_indices = {l: l - 1 for l in leads}  # lead day 3 → index 2

    panels: dict = {}
    for season, ens_dir in season_dirs:
        print(f'\nLoading {season} from {ens_dir} ...')
        # Denormalization happens inside load_predictions; NaN propagates to land pixels.
        ens_mean, ens_std = load_predictions(ens_dir, clim_mean)
        panels[season] = {'mean': {}, 'std': {}}

        for var in vars_to_plot:
            ch = _VAR_IDX[var]
            panels[season]['mean'][var] = {}
            panels[season]['std'][var]  = {}
            for lead in leads:
                li = lead_indices[lead]

                mean_field = ens_mean[li, ch]   # physical units, NaN on land
                std_field  = ens_std[li, ch]    # physical units, NaN on land

                panels[season]['mean'][var][lead] = mean_field
                panels[season]['std'][var][lead]  = std_field

                print(f'  {season:12s} {var} +{lead}d  '
                      f'mean={np.nanmean(mean_field):.2f}  '
                      f'spread={np.nanmean(std_field):.4f}')

    # --- Build figure ---
    print('\nBuilding figure ...')
    build_figure(panels, lons_model, lats_model, leads,
                 args.output, save_pdf=args.pdf)


if __name__ == '__main__':
    main()

"""Seasonal SST comparison: OSTIA L4 analysis vs AFNO forecast at lead +1 day.

Produces a 4-row × 3-column figure where each row shows one Bay-of-Bengal
season (Winter, Pre-monsoon, Monsoon, Post-monsoon). The three columns show:
  Col 1 — OSTIA L4 analysed SST (independent operational analysis, °C)
  Col 2 — AFNO RT prediction at lead day +1 (°C)
  Col 3 — Signed error: AFNO − OSTIA (°C, diverging colourmap)

OSTIA provides validation independent of the GLORYS reanalysis used in
training, demonstrating that model skill is not an artefact of reanalysis
consistency.

Inputs:
    --config_file (str): YAML config filename in config/ directory.
    --model_path (str): Path to .pth model weights.
    --ostia_file (str): Path to ostia_2020.nc (default: data/1993_2020/ostia_2020.nc).
    --dates (str): Comma-separated IC dates dd-mm-yyyy, one per row.
    --season_labels (str): Comma-separated row labels matching dates.
    --device (str): Compute device override.
    --output (str): Output base path (without extension).
    --sst_range (str): Optional 'vmin,vmax' for SST colourbars (°C).
    --err_range (float): Half-range for signed-error colourbar (default 2.0 °C).
    --dpi (int): Output raster DPI (default 300).
    --pdf (flag): Also write a vector PDF.

Outputs:
    {output}.png  — 300 dpi raster figure.
    {output}.pdf  — vector PDF (only with --pdf flag).

Example:
    conda activate BoB_Surf_2
    python src/visualization/plot_ostia_comparison.py \\
        --config_file afno_bob_surf_e14.yaml \\
        --model_path results/models/AFNO_BoB_Surf_E14.pth \\
        --dates "15-01-2020,15-04-2020,15-07-2020,15-10-2020" \\
        --season_labels "Winter,Pre-monsoon,Monsoon,Post-monsoon" \\
        --output results/figures/fig8_ostia_comparison \\
        --device cuda:0
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import FixedLocator

sys.path.append(str(Path(__file__).parent.parent))

from configmypy import ConfigPipeline, YamlConfig
from visualization.plot_seasonal_forecast import (
    load_model, load_norm_stats, build_input, model_forward,
    postprocess_var, get_coords, model_display_name,
)
from inference.utils import date_to_day_index
from data_pipeline.preprocessing.transformers.normalize_transform import PreprocessTransform

try:
    import cmocean
    _SST_CMAP = cmocean.cm.thermal
except ImportError:
    _SST_CMAP = 'RdYlBu_r'

_ERR_CMAP  = 'RdBu_r'
_REF_DATE  = '01-01-1993'
_PROJ      = ccrs.PlateCarree()
_LON_TICKS = [80, 85, 90, 95, 100]
_LAT_TICKS = [5, 10, 15, 20, 25]
OSTIA_KELVIN_OFFSET = 273.15


def _parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.

    Example:
        >>> args = _parse_args()
    """
    p = argparse.ArgumentParser(
        description='4-row × 3-col OSTIA vs AFNO SST comparison figure'
    )
    p.add_argument('--config_file', default='afno_bob_surf_e14.yaml')
    p.add_argument('--model_path',  default=None)
    p.add_argument('--ostia_file',  default='data/1993_2020/ostia_2020.nc')
    p.add_argument('--dates',
                   default='15-01-2020,15-04-2020,15-07-2020,15-10-2020')
    p.add_argument('--season_labels',
                   default='Winter,Pre-monsoon,Monsoon,Post-monsoon')
    p.add_argument('--device',    default=None)
    p.add_argument('--output',    default=None)
    p.add_argument('--sst_range', default=None, help='"vmin,vmax" in °C')
    p.add_argument('--err_range', type=float, default=2.0,
                   help='Half-range for signed error colourbar in °C (default 2.0)')
    p.add_argument('--dpi',       type=int, default=300)
    p.add_argument('--pdf',       action='store_true')
    return p.parse_args()


def _map_panel(ax, data, lon, lat, cmap, vmin, vmax, title='', label=''):
    """Draw a filled-contour map panel with coastlines and lat/lon ticks.

    Args:
        ax (GeoAxes): Cartopy axes.
        data (np.ndarray): 2-D field to plot.
        lon (np.ndarray): 1-D longitude array.
        lat (np.ndarray): 1-D latitude array.
        cmap: Matplotlib colormap.
        vmin (float): Colormap minimum.
        vmax (float): Colormap maximum.
        title (str): Panel title (top-left annotation).
        label (str): Row label placed on the left of the first column.

    Returns:
        matplotlib.contour.QuadContourSet: The pcolormesh/contourf handle
            (used for colourbars).

    Example:
        >>> h = _map_panel(ax, sst, lon, lat, _SST_CMAP, 24, 32)
    """
    h = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax,
                      transform=_PROJ, rasterized=True)
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=3)
    ax.coastlines(resolution='50m', linewidth=0.5, zorder=4)
    ax.set_extent([77, 99, 4, 23], crs=_PROJ)
    ax.set_xticks(_LON_TICKS, crs=_PROJ)
    ax.set_yticks(_LAT_TICKS, crs=_PROJ)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f'{int(v)}°E'))
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f'{int(v)}°N'))
    ax.tick_params(labelsize=6)
    if title:
        ax.set_title(title, fontsize=7, pad=2)
    return h


def main():
    """Build seasonal OSTIA-vs-AFNO SST comparison figure and save to disk.

    Args:
        None: All settings read from sys.argv (see module docstring).

    Returns:
        None: Writes PNG (and optionally PDF) figure files.

    Example:
        >>> # python src/visualization/plot_ostia_comparison.py \\
        >>> #     --config_file afno_bob_surf_e14.yaml \\
        >>> #     --dates 15-01-2020,15-04-2020,15-07-2020,15-10-2020
    """
    args         = _parse_args()
    dates        = [d.strip() for d in args.dates.split(',')]
    season_labels = [s.strip() for s in args.season_labels.split(',')]
    n_rows       = len(dates)

    # --- Config and model ---
    pipe = ConfigPipeline([
        YamlConfig(args.config_file, config_name='default', config_folder='config/'),
    ])
    config = pipe.read_conf()
    device = torch.device(args.device or str(config.device))
    model_path = args.model_path or f'results/models/{config.name}.pth'
    print(f'Model  : {model_path}')
    print(f'Device : {device}')
    model = load_model(config, model_path, device)

    output = args.output or f'results/figures/ostia_comparison_{config.name}'

    # --- Data ---
    mean, variance = load_norm_stats(config)
    transform      = PreprocessTransform(config)
    data_dir       = Path(config.data.data_dir)
    ocean_ds       = xr.open_dataset(data_dir / f'{config.data.file_prefix}.nc')
    atm_ds         = xr.open_dataset(data_dir / f'{config.data.file_prefix_atm}.nc')
    lons, lats     = get_coords(ocean_ds)

    # --- OSTIA: pre-interpolate to GLORYS grid ---
    print(f'OSTIA  : {args.ostia_file}')
    ostia_ds  = xr.open_dataset(args.ostia_file)
    ostia_sst = (ostia_ds['analysed_sst'] - OSTIA_KELVIN_OFFSET).interp(
        latitude=lats, longitude=lons, method='linear'
    )
    ostia_dates = [str(t)[:10] for t in ostia_sst.time.values]
    ostia_idx   = {d: i for i, d in enumerate(ostia_dates)}
    ostia_ds.close()

    # --- Colourbar limits ---
    sst_vmin, sst_vmax = (
        [float(v) for v in args.sst_range.split(',')] if args.sst_range
        else (24.0, 32.0)
    )
    err_abs = args.err_range

    # --- Run forecasts and collect panels ---
    rows = []
    for ic_str in dates:
        day_idx   = date_to_day_index(ic_str, _REF_DATE)
        pred_dt   = datetime.strptime(ic_str, '%d-%m-%Y') + timedelta(days=1)
        pred_key  = pred_dt.strftime('%Y-%m-%d')
        print(f'  IC {ic_str} → valid {pred_key}')

        x          = build_input(config, day_idx, ocean_ds, atm_ds,
                                 mean, variance, transform)
        output_arr = model_forward(model, x, device)
        sst_pred   = postprocess_var(output_arr, config, 'thetao', mean)  # (229,265)

        if pred_key in ostia_idx:
            sst_ostia = ostia_sst.values[ostia_idx[pred_key]]   # (229,265) °C
        else:
            print(f'    ⚠ no OSTIA for {pred_key}, filling NaN')
            sst_ostia = np.full_like(sst_pred, np.nan)

        rows.append(dict(
            label=season_labels[len(rows)],
            pred_dt=pred_dt.strftime('%d %b %Y'),
            sst_pred=sst_pred,
            sst_ostia=sst_ostia,
            err=sst_pred - sst_ostia,
        ))

    ocean_ds.close()
    atm_ds.close()

    # --- Layout ---
    fig_h = 2.2 * n_rows + 0.5
    fig = plt.figure(figsize=(10, fig_h))
    gs  = GridSpec(n_rows, 3, figure=fig,
                   hspace=0.08, wspace=0.05,
                   left=0.06, right=0.92, top=0.94, bottom=0.06)

    col_titles = ['OSTIA SST (°C)', 'AFNO RT (°C)', 'AFNO − OSTIA (°C)']

    for ri, row in enumerate(rows):
        for ci in range(3):
            ax = fig.add_subplot(gs[ri, ci], projection=_PROJ)
            if ri == 0:
                ax.set_title(col_titles[ci], fontsize=8, fontweight='bold', pad=3)

            if ci == 0:
                h_sst = _map_panel(ax, row['sst_ostia'], lons, lats,
                                   _SST_CMAP, sst_vmin, sst_vmax)
                ax.text(-0.18, 0.5, row['label'], transform=ax.transAxes,
                        fontsize=8, fontweight='bold', va='center', ha='right',
                        rotation=90)
            elif ci == 1:
                h_sst = _map_panel(ax, row['sst_pred'], lons, lats,
                                   _SST_CMAP, sst_vmin, sst_vmax,
                                   title=f'AFNO RT +1 day Forecast ({row["pred_dt"]})')
            else:
                h_err = _map_panel(ax, row['err'], lons, lats,
                                   _ERR_CMAP, -err_abs, err_abs)

            # Suppress y-axis ticks except on leftmost column
            if ci > 0:
                ax.set_yticklabels([])
            # Suppress x-axis ticks except on bottom row
            if ri < n_rows - 1:
                ax.set_xticklabels([])

    # Colourbars
    cb_ax_sst = fig.add_axes([0.06, 0.02, 0.55, 0.012])
    plt.colorbar(h_sst, cax=cb_ax_sst, orientation='horizontal',
                 label='SST (°C)', extend='both')
    cb_ax_err = fig.add_axes([0.65, 0.02, 0.27, 0.012])
    plt.colorbar(h_err, cax=cb_ax_err, orientation='horizontal',
                 label='Error (°C)', extend='both')

    fig.suptitle(f'OSTIA vs {model_display_name(config.name)} — SST at Lead +1 Day',
                 fontsize=9, fontweight='bold', y=0.97)

    for ext in (['png', 'pdf'] if args.pdf else ['png']):
        p = f'{output}.{ext}'
        fig.savefig(p, dpi=args.dpi if ext == 'png' else None, bbox_inches='tight')
        print(f'Saved: {p}')
    plt.close(fig)


if __name__ == '__main__':
    main()

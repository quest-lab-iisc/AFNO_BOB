"""Forecast evolution figure: AFNO RT prediction panels at lead days 3, 5, 7, 9.

Produces a 4-row × 4-column figure where:
  - Rows   : SST (°C), SSS (psu), SSH (m), current speed (m s⁻¹)
  - Columns: lead day +3, +5, +7, +9 from a single initialisation date

Each panel shows the AFNO RT prediction as a pcolormesh with GLORYS
reanalysis truth overlaid as white contour lines.  An RMSE annotation
appears in the bottom-left corner of each panel.  Colorbars are shared
across each row and placed at the right of the figure.

Inputs:
    config_file (str): YAML config filename in config/ directory.
    model_path (str): Path to .pth weights; defaults to results/models/{name}.pth.
    init_date (str): Forecast initialisation date in dd-mm-yyyy format.
    lead_days (str): Comma-separated lead days to display (default: 3,5,7,9).
    device (str): Compute device override, e.g. cuda:0 or cpu.
    output (str): Output path without file extension.
    sst_range (str): Optional "vmin,vmax" for SST colourbar (°C).
    sss_range (str): Optional "vmin,vmax" for SSS colourbar (psu).
    ssh_range (str): Optional "vmin,vmax" for SSH colourbar (m).
    speed_range (str): Optional "vmin,vmax" for speed colourbar (m s⁻¹).

Outputs:
    <output>.pdf (file): Vector figure for paper submission.
    <output>.png (file): Raster figure at 300 dpi for review.

Example:
    python src/visualization/plot_forecast_evolution.py \\
        --config_file afno_bob_surf_e11p1.yaml \\
        --init_date 09-07-2020 \\
        --device cuda:0
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.append(str(Path(__file__).parent.parent))

from plot_seasonal_forecast import (
    load_config, load_model, load_norm_stats,
    _detect_arch, _preprocess_atm, _preprocess_ocean,
    model_forward, postprocess_var, get_coords,
    setup_map, draw_panel, _error_metrics,
    _LON_RANGE, _LAT_RANGE, _ERR_CMAP, _REF_DATE,
    model_display_name,
)
from inference.utils import date_to_day_index
from data_pipeline.preprocessing.transformers.normalize_transform import PreprocessTransform

try:
    import cmocean
    _SST_CMAP   = cmocean.cm.thermal
    _SSS_CMAP   = cmocean.cm.haline
    _SSH_CMAP   = cmocean.cm.matter
    _SPD_CMAP   = cmocean.cm.speed
except ImportError:
    _SST_CMAP   = 'RdYlBu_r'
    _SSS_CMAP   = 'viridis'
    _SSH_CMAP   = 'PuBu'
    _SPD_CMAP   = 'YlOrBr'

_DEFAULT_LEADS = [3, 5, 7, 9]
_N_CONTOURS    = 7

# Dual-contour / error-bg style colours
_AFNO_COLOR  = '#FF6B35'   # orange-red  — AFNO RT contours
_TRUTH_COLOR = '#1B98E0'   # sky blue    — GLORYS contours


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.

    Example:
        >>> args = parse_args()
    """
    p = argparse.ArgumentParser(
        description='Forecast evolution panel: AFNO RT at lead days 3/5/7/9'
    )
    p.add_argument('--config_file', default='afno_bob_surf_e11p1.yaml',
                   help='YAML config filename inside config/')
    p.add_argument('--model_path', default=None,
                   help='Explicit .pth path; default: results/models/{name}.pth')
    p.add_argument('--init_date', default='09-07-2020',
                   help='Forecast initialisation date dd-mm-yyyy (default: 09-07-2020)')
    p.add_argument('--lead_days', default='3,5,7,9',
                   help='Comma-separated lead days to display (default: 3,5,7,9)')
    p.add_argument('--device', default=None,
                   help='Compute device, e.g. cuda:0 or cpu')
    p.add_argument('--output', default=None,
                   help='Output path without extension (default: results/figures/forecast_evolution)')
    p.add_argument('--sst_range',   default=None, help='SST vmin,vmax in °C')
    p.add_argument('--sss_range',   default=None, help='SSS vmin,vmax in psu')
    p.add_argument('--ssh_range',   default=None, help='SSH vmin,vmax in m')
    p.add_argument('--speed_range', default=None, help='Speed vmin,vmax in m/s')
    p.add_argument('--pdf', action='store_true',
                   help='Also save a PDF alongside the PNG (default: PNG only)')
    p.add_argument('--style', default='default',
                   choices=['default', 'dual_contour', 'error_bg'],
                   help=(
                       'default     — AFNO RT pcolormesh + GLORYS white contours; '
                       'dual_contour — orange AFNO RT contours + blue GLORYS contours; '
                       'error_bg    — pred-truth error pcolormesh + both contours'
                   ))
    p.add_argument('--extra_data_dir', default=None,
                   help='Directory containing 2021-2025 ocean/atm NetCDF files; '
                        'used when init_date year >= 2021')
    p.add_argument('--extra_ref_date', default='01-01-2021',
                   help='Index-0 date for extra NetCDF (dd-mm-yyyy); default: 01-01-2021')
    return p.parse_args()


def _parse_range(s):
    """Convert a 'vmin,vmax' string to a (float, float) tuple, or None.

    Args:
        s (str | None): String like '28.0,31.0' or None.

    Returns:
        tuple[float, float] | None: Parsed range, or None if input is None.

    Example:
        >>> _parse_range('28.0,31.0')
        (28.0, 31.0)
    """
    if s is None:
        return None
    lo, hi = s.split(',')
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def run_rollout(config, model, init_date, lead_days, ocean_ds, atm_ds,
                mean, variance, device, ref_date=_REF_DATE):
    """Autoregressive rollout from init_date, collecting results at each lead day.

    Rolls the ocean-only model forward step by step, feeding its own 5-channel
    ocean output back as the next-step ocean input.  At each requested lead day
    the AFNO prediction and corresponding GLORYS ground truth are collected and
    postprocessed to physical units.

    Args:
        config: configmypy configuration object.
        model (torch.nn.Module): Loaded AFNO model in eval mode.
        init_date (str): Initialisation date in dd-mm-yyyy format.
        lead_days (list[int]): Lead days at which to save output (e.g. [3,5,7,9]).
        ocean_ds (xr.Dataset): Ocean variable dataset (GLORYS).
        atm_ds (xr.Dataset): Atmospheric variable dataset.
        mean (dict): Per-variable normalisation means.
        variance (dict): Per-variable normalisation standard deviations.
        device (torch.device): Compute device.
        ref_date (str): Index-0 date of the datasets (dd-mm-yyyy). Use
            '01-01-2021' for the extended 2021-2025 NetCDF files.

    Returns:
        dict[int, dict]: Maps each lead day to a dict with keys:
            sst_pred, sss_pred, ssh_pred, speed_pred,
            sst_truth, sss_truth, ssh_truth, speed_truth (all np.ndarray H×W).

    Example:
        >>> results = run_rollout(config, model, '09-07-2020', [3,5,7,9], ...)
        >>> results[3]['sst_pred'].shape
        (229, 265)
    """
    transform   = PreprocessTransform(config)
    init_idx    = date_to_day_index(init_date, ref_date)
    max_lead    = max(lead_days)
    out_vars    = list(config.data.out_variable)

    # Initialise ocean state in preprocessed (224×224) space
    ocean_state = {}
    for var in config.data.variable:
        raw = ocean_ds[var][init_idx:init_idx + 1].values
        ocean_state[var] = _preprocess_ocean(raw, var, mean, transform)  # (1,224,224)

    results = {}

    for step in range(1, max_lead + 1):
        # Atmospheric forcing at t + step (ocean-only model uses future atm)
        atm_channels = []
        for var in config.data.atm_variable:
            raw = atm_ds[var][init_idx + step:init_idx + step + 1].values
            atm_channels.append(_preprocess_atm(raw, var, mean, variance, transform))

        ocean_channels = [ocean_state[v] for v in config.data.variable]
        x = torch.cat(atm_channels + ocean_channels, dim=0)   # (11, 224, 224)

        output_arr = model_forward(model, x, device)           # (5, 224, 224)

        # Update ocean state for next step
        for i, var in enumerate(out_vars):
            ocean_state[var] = torch.tensor(output_arr[i]).unsqueeze(0)

        if step not in lead_days:
            continue

        target_idx = init_idx + step   # GLORYS truth index

        # Postprocess AFNO predictions
        sst_pred   = postprocess_var(output_arr, config, 'thetao', mean)
        sss_pred   = postprocess_var(output_arr, config, 'so',     mean)
        ssh_pred   = postprocess_var(output_arr, config, 'zos',    mean)
        uo_pred    = postprocess_var(output_arr, config, 'uo',     mean)
        vo_pred    = postprocess_var(output_arr, config, 'vo',     mean)
        speed_pred = np.sqrt(uo_pred ** 2 + vo_pred ** 2)

        # GLORYS ground truth at target date
        sst_truth   = ocean_ds['thetao'][target_idx].values.squeeze()
        sss_truth   = ocean_ds['so'][target_idx].values.squeeze()
        ssh_truth   = ocean_ds['zos'][target_idx].values.squeeze()
        uo_truth    = ocean_ds['uo'][target_idx].values.squeeze()
        vo_truth    = ocean_ds['vo'][target_idx].values.squeeze()
        speed_truth = np.sqrt(np.where(np.isnan(uo_truth), np.nan, uo_truth) ** 2 +
                              np.where(np.isnan(vo_truth), np.nan, vo_truth) ** 2)

        # Apply the GLORYS uo truth NaN pattern as land mask — same approach
        # as plot_seasonal_uvssh.py: model outputs have no NaN on land pixels.
        land_mask  = np.isnan(uo_truth)
        ssh_pred   = np.where(land_mask, np.nan, ssh_pred)
        uo_pred    = np.where(land_mask, np.nan, uo_pred)
        vo_pred    = np.where(land_mask, np.nan, vo_pred)
        speed_pred = np.where(land_mask, np.nan, speed_pred)

        # Gaussian-smooth the northern boundary rows only for models trained
        # with north masking (north_mask_rows > 0) to suppress the flat
        # artifact from preprocessing.  When n_north == 0 (e.g. E14), skip
        # entirely — ssh_pred[-0:] would overwrite the whole array with NaN.
        n_north = round(config.data.north_mask_rows * ssh_pred.shape[0] / 224)
        if n_north > 0:
            boundary_val = ssh_pred[-(n_north + 1), :]
            ssh_pred[-n_north:, :] = boundary_val[np.newaxis, :]
            tmp = ssh_pred.copy()
            tmp[np.isnan(tmp)] = np.nanmedian(tmp)
            smoothed = gaussian_filter(tmp, sigma=8)
            ssh_pred[-(n_north + 5):, :] = smoothed[-(n_north + 5):, :]
            ssh_pred = np.where(land_mask, np.nan, ssh_pred)  # re-apply land mask

        results[step] = {
            'sst_pred':    sst_pred,
            'sss_pred':    sss_pred,
            'ssh_pred':    ssh_pred,
            'speed_pred':  speed_pred,
            'sst_truth':   sst_truth,
            'sss_truth':   sss_truth,
            'ssh_truth':   ssh_truth,
            'speed_truth': speed_truth,
        }

        print(f'  Lead +{step:2d}d done')

    return results


# ---------------------------------------------------------------------------
# Colour ranges
# ---------------------------------------------------------------------------

def _pool_valid(*arrays):
    """Concatenate non-NaN values from multiple arrays.

    Args:
        *arrays (np.ndarray): Arbitrary number of 2-D arrays.

    Returns:
        np.ndarray: 1-D array of all finite values.

    Example:
        >>> vals = _pool_valid(a, b)
    """
    return np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])


def compute_ranges(results, lead_days,
                   sst_range=None, sss_range=None, ssh_range=None, speed_range=None):
    """Compute per-variable colourbar limits pooled across all lead days.

    Args:
        results (dict): Output of run_rollout.
        lead_days (list[int]): Lead days included in results.
        sst_range (tuple | None): (vmin, vmax) override for SST.
        sss_range (tuple | None): (vmin, vmax) override for SSS.
        ssh_range (tuple | None): (vmin, vmax) override for SSH.
        speed_range (tuple | None): (vmin, vmax) override for speed.

    Returns:
        dict: Keys sst, sss, ssh, speed — each a (vmin, vmax) tuple.

    Example:
        >>> rngs = compute_ranges(results, [3,5,7,9])
        >>> rngs['sst']
        (27.1, 30.8)
    """
    def _auto(key_pred, key_truth, override):
        if override:
            return override
        vals = _pool_valid(*[results[ld][key_pred]   for ld in lead_days],
                           *[results[ld][key_truth]  for ld in lead_days])
        return float(np.percentile(vals, 2)), float(np.percentile(vals, 98))

    return {
        'sst':   _auto('sst_pred',   'sst_truth',   sst_range),
        'sss':   _auto('sss_pred',   'sss_truth',   sss_range),
        'ssh':   _auto('ssh_pred',   'ssh_truth',   ssh_range),
        'speed': _auto('speed_pred', 'speed_truth', speed_range),
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def build_figure(results, lead_days, ranges, lons, lats, init_date, config, output_path,
                 save_pdf=False):
    """Assemble and save the 4×4 forecast evolution figure.

    Each cell shows AFNO RT as a pcolormesh and GLORYS truth as white contour
    lines.  RMSE is annotated in the lower-left corner.  One colourbar is placed
    at the right of each row.

    Args:
        results (dict): Output of run_rollout; maps lead_day -> field dict.
        lead_days (list[int]): Ordered list of lead days to plot.
        ranges (dict): Per-variable (vmin, vmax) from compute_ranges.
        lons (np.ndarray): 1-D longitude array.
        lats (np.ndarray): 1-D latitude array.
        init_date (str): Initialisation date string for the figure title.
        config: configmypy configuration object (for model name).
        output_path (str): Output path without file extension.

    Returns:
        str: Path to the saved PDF file.

    Example:
        >>> build_figure(results, [3,5,7,9], rngs, lons, lats, '09-07-2020', config, 'out')
    """
    n_rows = 4
    n_cols = len(lead_days)

    # Variable metadata: (pred_key, truth_key, label, unit, cmap)
    var_meta = [
        ('sst_pred',   'sst_truth',   'SST',   '°C',        _SST_CMAP, 'sst'),
        ('sss_pred',   'sss_truth',   'SSS',   'psu',       _SSS_CMAP, 'sss'),
        ('ssh_pred',   'ssh_truth',   'SSH',   'm',         _SSH_CMAP, 'ssh'),
        ('speed_pred', 'speed_truth', 'Speed', 'm s$^{-1}$', _SPD_CMAP, 'speed'),
    ]

    # Colourbar extra column for each row
    col_widths = [1.0] * n_cols + [0.05]   # last entry = colourbar
    fig_w = 3.2 * n_cols + 0.6
    fig_h = 3.0 * n_rows + 0.5

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(
        n_rows, n_cols + 1,
        width_ratios=col_widths,
        hspace=0.08, wspace=0.04,
        top=0.93, bottom=0.04, left=0.07, right=0.96,
    )

    lon2d, lat2d = np.meshgrid(lons, lats)

    for row, (pred_key, truth_key, vlabel, unit, cmap, rng_key) in enumerate(var_meta):
        vmin, vmax = ranges[rng_key]
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # Contour levels for GLORYS overlay
        truth_vals = _pool_valid(*[results[ld][truth_key] for ld in lead_days])
        c_levels = np.linspace(
            np.percentile(truth_vals, 5),
            np.percentile(truth_vals, 95),
            _N_CONTOURS,
        )

        col_axes = []
        for col, ld in enumerate(lead_days):
            is_left   = col == 0
            is_bottom = row == n_rows - 1

            ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
            setup_map(ax, left_labels=is_left, bottom_labels=is_bottom)

            pred  = results[ld][pred_key]
            truth = results[ld][truth_key]

            # AFNO prediction as filled field
            mesh = ax.pcolormesh(lon2d, lat2d, pred, cmap=cmap, norm=norm,
                                 transform=ccrs.PlateCarree(), zorder=1, shading='auto')

            # GLORYS truth as white contour lines
            valid_truth = np.where(np.isfinite(truth), truth, np.nanmedian(truth))
            ax.contour(lon2d, lat2d, valid_truth, levels=c_levels,
                       colors='white', linewidths=0.7, alpha=0.85,
                       transform=ccrs.PlateCarree(), zorder=2)

            # RMSE annotation
            err   = pred - truth
            valid = err[np.isfinite(err)]
            rmse  = float(np.sqrt(np.mean(valid ** 2)))
            ax.text(0.03, 0.04,
                    f'RMSE = {rmse:.3f} {unit}',
                    transform=ax.transAxes,
                    ha='left', va='bottom', fontsize=6.5, fontfamily='monospace',
                    color='white',
                    bbox=dict(facecolor='black', alpha=0.45, edgecolor='none', pad=1.5),
                    zorder=5)

            # Column header above top row
            if row == 0:
                ax.annotate(f'Day +{ld}',
                            xy=(0.5, 1.03), xycoords='axes fraction',
                            ha='center', va='bottom', fontsize=9, fontweight='bold',
                            annotation_clip=False)

            col_axes.append((ax, mesh))

        # Row label on the left of first column
        col_axes[0][0].text(
            -0.18, 0.5, f'{vlabel}\n({unit})',
            transform=col_axes[0][0].transAxes,
            ha='center', va='center', fontsize=9, fontweight='bold',
            rotation=90,
        )

        # Colourbar at far right
        cax = fig.add_subplot(gs[row, n_cols])
        sm  = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb  = fig.colorbar(sm, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=7)

    # Overall title
    dt   = datetime.strptime(init_date, '%d-%m-%Y')
    name = model_display_name(getattr(config, 'name', 'AFNO RT'))
    fig.suptitle(
        f'{name}  —  Forecast evolution  |  IC: {dt.strftime("%-d %b %Y")}  '
        f'|  AFNO RT (shading) vs GLORYS (contours)',
        fontsize=10, fontweight='bold', y=0.97,
    )

    exts = ['png'] + (['pdf'] if save_pdf else [])
    for ext in exts:
        path = f'{output_path}.{ext}'
        fig.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f'Saved: {path}')

    plt.close(fig)
    return f'{output_path}.png'


# ---------------------------------------------------------------------------
# Style: dual_contour
# ---------------------------------------------------------------------------

def build_figure_dual_contour(results, lead_days, ranges, lons, lats, init_date,
                               config, output_path, save_pdf=False):
    """4×4 figure with AFNO RT (orange) and GLORYS (blue) contour lines — no background fill.

    No pcolormesh is drawn; instead contour lines from both datasets are overlaid
    on a clean cartopy map, making it easy to see where the prediction and truth
    iso-contours align or diverge.

    Args:
        results (dict): Output of run_rollout; maps lead_day -> field dict.
        lead_days (list[int]): Ordered list of lead days to plot.
        ranges (dict): Per-variable (vmin, vmax) from compute_ranges (used only for
            deriving shared contour levels).
        lons (np.ndarray): 1-D longitude array.
        lats (np.ndarray): 1-D latitude array.
        init_date (str): Initialisation date string for the figure title.
        config: configmypy configuration object (for model name).
        output_path (str): Output path without file extension.

    Returns:
        str: Path to the saved PDF file.

    Example:
        >>> build_figure_dual_contour(results, [3,5,7,9], rngs, lons, lats,
        ...                           '09-07-2020', config, 'out_dual')
    """
    n_rows = 4
    n_cols = len(lead_days)

    var_meta = [
        ('sst_pred',   'sst_truth',   'SST',   '°C',         'sst'),
        ('sss_pred',   'sss_truth',   'SSS',   'psu',        'sss'),
        ('ssh_pred',   'ssh_truth',   'SSH',   'm',          'ssh'),
        ('speed_pred', 'speed_truth', 'Speed', 'm s$^{-1}$', 'speed'),
    ]

    fig_w = 3.2 * n_cols + 0.6
    fig_h = 3.0 * n_rows + 0.8

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(
        n_rows, n_cols,
        hspace=0.08, wspace=0.04,
        top=0.90, bottom=0.06, left=0.08, right=0.97,
    )

    lon2d, lat2d = np.meshgrid(lons, lats)

    for row, (pred_key, truth_key, vlabel, unit, rng_key) in enumerate(var_meta):
        vmin, vmax = ranges[rng_key]
        c_levels = np.linspace(vmin, vmax, _N_CONTOURS)

        for col, ld in enumerate(lead_days):
            is_left   = col == 0
            is_bottom = row == n_rows - 1

            ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
            setup_map(ax, left_labels=is_left, bottom_labels=is_bottom)

            pred  = results[ld][pred_key]
            truth = results[ld][truth_key]

            valid_pred  = np.where(np.isfinite(pred),  pred,  np.nanmedian(pred))
            valid_truth = np.where(np.isfinite(truth), truth, np.nanmedian(truth))

            ax.contour(lon2d, lat2d, valid_truth, levels=c_levels,
                       colors=_TRUTH_COLOR, linewidths=0.9, linestyles='dashed',
                       transform=ccrs.PlateCarree(), zorder=2)
            ax.contour(lon2d, lat2d, valid_pred, levels=c_levels,
                       colors=_AFNO_COLOR, linewidths=0.9, linestyles='solid',
                       transform=ccrs.PlateCarree(), zorder=3)

            err   = pred - truth
            valid = err[np.isfinite(err)]
            rmse  = float(np.sqrt(np.mean(valid ** 2)))
            ax.text(0.03, 0.04,
                    f'RMSE = {rmse:.3f} {unit}',
                    transform=ax.transAxes,
                    ha='left', va='bottom', fontsize=6.5, fontfamily='monospace',
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.5),
                    zorder=5)

            if row == 0:
                ax.annotate(f'Day +{ld}',
                            xy=(0.5, 1.03), xycoords='axes fraction',
                            ha='center', va='bottom', fontsize=9, fontweight='bold',
                            annotation_clip=False)

        # Row label on the left
        ax0 = fig.axes[row * n_cols]
        ax0.text(
            -0.18, 0.5, f'{vlabel}\n({unit})',
            transform=ax0.transAxes,
            ha='center', va='center', fontsize=9, fontweight='bold', rotation=90,
        )

    # Legend
    leg_handles = [
        mlines.Line2D([], [], color=_AFNO_COLOR, linewidth=1.2, linestyle='solid',
                      label='AFNO RT'),
        mlines.Line2D([], [], color=_TRUTH_COLOR, linewidth=1.2, linestyle='dashed',
                      label='GLORYS'),
    ]
    fig.legend(handles=leg_handles, loc='lower center', ncol=2,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.01))

    dt   = datetime.strptime(init_date, '%d-%m-%Y')
    name = model_display_name(getattr(config, 'name', 'AFNO RT'))
    fig.suptitle(
        f'{name}  —  Forecast evolution  |  IC: {dt.strftime("%-d %b %Y")}  '
        f'|  AFNO RT (orange) vs GLORYS (blue, dashed)',
        fontsize=10, fontweight='bold', y=0.96,
    )

    exts = ['png'] + (['pdf'] if save_pdf else [])
    for ext in exts:
        path = f'{output_path}.{ext}'
        fig.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f'Saved: {path}')

    plt.close(fig)
    return f'{output_path}.png'


# ---------------------------------------------------------------------------
# Style: error_bg
# ---------------------------------------------------------------------------

def build_figure_error_bg(results, lead_days, ranges, lons, lats, init_date,
                           config, output_path, save_pdf=False):
    """4×4 figure with pred-truth error as pcolormesh and dual contours overlaid.

    The error (AFNO RT − GLORYS) is shown as a diverging red-blue background.
    AFNO RT iso-contours (orange) and GLORYS iso-contours (blue dashed) are
    overlaid to indicate spatial structure.  Panels where prediction and truth
    contours agree closely will show small errors.

    Args:
        results (dict): Output of run_rollout; maps lead_day -> field dict.
        lead_days (list[int]): Ordered list of lead days to plot.
        ranges (dict): Per-variable (vmin, vmax) from compute_ranges (used for
            deriving shared contour levels).
        lons (np.ndarray): 1-D longitude array.
        lats (np.ndarray): 1-D latitude array.
        init_date (str): Initialisation date string for the figure title.
        config: configmypy configuration object (for model name).
        output_path (str): Output path without file extension.

    Returns:
        str: Path to the saved PDF file.

    Example:
        >>> build_figure_error_bg(results, [3,5,7,9], rngs, lons, lats,
        ...                       '09-07-2020', config, 'out_err')
    """
    n_rows = 4
    n_cols = len(lead_days)

    var_meta = [
        ('sst_pred',   'sst_truth',   'SST',   '°C',         'sst'),
        ('sss_pred',   'sss_truth',   'SSS',   'psu',        'sss'),
        ('ssh_pred',   'ssh_truth',   'SSH',   'm',          'ssh'),
        ('speed_pred', 'speed_truth', 'Speed', 'm s$^{-1}$', 'speed'),
    ]

    col_widths = [1.0] * n_cols + [0.05]
    fig_w = 3.2 * n_cols + 0.8
    fig_h = 3.0 * n_rows + 0.8

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(
        n_rows, n_cols + 1,
        width_ratios=col_widths,
        hspace=0.08, wspace=0.04,
        top=0.90, bottom=0.06, left=0.08, right=0.96,
    )

    lon2d, lat2d = np.meshgrid(lons, lats)

    for row, (pred_key, truth_key, vlabel, unit, rng_key) in enumerate(var_meta):
        vmin, vmax = ranges[rng_key]
        c_levels = np.linspace(vmin, vmax, _N_CONTOURS)

        # Symmetric error colour limits pooled across all lead days
        all_err = np.concatenate([
            (results[ld][pred_key] - results[ld][truth_key])[
                np.isfinite(results[ld][pred_key] - results[ld][truth_key])
            ].ravel()
            for ld in lead_days
        ])
        elim = float(np.percentile(np.abs(all_err), 98))
        err_norm = mcolors.TwoSlopeNorm(vmin=-elim, vcenter=0.0, vmax=elim)

        col_axes = []
        for col, ld in enumerate(lead_days):
            is_left   = col == 0
            is_bottom = row == n_rows - 1

            ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
            setup_map(ax, left_labels=is_left, bottom_labels=is_bottom)

            pred  = results[ld][pred_key]
            truth = results[ld][truth_key]
            error = pred - truth

            mesh = ax.pcolormesh(lon2d, lat2d, error, cmap='RdBu_r', norm=err_norm,
                                 transform=ccrs.PlateCarree(), zorder=1, shading='auto')

            valid_truth = np.where(np.isfinite(truth), truth, np.nanmedian(truth))
            valid_pred  = np.where(np.isfinite(pred),  pred,  np.nanmedian(pred))

            # Black (AFNO RT) + lime green (GLORYS): visible across the full RdBu_r range
            ax.contour(lon2d, lat2d, valid_truth, levels=c_levels,
                       colors='#32CD32', linewidths=0.9, linestyles='dashed',
                       transform=ccrs.PlateCarree(), zorder=2)
            ax.contour(lon2d, lat2d, valid_pred, levels=c_levels,
                       colors='black', linewidths=0.9, linestyles='solid',
                       transform=ccrs.PlateCarree(), zorder=3)

            valid = error[np.isfinite(error)]
            rmse  = float(np.sqrt(np.mean(valid ** 2)))
            # Top-left corner sits over land (India/Bangladesh) — always visible
            ax.text(0.03, 0.97,
                    f'RMSE = {rmse:.3f} {unit}',
                    transform=ax.transAxes,
                    ha='left', va='top', fontsize=6.5, fontfamily='monospace',
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.5),
                    zorder=5)

            if row == 0:
                ax.annotate(f'Day +{ld}',
                            xy=(0.5, 1.03), xycoords='axes fraction',
                            ha='center', va='bottom', fontsize=9, fontweight='bold',
                            annotation_clip=False)

            col_axes.append((ax, mesh))

        col_axes[0][0].text(
            -0.18, 0.5, f'{vlabel}\n({unit})',
            transform=col_axes[0][0].transAxes,
            ha='center', va='center', fontsize=9, fontweight='bold', rotation=90,
        )

        cax = fig.add_subplot(gs[row, n_cols])
        sm  = plt.cm.ScalarMappable(cmap='RdBu_r', norm=err_norm)
        sm.set_array([])
        cb  = fig.colorbar(sm, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=7)
        cb.set_label('Error', fontsize=7)

    # Legend
    leg_handles = [
        mlines.Line2D([], [], color='black', linewidth=1.2, linestyle='solid',
                      label='AFNO RT'),
        mlines.Line2D([], [], color='#32CD32', linewidth=1.2, linestyle='dashed',
                      label='GLORYS'),
    ]
    fig.legend(handles=leg_handles, loc='lower center', ncol=2,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.01))

    dt   = datetime.strptime(init_date, '%d-%m-%Y')
    name = model_display_name(getattr(config, 'name', 'AFNO RT'))
    fig.suptitle(
        f'{name}  —  Forecast evolution  |  IC: {dt.strftime("%-d %b %Y")}  '
        f'|  Error field (RdBu)  +  AFNO RT (black) vs GLORYS (green, dashed)',
        fontsize=10, fontweight='bold', y=0.96,
    )

    exts = ['png'] + (['pdf'] if save_pdf else [])
    for ext in exts:
        path = f'{output_path}.{ext}'
        fig.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f'Saved: {path}')

    plt.close(fig)
    return f'{output_path}.png'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Entry point: load config, model, data; run rollout; build and save figure.

    Example:
        >>> # From project root:
        >>> # python src/visualization/plot_forecast_evolution.py \\
        >>> # --config_file afno_bob_surf_e11p1.yaml --init_date 09-07-2020
    """
    args = parse_args()

    lead_days = [int(x.strip()) for x in args.lead_days.split(',')]

    print('=== Loading configuration ===')
    config = load_config(args.config_file)

    device_str = args.device or config.device
    device = torch.device(
        device_str if (torch.cuda.is_available() or 'cpu' in device_str) else 'cpu'
    )
    print(f'Device     : {device}')
    print(f'Model name : {config.name}')
    print(f'Init date  : {args.init_date}')
    print(f'Lead days  : {lead_days}')

    model_path = args.model_path or f'results/models/{config.name}.pth'
    print(f'\nLoading model from {model_path} ...')
    model = load_model(config, model_path, device)

    print('Loading normalisation stats ...')
    mean, variance = load_norm_stats(config)

    print('Opening ocean and atmospheric datasets ...')
    init_year = datetime.strptime(args.init_date, '%d-%m-%Y').year
    if args.extra_data_dir and init_year >= 2021:
        extra_dir = Path(args.extra_data_dir)
        ocean_ds  = xr.open_dataset(extra_dir / 'ocean_2021_2025.nc')
        atm_ds    = xr.open_dataset(extra_dir / 'atm_2021_2025.nc')
        ref_date  = args.extra_ref_date
        print(f'  Using extended dataset: {extra_dir}  (ref_date={ref_date})')
    else:
        data_dir = Path(config.data.data_dir)
        ocean_ds = xr.open_dataset(data_dir / f'{config.data.file_prefix}.nc')
        atm_ds   = xr.open_dataset(data_dir / f'{config.data.file_prefix_atm}.nc')
        ref_date = _REF_DATE
        print(f'  Using primary dataset: {data_dir}  (ref_date={ref_date})')

    lons, lats = get_coords(ocean_ds)

    print(f'\nRunning autoregressive rollout (max lead = {max(lead_days)} days) ...')
    results = run_rollout(
        config, model, args.init_date, lead_days,
        ocean_ds, atm_ds, mean, variance, device,
        ref_date=ref_date,
    )

    print('Computing colour ranges ...')
    ranges = compute_ranges(
        results, lead_days,
        sst_range   = _parse_range(args.sst_range),
        sss_range   = _parse_range(args.sss_range),
        ssh_range   = _parse_range(args.ssh_range),
        speed_range = _parse_range(args.speed_range),
    )

    output_path = args.output or 'results/figures/forecast_evolution'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f'\nBuilding figure (style={args.style}) ...')
    if args.style == 'dual_contour':
        build_figure_dual_contour(results, lead_days, ranges, lons, lats,
                                  args.init_date, config, output_path,
                                  save_pdf=args.pdf)
    elif args.style == 'error_bg':
        build_figure_error_bg(results, lead_days, ranges, lons, lats,
                              args.init_date, config, output_path,
                              save_pdf=args.pdf)
    else:
        build_figure(results, lead_days, ranges, lons, lats,
                     args.init_date, config, output_path,
                     save_pdf=args.pdf)

    ocean_ds.close()
    atm_ds.close()


if __name__ == '__main__':
    main()

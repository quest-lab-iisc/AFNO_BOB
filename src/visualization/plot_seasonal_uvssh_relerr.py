"""
Seasonal one-day-ahead velocity + SSH forecast panel with signed relative error.

Identical layout to plot_seasonal_uvssh.py (4 rows × 6 columns, left half =
speed+quiver, right half = SSH) but error panels show signed relative error:

    err (%) = (pred − truth) / |truth| × 100

Pixels where |truth| == 0 (and land NaNs) are set to NaN.

All shared I/O, model-loading, map-drawing, and velocity utilities are imported
from plot_seasonal_uvssh.py and plot_seasonal_forecast.py.

Inputs:
    config_file (str): YAML config filename in config/ directory
    model_path (str): Path to .pth weights; defaults to results/models/{name}.pth
    dates (str): Comma-separated input dates dd-mm-yyyy (one per row)
    season_labels (str): Comma-separated row labels matching dates
    device (str): Compute device override, e.g. cuda:0 or cpu
    output (str): Output file path without extension
    speed_range (str): Optional "vmin,vmax" for speed field colourbar (m/s)
    ssh_range (str): Optional "vmin,vmax" for SSH field colourbar (m)
    err_range (float): Optional symmetric ± limit for relative-error colourbar (%)
    quiver_stride (int): Grid-point stride for quiver sub-sampling; default 8
    quiver_scale (float): Quiver scale (data-units per inch); default auto

Outputs:
    <output>.pdf (file): Vector figure for paper submission
    <output>.png (file): Raster figure at 300 dpi

Example:
    python src/visualization/plot_seasonal_uvssh_relerr.py \\
        --config_file afno_bob_surf_e06p1.yaml \\
        --device cuda:0 \\
        --quiver_stride 6
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import torch
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs

sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from plot_seasonal_forecast import (
    load_config, load_model, load_norm_stats,
    _detect_arch, build_input, model_forward,
    postprocess_var, load_truth, get_coords,
    setup_map, draw_panel,
    DEFAULT_DATES, DEFAULT_LABELS,
    _ERR_CMAP, _LON_RANGE, _LAT_RANGE, _REF_DATE,
)
from plot_seasonal_uvssh import (
    compute_speed, draw_quiver,
    _SPEED_CMAP, _SSH_CMAP,
)
from inference.utils import date_to_day_index
from data_pipeline.preprocessing.transformers.normalize_transform import PreprocessTransform


def parse_args():
    """Parse command-line arguments for the relative-error UV+SSH plot.

    Returns:
        argparse.Namespace: Parsed arguments.

    Example:
        >>> args = parse_args()
    """
    p = argparse.ArgumentParser(
        description='Seasonal one-day-ahead velocity+SSH forecast panel (relative error)'
    )
    p.add_argument('--config_file', default='afno_bob_surf_e06p1.yaml',
                   help='YAML config filename inside config/')
    p.add_argument('--model_path', default=None,
                   help='Explicit path to .pth weights; default: results/models/{name}.pth')
    p.add_argument('--dates', default=','.join(DEFAULT_DATES),
                   help='Comma-separated input dates dd-mm-yyyy, one per panel row')
    p.add_argument('--season_labels', default=','.join(DEFAULT_LABELS),
                   help='Comma-separated row labels matching --dates')
    p.add_argument('--device', default=None,
                   help='Compute device, e.g. cuda:0 or cpu (overrides config)')
    p.add_argument('--output', default=None,
                   help='Output path without extension; default: results/plots/{name}_seasonal_uvssh_relerr')
    p.add_argument('--speed_range', default=None,
                   help='Speed field colourbar "vmin,vmax" in m/s; default: auto')
    p.add_argument('--ssh_range', default=None,
                   help='SSH field colourbar "vmin,vmax" in m; default: auto')
    p.add_argument('--err_range', default=None, type=float,
                   help='Symmetric ± limit for relative-error colourbar in %%; default: auto')
    p.add_argument('--relerr_threshold', type=float, default=0.05,
                   help='|truth| threshold below which absolute error is used instead of relative (default: 0.05)')
    p.add_argument('--quiver_stride', type=int, default=8,
                   help='Grid-point stride for quiver sub-sampling (default: 8)')
    p.add_argument('--quiver_scale', type=float, default=None,
                   help='Quiver scale in data-units per inch; omit for auto-scale')
    return p.parse_args()


def _rel_error(pred, truth, threshold=0.05):
    """Compute signed relative error in percent, with absolute-error fallback.

    Where ``|truth| >= threshold``: returns ``(pred − truth) / |truth| × 100``.
    Where ``|truth| < threshold``:  returns ``pred − truth`` (absolute error, same
    units as the field) to avoid division by very small numbers.
    Land NaN pixels in truth propagate naturally.

    Args:
        pred (np.ndarray): Predicted field at native GLORYS resolution.
        truth (np.ndarray): Ground-truth field; NaN on land.
        threshold (float): Minimum |truth| for relative-error mode; pixels below
            this use absolute error instead.

    Returns:
        np.ndarray: Relative error (%) where |truth| >= threshold, else absolute
            error; NaN on land.

    Example:
        >>> err = _rel_error(speed_pred, speed_truth, threshold=0.05)
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(
            np.abs(truth) >= threshold,
            (pred - truth) / np.abs(truth) * 100.0,
            pred - truth,
        )


def _error_metrics_rel(rel_err):
    """Compute RRMSE and RMAE over valid (non-NaN) pixels of a relative error field.

    Args:
        rel_err (np.ndarray): (H, W) signed relative error (%); NaN on land.

    Returns:
        rrmse (float): Root-mean-square relative error (%).
        rmae (float): Mean absolute relative error (%).

    Example:
        >>> rrmse, rmae = _error_metrics_rel(speed_rel_err)
    """
    valid = rel_err[~np.isnan(rel_err)]
    return np.sqrt(np.mean(valid ** 2)), np.mean(np.abs(valid))


def run_season_forecasts_uvssh_rel(config, model, dates, ocean_ds, atm_ds,
                                    mean, variance, device, threshold=0.05):
    """Run one-day-ahead forecasts and collect velocity + SSH relative errors.

    Args:
        config: Configuration object.
        model (torch.nn.Module): Loaded model in eval mode.
        dates (list[str]): Input dates in dd-mm-yyyy format.
        ocean_ds (xr.Dataset): Ocean variable dataset.
        atm_ds (xr.Dataset): Atmospheric variable dataset.
        mean (dict): Normalisation means.
        variance (dict): Normalisation standard deviations.
        device (torch.device): Compute device.
        threshold (float): |truth| threshold below which absolute error is used
            instead of relative error (see _rel_error).

    Returns:
        list[dict]: One dict per date with keys:
            pred_date, speed_pred, speed_truth, speed_err (% or abs),
            u_pred, v_pred, u_truth, v_truth,
            ssh_pred, ssh_truth, ssh_err (% or abs),
            lons, lats.

    Example:
        >>> results = run_season_forecasts_uvssh_rel(config, model, ['14-01-2020'], ..., threshold=0.05)
    """
    transform = PreprocessTransform(config)
    lons, lats = get_coords(ocean_ds)
    results = []

    for date_str in dates:
        day_idx   = date_to_day_index(date_str, _REF_DATE)
        pred_date = (
            datetime.strptime(date_str, '%d-%m-%Y') + timedelta(days=1)
        ).strftime('%d-%b-%Y')
        print(f'  {date_str} → {pred_date}')

        x          = build_input(config, day_idx, ocean_ds, atm_ds, mean, variance, transform)
        output_arr = model_forward(model, x, device)

        u_pred    = postprocess_var(output_arr, config, 'uo',  mean)
        v_pred    = postprocess_var(output_arr, config, 'vo',  mean)
        ssh_pred  = postprocess_var(output_arr, config, 'zos', mean)

        u_truth   = load_truth(ocean_ds, 'uo',  day_idx)
        v_truth   = load_truth(ocean_ds, 'vo',  day_idx)
        ssh_truth = load_truth(ocean_ds, 'zos', day_idx)

        # Apply the truth's land mask to model predictions so quiver arrows
        # are not drawn over land (model outputs have no NaN on land pixels).
        land_mask = np.isnan(u_truth)
        u_pred    = np.where(land_mask, np.nan, u_pred)
        v_pred    = np.where(land_mask, np.nan, v_pred)
        ssh_pred  = np.where(land_mask, np.nan, ssh_pred)

        speed_pred  = compute_speed(u_pred,  v_pred)
        speed_truth = compute_speed(u_truth, v_truth)

        speed_err = _rel_error(speed_pred, speed_truth, threshold=threshold)
        ssh_err   = _rel_error(ssh_pred,   ssh_truth,  threshold=threshold)

        results.append({
            'pred_date':   pred_date,
            'speed_pred':  speed_pred,
            'speed_truth': speed_truth,
            'speed_err':   speed_err,
            'u_pred':      u_pred,
            'v_pred':      v_pred,
            'u_truth':     u_truth,
            'v_truth':     v_truth,
            'ssh_pred':    ssh_pred,
            'ssh_truth':   ssh_truth,
            'ssh_err':     ssh_err,
            'lons':        lons,
            'lats':        lats,
        })

    return results


def compute_color_ranges_uvssh_rel(results, speed_range=None,
                                    ssh_range=None, err_range=None):
    """Compute shared colourbar ranges for the relative-error UV+SSH figure.

    Speed / SSH field ranges are the same as the absolute-error version.
    Error panels use ± 98th percentile of |relative errors| (in %).

    Args:
        results (list[dict]): Output of run_season_forecasts_uvssh_rel.
        speed_range (tuple | None): (vmin, vmax) override for speed field (m/s).
        ssh_range (tuple | None): (vmin, vmax) override for SSH field (m).
        err_range (float | None): Symmetric ± limit override for error panels (%).

    Returns:
        dict: Keys speed_vmin, speed_vmax, speed_err_lim, ssh_vmin, ssh_vmax, ssh_err_lim.

    Example:
        >>> ranges = compute_color_ranges_uvssh_rel(results)
    """
    def _pool(arrays):
        """Concatenate valid (non-NaN) values across a list of arrays."""
        return np.concatenate([a[~np.isnan(a)].ravel() for a in arrays])

    speed_all = _pool([r['speed_pred'] for r in results] +
                      [r['speed_truth'] for r in results])
    ssh_all   = _pool([r['ssh_pred']   for r in results] +
                      [r['ssh_truth']  for r in results])
    serr_all  = _pool([r['speed_err']  for r in results])
    herr_all  = _pool([r['ssh_err']    for r in results])

    if speed_range:
        sp_vmin, sp_vmax = speed_range
    else:
        sp_vmin = 0.0
        sp_vmax = np.percentile(speed_all, 98)

    if ssh_range:
        h_vmin, h_vmax = ssh_range
    else:
        h_vmin = np.percentile(ssh_all, 2)
        h_vmax = np.percentile(ssh_all, 98)

    if err_range is not None:
        sp_elim = h_elim = err_range
    else:
        sp_elim = np.percentile(np.abs(serr_all), 98)
        h_elim  = np.percentile(np.abs(herr_all), 98)

    return {
        'speed_vmin':    sp_vmin,
        'speed_vmax':    sp_vmax,
        'speed_err_lim': sp_elim,
        'ssh_vmin':      h_vmin,
        'ssh_vmax':      h_vmax,
        'ssh_err_lim':   h_elim,
    }


def build_figure_uvssh_rel(results, season_labels, ranges, config,
                            output_path, arch_label='Model',
                            quiver_stride=8, quiver_scale=None):
    """Assemble and save the seasonal 4×6 velocity + SSH relative-error figure.

    Args:
        results (list[dict]): Season data from run_season_forecasts_uvssh_rel.
        season_labels (list[str]): Row labels (one per season / date).
        ranges (dict): Colourbar ranges from compute_color_ranges_uvssh_rel.
        config: Configuration object.
        output_path (str): Output file path without extension.
        arch_label (str): Short model-type label for column header.
        quiver_stride (int): Grid-point stride passed to draw_quiver.
        quiver_scale (float | None): Quiver scale passed to draw_quiver.

    Returns:
        str: Path to the saved PDF file.

    Example:
        >>> build_figure_uvssh_rel(results, DEFAULT_LABELS, ranges, config,
        ...                        'results/plots/seasonal_uvssh_relerr', arch_label='AFNO')
    """
    n = len(results)

    speed_norm  = mcolors.Normalize(vmin=ranges['speed_vmin'], vmax=ranges['speed_vmax'])
    speed_enorm = mcolors.TwoSlopeNorm(vcenter=0,
                                       vmin=-ranges['speed_err_lim'],
                                       vmax= ranges['speed_err_lim'])
    ssh_norm    = mcolors.Normalize(vmin=ranges['ssh_vmin'], vmax=ranges['ssh_vmax'])
    ssh_enorm   = mcolors.TwoSlopeNorm(vcenter=0,
                                       vmin=-ranges['ssh_err_lim'],
                                       vmax= ranges['ssh_err_lim'])

    col_cmaps = [_SPEED_CMAP, _SPEED_CMAP, _ERR_CMAP,
                 _SSH_CMAP,   _SSH_CMAP,   _ERR_CMAP]
    col_norms = [speed_norm,  speed_norm,  speed_enorm,
                 ssh_norm,    ssh_norm,    ssh_enorm]

    fig = plt.figure(figsize=(18, 3.6 * n + 1.4))
    gs  = GridSpec(n + 1, 6,
                   height_ratios=[1] * n + [0.06],
                   hspace=0.12, wspace=0.07,
                   top=0.91, bottom=0.08, left=0.08, right=0.99)

    # --- map axes ---
    axes = []
    for r in range(n):
        row = []
        for c in range(6):
            ax = fig.add_subplot(gs[r, c], projection=ccrs.PlateCarree())
            setup_map(ax, left_labels=(c == 0), bottom_labels=(r == n - 1))
            row.append(ax)
        axes.append(row)

    # --- draw panels ---
    for r, res in enumerate(results):
        lons, lats = res['lons'], res['lats']
        panels = [
            res['speed_truth'], res['speed_pred'], res['speed_err'],
            res['ssh_truth'],   res['ssh_pred'],   res['ssh_err'],
        ]
        for c, (data, cmap, norm) in enumerate(zip(panels, col_cmaps, col_norms)):
            draw_panel(axes[r][c], data, lons, lats, cmap, norm)

        # Quiver overlay on GLORYS speed (col 0) and model speed (col 1)
        draw_quiver(axes[r][0], res['u_truth'], res['v_truth'],
                    lons, lats, stride=quiver_stride, scale=quiver_scale)
        draw_quiver(axes[r][1], res['u_pred'],  res['v_pred'],
                    lons, lats, stride=quiver_stride, scale=quiver_scale)

    # --- RRMSE / RMAE annotations on relative error panels ---
    for r, res in enumerate(results):
        for err_arr, col, label in [
            (res['speed_err'], 2, 'speed'),
            (res['ssh_err'],   5, 'SSH'),
        ]:
            rrmse, rmae = _error_metrics_rel(err_arr)
            axes[r][col].text(
                0.03, 0.97,
                f'RRMSE={rrmse:.2f}%\nRMAE ={rmae:.2f}%',
                transform=axes[r][col].transAxes,
                ha='left', va='top', fontsize=7, fontfamily='monospace',
                bbox=dict(facecolor='white', alpha=0.65, edgecolor='none', pad=2),
            )

    # --- column sub-headers ---
    for c, title in enumerate(['GLORYS', arch_label, 'Rel. Error',
                                'GLORYS', arch_label, 'Rel. Error']):
        axes[0][c].annotate(
            title, xy=(0.5, 1.04), xycoords='axes fraction',
            ha='center', va='bottom', fontsize=9, fontweight='bold',
            annotation_clip=False,
        )

    # --- group headers ---
    for col_idx, label in [(1, 'Speed  (m/s)'), (4, 'SSH  (m)')]:
        axes[0][col_idx].annotate(
            label, xy=(0.5, 1.18), xycoords='axes fraction',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            annotation_clip=False,
        )

    # --- season labels (vertical, bottom-to-top) ---
    for r, (res, label) in enumerate(zip(results, season_labels)):
        axes[r][0].text(
            -0.14, 0.5, f'{label}  ({res["pred_date"]})',
            transform=axes[r][0].transAxes,
            ha='center', va='center', fontsize=9, fontweight='bold',
            rotation=90,
        )

    # --- colourbar axes ---
    cax_sp_f = fig.add_subplot(gs[n, 0:2])
    cax_sp_e = fig.add_subplot(gs[n, 2])
    cax_sh_f = fig.add_subplot(gs[n, 3:5])
    cax_sh_e = fig.add_subplot(gs[n, 5])

    for sm, cax, label in [
        (plt.cm.ScalarMappable(cmap=_SPEED_CMAP, norm=speed_norm),  cax_sp_f, 'Speed (m/s)'),
        (plt.cm.ScalarMappable(cmap=_ERR_CMAP,   norm=speed_enorm), cax_sp_e, 'Rel. Error (%)'),
        (plt.cm.ScalarMappable(cmap=_SSH_CMAP,   norm=ssh_norm),    cax_sh_f, 'SSH (m)'),
        (plt.cm.ScalarMappable(cmap=_ERR_CMAP,   norm=ssh_enorm),   cax_sh_e, 'Rel. Error (%)'),
    ]:
        sm.set_array([])
        fig.colorbar(sm, cax=cax, orientation='horizontal', label=label)

    fig.suptitle(
        f'One-day-ahead velocity and SSH forecasts — {arch_label} Model (relative error)',
        fontsize=12, fontweight='bold', y=0.965,
    )

    for ext in ('pdf', 'png'):
        path = f'{output_path}.{ext}'
        fig.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f'Saved: {path}')

    plt.close(fig)
    return f'{output_path}.pdf'


def main():
    """Entry point: load config + model, run forecasts, build relative-error UV+SSH figure.

    Example:
        >>> # run from project root:
        >>> # python src/visualization/plot_seasonal_uvssh_relerr.py --config_file afno_bob_surf_e06p1.yaml
    """
    args = parse_args()

    dates         = [d.strip() for d in args.dates.split(',')]
    season_labels = [s.strip() for s in args.season_labels.split(',')]

    if len(dates) != len(season_labels):
        raise ValueError(
            f'--dates has {len(dates)} items but --season_labels has {len(season_labels)}'
        )

    print('=== Loading configuration ===')
    config = load_config(args.config_file)

    device_str = args.device or config.device
    device = torch.device(
        device_str if torch.cuda.is_available() or 'cpu' in device_str else 'cpu'
    )
    print(f'Device: {device}')

    arch = _detect_arch(config)
    arch_label = {'afno': 'AFNO', 'tfno': 'TFNO'}.get(arch, arch.upper())
    print(f'Architecture: {arch_label}')

    model_path = args.model_path or f'{config.results.model_dir}/{config.name}.pth'
    if not Path(model_path).exists():
        raise FileNotFoundError(f'Model weights not found: {model_path}')
    print(f'=== Loading model: {model_path} ===')
    model = load_model(config, model_path, device)

    print('=== Loading normalisation statistics ===')
    mean, variance = load_norm_stats(config)

    print('=== Opening datasets ===')
    data_dir = Path(config.data.data_dir)
    ocean_ds = xr.open_dataset(data_dir / f'{config.data.file_prefix}.nc')
    atm_ds   = xr.open_dataset(data_dir / f'{config.data.file_prefix_atm}.nc')

    print(f'Relative-error threshold: {args.relerr_threshold}')
    print('=== Running seasonal forecasts ===')
    results = run_season_forecasts_uvssh_rel(
        config, model, dates, ocean_ds, atm_ds, mean, variance, device,
        threshold=args.relerr_threshold,
    )

    speed_range = (
        tuple(float(v) for v in args.speed_range.split(',')) if args.speed_range else None
    )
    ssh_range = (
        tuple(float(v) for v in args.ssh_range.split(',')) if args.ssh_range else None
    )
    ranges = compute_color_ranges_uvssh_rel(results,
                                             speed_range=speed_range,
                                             ssh_range=ssh_range,
                                             err_range=args.err_range)

    output_path = (
        args.output or
        f'{config.results.plot_dir}/{config.name}_seasonal_uvssh_relerr'
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print('=== Building figure ===')
    build_figure_uvssh_rel(results, season_labels, ranges, config, output_path,
                            arch_label=arch_label,
                            quiver_stride=args.quiver_stride,
                            quiver_scale=args.quiver_scale)
    print('=== Done ===')


if __name__ == '__main__':
    main()

"""
Seasonal one-day-ahead forecast panel for ocean velocity and SSH.

Produces a 4-row × 6-column figure:
  - Left 3 columns  : Current speed (m/s) with UV quiver overlay
                      [GLORYS | Model | Signed speed error]
  - Right 3 columns : Sea-surface height / SSH (m)
                      [GLORYS | Model | Signed SSH error]

Error panels use a diverging RdBu_r colorbar (blue = under-prediction,
red = over-prediction, white = 0).

All I/O utilities (config loading, model loading, normalisation, map drawing)
are imported from plot_seasonal_forecast.py.

Inputs:
    config_file (str): YAML config filename in config/ directory
    model_path (str): Path to .pth weights; defaults to results/models/{name}.pth
    dates (str): Comma-separated input dates dd-mm-yyyy (one per row)
    season_labels (str): Comma-separated row labels matching dates
    device (str): Compute device override, e.g. cuda:0 or cpu
    output (str): Output file path without extension
    speed_range (str): Optional "vmin,vmax" for speed colourbar (m/s); default auto
    ssh_range (str): Optional "vmin,vmax" for SSH colourbar (m); default auto
    quiver_stride (int): Grid-point stride for quiver sub-sampling; default 8
    quiver_scale (float): Quiver scale (data-units per inch); default auto

Outputs:
    <output>.pdf (file): Vector figure for paper submission
    <output>.png (file): Raster figure at 300 dpi

Example:
    python src/visualization/plot_seasonal_uvssh.py \\
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
from inference.utils import date_to_day_index
from data_pipeline.preprocessing.transformers.normalize_transform import PreprocessTransform

try:
    import cmocean
    _SPEED_CMAP = cmocean.cm.speed
    _SSH_CMAP   = cmocean.cm.matter   # sequential: SSH is always positive
except ImportError:
    _SPEED_CMAP = 'YlOrBr'
    _SSH_CMAP   = 'YlOrBr'


def parse_args():
    """Parse command-line arguments for the UV+SSH seasonal forecast plot.

    Returns:
        argparse.Namespace: Parsed arguments.

    Example:
        >>> args = parse_args()
    """
    p = argparse.ArgumentParser(
        description='Seasonal one-day-ahead velocity (speed+quiver) and SSH forecast panel'
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
                   help='Output path without extension; default: results/plots/{name}_seasonal_uvssh')
    p.add_argument('--speed_range', default=None,
                   help='Speed colourbar "vmin,vmax" in m/s; default: auto (0, 98th pct)')
    p.add_argument('--ssh_range', default=None,
                   help='SSH colourbar "vmin,vmax" in m; default: auto (2nd–98th pct)')
    p.add_argument('--quiver_stride', type=int, default=8,
                   help='Grid-point stride for quiver sub-sampling (default: 8)')
    p.add_argument('--quiver_scale', type=float, default=None,
                   help='Quiver scale in data-units per inch; omit for matplotlib auto-scale')
    return p.parse_args()


def compute_speed(u, v):
    """Compute current speed magnitude from U and V components.

    Args:
        u (np.ndarray): Zonal velocity field (H, W); NaN on land.
        v (np.ndarray): Meridional velocity field (H, W); NaN on land.

    Returns:
        np.ndarray: Speed magnitude (H, W) in the same units as u/v; NaN on land.

    Example:
        >>> speed = compute_speed(u_field, v_field)
    """
    return np.sqrt(u ** 2 + v ** 2)


def draw_quiver(ax, u, v, lons, lats, stride=8, scale=None):
    """Overlay UV quiver arrows on a cartopy axes.

    Land pixels (NaN in u or v) are excluded. The arrow colour is black so
    that the vectors remain visible over any speed colourmap.

    Args:
        ax: Cartopy GeoAxes to draw on.
        u (np.ndarray): Zonal velocity (H, W); NaN on land.
        v (np.ndarray): Meridional velocity (H, W); NaN on land.
        lons (np.ndarray): 1-D longitude array of length W.
        lats (np.ndarray): 1-D latitude array of length H.
        stride (int): Sub-sampling stride in grid points.
        scale (float | None): Quiver scale (data-units per inch); None = auto.

    Example:
        >>> draw_quiver(ax, u_pred, v_pred, lons, lats, stride=8)
    """
    qlons = lons[::stride]
    qlats = lats[::stride]
    qu    = u[::stride, ::stride]
    qv    = v[::stride, ::stride]

    lon2d, lat2d = np.meshgrid(qlons, qlats)
    valid = ~(np.isnan(qu) | np.isnan(qv))

    kw = dict(
        transform=ccrs.PlateCarree(),
        color='k', alpha=0.75,
        width=0.003, headwidth=3, headlength=4,
        zorder=5,
    )
    if scale is not None:
        kw['scale'] = scale
        kw['scale_units'] = 'inches'

    ax.quiver(lon2d[valid], lat2d[valid], qu[valid], qv[valid], **kw)


def run_season_forecasts_uvssh(config, model, dates, ocean_ds, atm_ds,
                                mean, variance, device):
    """Run one-day-ahead forecasts and collect velocity + SSH results.

    For each input date produces the predicted and ground-truth speed magnitude,
    UV component fields (for quiver), and SSH, all at native GLORYS resolution.

    Args:
        config: Configuration object.
        model (torch.nn.Module): Loaded model in eval mode.
        dates (list[str]): Input dates in dd-mm-yyyy format.
        ocean_ds (xr.Dataset): Ocean variable dataset.
        atm_ds (xr.Dataset): Atmospheric variable dataset.
        mean (dict): Normalisation means.
        variance (dict): Normalisation standard deviations.
        device (torch.device): Compute device.

    Returns:
        list[dict]: One dict per date with keys:
            pred_date, speed_pred, speed_truth, speed_err,
            u_pred, v_pred, u_truth, v_truth,
            ssh_pred, ssh_truth, ssh_err,
            lons, lats.

    Example:
        >>> results = run_season_forecasts_uvssh(config, model, ['14-01-2020'], ...)
    """
    transform = PreprocessTransform(config)
    lons, lats = get_coords(ocean_ds)
    results = []

    for date_str in dates:
        day_idx  = date_to_day_index(date_str, _REF_DATE)
        pred_date = (
            datetime.strptime(date_str, '%d-%m-%Y') + timedelta(days=1)
        ).strftime('%d-%b-%Y')
        print(f'  {date_str} → {pred_date}')

        x          = build_input(config, day_idx, ocean_ds, atm_ds, mean, variance, transform)
        output_arr = model_forward(model, x, device)

        u_pred     = postprocess_var(output_arr, config, 'uo',  mean)
        v_pred     = postprocess_var(output_arr, config, 'vo',  mean)
        ssh_pred   = postprocess_var(output_arr, config, 'zos', mean)

        u_truth    = load_truth(ocean_ds, 'uo',  day_idx)
        v_truth    = load_truth(ocean_ds, 'vo',  day_idx)
        ssh_truth  = load_truth(ocean_ds, 'zos', day_idx)

        # Apply the truth's land mask to model predictions so quiver arrows
        # are not drawn over land (model outputs have no NaN on land pixels).
        land_mask = np.isnan(u_truth)
        u_pred    = np.where(land_mask, np.nan, u_pred)
        v_pred    = np.where(land_mask, np.nan, v_pred)
        ssh_pred  = np.where(land_mask, np.nan, ssh_pred)

        speed_pred  = compute_speed(u_pred,  v_pred)
        speed_truth = compute_speed(u_truth, v_truth)

        # Land NaNs propagate from truth into error naturally
        speed_err = speed_pred  - speed_truth
        ssh_err   = ssh_pred    - ssh_truth

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


def compute_color_ranges_uvssh(results, speed_range=None, ssh_range=None):
    """Compute shared colourbar ranges from all season data.

    Speed field   : 0 → 98th percentile of all speed values pooled across seasons.
    SSH field     : 2nd–98th percentile, symmetric around 0 to support TwoSlopeNorm.
    Error panels  : ± 98th percentile of absolute errors (symmetric).

    Args:
        results (list[dict]): Output of run_season_forecasts_uvssh.
        speed_range (tuple | None): (vmin, vmax) override for speed panels (m/s).
        ssh_range (tuple | None): (vmin, vmax) override for SSH panels (m).

    Returns:
        dict: Keys speed_vmax, speed_err_lim, ssh_vmin, ssh_vmax, ssh_err_lim.

    Example:
        >>> ranges = compute_color_ranges_uvssh(results)
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

    sp_elim = np.percentile(np.abs(serr_all), 98)
    h_elim  = np.percentile(np.abs(herr_all), 98)

    return {
        'speed_vmin':   sp_vmin,
        'speed_vmax':   sp_vmax,
        'speed_err_lim': sp_elim,
        'ssh_vmin':     h_vmin,
        'ssh_vmax':     h_vmax,
        'ssh_err_lim':  h_elim,
    }


def _error_metrics(err):
    """Compute RMSE and MAE over valid (non-NaN) pixels of an error field.

    Args:
        err (np.ndarray): (H, W) signed error array; NaN on land.

    Returns:
        rmse (float): Root-mean-square error over ocean pixels.
        mae (float): Mean absolute error over ocean pixels.

    Example:
        >>> rmse, mae = _error_metrics(speed_err)
    """
    valid = err[~np.isnan(err)]
    return np.sqrt(np.mean(valid ** 2)), np.mean(np.abs(valid))


def build_figure_uvssh(results, season_labels, ranges, config,
                        output_path, arch_label='Model',
                        quiver_stride=8, quiver_scale=None):
    """Assemble and save the seasonal 4×6 velocity + SSH forecast panel.

    Args:
        results (list[dict]): Season data from run_season_forecasts_uvssh.
        season_labels (list[str]): Row labels (one per season / date).
        ranges (dict): Colourbar ranges from compute_color_ranges_uvssh.
        config: Configuration object.
        output_path (str): Output file path without extension.
        arch_label (str): Short model-type label for column header (e.g. ``'AFNO'``).
        quiver_stride (int): Grid-point stride passed to draw_quiver.
        quiver_scale (float | None): Quiver scale passed to draw_quiver.

    Returns:
        str: Path to the saved PDF file.

    Example:
        >>> build_figure_uvssh(results, DEFAULT_LABELS, ranges, config,
        ...                    'results/plots/seasonal_uvssh', arch_label='AFNO')
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

    # --- RMSE / MAE annotations on error panels (cols 2 and 5) ---
    for r, res in enumerate(results):
        for err_arr, col, unit in [
            (res['speed_err'], 2, 'm/s'),
            (res['ssh_err'],   5, 'm'),
        ]:
            rmse, mae = _error_metrics(err_arr)
            axes[r][col].text(
                0.03, 0.97,
                f'RMSE={rmse:.4f} {unit}\nMAE ={mae:.4f} {unit}',
                transform=axes[r][col].transAxes,
                ha='left', va='top', fontsize=7, fontfamily='monospace',
                bbox=dict(facecolor='white', alpha=0.65, edgecolor='none', pad=2),
            )

    # --- column sub-headers ---
    for c, title in enumerate(['GLORYS', arch_label, 'Error',
                                'GLORYS', arch_label, 'Error']):
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
        (plt.cm.ScalarMappable(cmap=_ERR_CMAP,   norm=speed_enorm), cax_sp_e, 'Error (m/s)'),
        (plt.cm.ScalarMappable(cmap=_SSH_CMAP,   norm=ssh_norm),    cax_sh_f, 'SSH (m)'),
        (plt.cm.ScalarMappable(cmap=_ERR_CMAP,   norm=ssh_enorm),   cax_sh_e, 'Error (m)'),
    ]:
        sm.set_array([])
        fig.colorbar(sm, cax=cax, orientation='horizontal', label=label)

    fig.suptitle(
        f'One-day-ahead velocity and SSH forecasts — {arch_label} Model',
        fontsize=12, fontweight='bold', y=0.965,
    )

    for ext in ('pdf', 'png'):
        path = f'{output_path}.{ext}'
        fig.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f'Saved: {path}')

    plt.close(fig)
    return f'{output_path}.pdf'


def main():
    """Entry point: load config + model, run forecasts, build velocity+SSH figure.

    Example:
        >>> # run from project root:
        >>> # python src/visualization/plot_seasonal_uvssh.py --config_file afno_bob_surf_e06p1.yaml
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
    from plot_seasonal_forecast import model_display_name, _NAME_MAP
    arch_label = model_display_name(config.name) if config.name in _NAME_MAP \
        else {'afno': 'AFNO', 'tfno': 'TFNO'}.get(arch, arch.upper())
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

    print('=== Running seasonal forecasts ===')
    results = run_season_forecasts_uvssh(
        config, model, dates, ocean_ds, atm_ds, mean, variance, device
    )

    speed_range = (
        tuple(float(v) for v in args.speed_range.split(',')) if args.speed_range else None
    )
    ssh_range = (
        tuple(float(v) for v in args.ssh_range.split(',')) if args.ssh_range else None
    )
    ranges = compute_color_ranges_uvssh(results,
                                         speed_range=speed_range,
                                         ssh_range=ssh_range)

    output_path = (
        args.output or
        f'{config.results.plot_dir}/{config.name}_seasonal_uvssh'
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print('=== Building figure ===')
    build_figure_uvssh(results, season_labels, ranges, config, output_path,
                        arch_label=arch_label,
                        quiver_stride=args.quiver_stride,
                        quiver_scale=args.quiver_scale)
    print('=== Done ===')


if __name__ == '__main__':
    main()

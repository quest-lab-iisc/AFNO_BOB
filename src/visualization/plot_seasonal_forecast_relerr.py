"""
Seasonal one-day-ahead forecast comparison panel with signed relative error.

Produces a 4-row × 6-column figure comparing GLORYS reanalysis against model
predictions for SST (thetao) and SSS (so) across four seasonal regimes. Each
row shows one season; columns show [GLORYS | Model | Rel. Error (%)] for SST
then SSS.

Relative error is defined as:
    err (%) = (pred − truth) / |truth| × 100

and is displayed with a diverging RdBu_r colorbar (blue = under-prediction,
red = over-prediction, white = 0 %).

All shared I/O utilities (config loading, model loading, normalisation,
data assembly, map drawing) are imported from plot_seasonal_forecast.py.
Only the error computation, colour ranges, and figure labels differ.

Inputs:
    config_file (str): YAML config filename in config/ directory
    model_path (str): Path to .pth weights; defaults to results/models/{name}.pth
    dates (str): Comma-separated input dates dd-mm-yyyy (one per row)
    season_labels (str): Comma-separated row labels matching dates
    device (str): Compute device override, e.g. cuda:0 or cpu
    output (str): Output file path without extension
    sst_range (str): Optional "vmin,vmax" for SST field colourbar (°C); default auto
    sss_range (str): Optional "vmin,vmax" for SSS field colourbar (psu); default auto
    err_range (str): Optional "±limit" for relative error colourbar (%); default auto

Outputs:
    <output>.pdf (file): Vector figure for paper submission
    <output>.png (file): Raster figure at 300 dpi

Example:
    python src/visualization/plot_seasonal_forecast_relerr.py \\
        --config_file afno_bob_surf_e06p1.yaml \\
        --dates 14-01-2020,19-04-2020,14-07-2020,19-10-2020 \\
        --season_labels "Winter,Pre-monsoon,Monsoon,Post-monsoon" \\
        --device cuda:0
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Allow imports from src/
sys.path.append(str(Path(__file__).parent.parent))
# Allow importing shared utilities from the same visualization package
sys.path.insert(0, str(Path(__file__).parent))

from plot_seasonal_forecast import (
    load_config,
    load_model,
    load_norm_stats,
    _detect_arch,
    build_input,
    model_forward,
    postprocess_var,
    load_truth,
    get_coords,
    setup_map,
    draw_panel,
    DEFAULT_DATES,
    DEFAULT_LABELS,
    _SST_CMAP,
    _SSS_CMAP,
    _ERR_CMAP,
    _LON_RANGE,
    _LAT_RANGE,
    _REF_DATE,
)

from inference.utils import date_to_day_index
from data_pipeline.preprocessing.transformers.normalize_transform import PreprocessTransform


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.

    Example:
        >>> args = parse_args()
    """
    p = argparse.ArgumentParser(
        description='Seasonal one-day-ahead SST/SSS forecast panel with relative error'
    )
    p.add_argument('--config_file', default='afno_bob_surf_e06p1.yaml',
                   help='YAML config filename inside config/ (selects model + data paths)')
    p.add_argument('--model_path', default=None,
                   help='Explicit path to .pth weights; default: results/models/{name}.pth')
    p.add_argument('--dates', default=','.join(DEFAULT_DATES),
                   help='Comma-separated input dates dd-mm-yyyy, one per panel row')
    p.add_argument('--season_labels', default=','.join(DEFAULT_LABELS),
                   help='Comma-separated row labels matching --dates')
    p.add_argument('--device', default=None,
                   help='Compute device, e.g. cuda:0 or cpu (overrides config)')
    p.add_argument('--output', default=None,
                   help='Output path without extension; default: results/plots/{name}_seasonal_relerr')
    p.add_argument('--sst_range', default=None,
                   help='SST field colourbar "vmin,vmax" in °C; default: auto from data')
    p.add_argument('--sss_range', default=None,
                   help='SSS field colourbar "vmin,vmax" in psu; default: auto from data')
    p.add_argument('--err_range', default=None,
                   help='Relative error colourbar limit in %% (symmetric ±limit); default: auto')
    p.add_argument('--relerr_threshold', type=float, default=0.05,
                   help='|truth| threshold below which absolute error is used instead of relative (default: 0.05)')
    return p.parse_args()


def _rel_error(pred, truth, threshold=0.05):
    """Compute signed relative error in percent, with absolute-error fallback.

    Where ``|truth| >= threshold``: returns ``(pred − truth) / |truth| × 100``.
    Where ``|truth| < threshold``:  returns ``pred − truth`` (absolute error) to
    avoid division by very small numbers. Land NaN pixels propagate naturally.

    Args:
        pred (np.ndarray): Predicted field at native GLORYS resolution.
        truth (np.ndarray): Ground-truth field at native GLORYS resolution; NaN on land.
        threshold (float): Minimum |truth| for relative-error mode.

    Returns:
        np.ndarray: Relative error (%) where |truth| >= threshold, else absolute
            error; NaN on land.

    Example:
        >>> err_pct = _rel_error(sst_pred, sst_truth, threshold=0.05)
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
        rel_err (np.ndarray): (H, W) signed relative error in %; NaN on land.

    Returns:
        rrmse (float): Root-mean-square relative error (%).
        rmae (float): Mean absolute relative error (%).

    Example:
        >>> rrmse, rmae = _error_metrics_rel(sst_rel_err)
    """
    valid = rel_err[~np.isnan(rel_err)]
    rrmse = np.sqrt(np.mean(valid ** 2))
    rmae  = np.mean(np.abs(valid))
    return rrmse, rmae


def run_season_forecasts_rel(config, model, dates, ocean_ds, atm_ds, mean, variance,
                             device, threshold=0.05):
    """Run one-day-ahead forecasts for each input date and collect SST/SSS relative errors.

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
        list[dict]: One dict per date with keys: pred_date (str), sst_pred,
            sst_truth, sst_err (% or abs), sss_pred, sss_truth, sss_err
            (% or abs), lons, lats — all at native GLORYS resolution.

    Example:
        >>> results = run_season_forecasts_rel(config, model, ['14-01-2020'], ..., threshold=0.05)
    """
    transform = PreprocessTransform(config)
    lons, lats = get_coords(ocean_ds)
    results = []

    for date_str in dates:
        day_idx = date_to_day_index(date_str, _REF_DATE)
        pred_date = (
            datetime.strptime(date_str, '%d-%m-%Y') + timedelta(days=1)
        ).strftime('%d-%b-%Y')
        print(f'  {date_str} → {pred_date}')

        x = build_input(config, day_idx, ocean_ds, atm_ds, mean, variance, transform)
        output_arr = model_forward(model, x, device)

        sst_pred  = postprocess_var(output_arr, config, 'thetao', mean)
        sss_pred  = postprocess_var(output_arr, config, 'so',     mean)
        sst_truth = load_truth(ocean_ds, 'thetao', day_idx)
        sss_truth = load_truth(ocean_ds, 'so',     day_idx)

        sst_err = _rel_error(sst_pred, sst_truth, threshold=threshold)
        sss_err = _rel_error(sss_pred, sss_truth, threshold=threshold)

        results.append({
            'pred_date': pred_date,
            'sst_pred':  sst_pred,
            'sst_truth': sst_truth,
            'sst_err':   sst_err,
            'sss_pred':  sss_pred,
            'sss_truth': sss_truth,
            'sss_err':   sss_err,
            'lons':      lons,
            'lats':      lats,
        })

    return results


def compute_color_ranges_rel(results, sst_range=None, sss_range=None, err_range=None):
    """Compute shared colourbar ranges from all season data.

    Field panels: 2nd–98th percentile of valid ocean pixels pooled across seasons.
    Error panels: ± 98th percentile of |relative errors| (symmetric, in %).

    Args:
        results (list[dict]): Output of run_season_forecasts_rel.
        sst_range (tuple | None): (vmin, vmax) override for SST field panels (°C).
        sss_range (tuple | None): (vmin, vmax) override for SSS field panels (psu).
        err_range (float | None): Symmetric ± limit override for relative error panels (%).

    Returns:
        dict: Keys sst_vmin, sst_vmax, sst_err_lim, sss_vmin, sss_vmax, sss_err_lim.

    Example:
        >>> ranges = compute_color_ranges_rel(results, sst_range=(28.0, 31.0))
    """
    def _pool(arrays):
        """Concatenate valid (non-NaN) values from a list of arrays."""
        return np.concatenate([a[~np.isnan(a)].ravel() for a in arrays])

    sst_all  = _pool([r['sst_pred']  for r in results] + [r['sst_truth'] for r in results])
    sss_all  = _pool([r['sss_pred']  for r in results] + [r['sss_truth'] for r in results])
    sst_errs = _pool([r['sst_err']   for r in results])
    sss_errs = _pool([r['sss_err']   for r in results])

    if sst_range:
        s_vmin, s_vmax = sst_range
    else:
        s_vmin, s_vmax = np.percentile(sst_all, 2), np.percentile(sst_all, 98)

    if sss_range:
        ss_vmin, ss_vmax = sss_range
    else:
        ss_vmin, ss_vmax = np.percentile(sss_all, 2), np.percentile(sss_all, 98)

    if err_range is not None:
        sst_elim = sss_elim = err_range
    else:
        sst_elim = np.percentile(np.abs(sst_errs), 98)
        sss_elim = np.percentile(np.abs(sss_errs), 98)

    return {
        'sst_vmin':    s_vmin,
        'sst_vmax':    s_vmax,
        'sst_err_lim': sst_elim,
        'sss_vmin':    ss_vmin,
        'sss_vmax':    ss_vmax,
        'sss_err_lim': sss_elim,
    }


def build_figure_rel(results, season_labels, ranges, config, output_path, arch_label='Model'):
    """Assemble and save the seasonal 4×6 forecast panel with relative error.

    Args:
        results (list[dict]): Season forecast data from run_season_forecasts_rel.
        season_labels (list[str]): Row labels (one per season / date).
        ranges (dict): Shared colourbar ranges from compute_color_ranges_rel.
        config: Configuration object (for the figure title).
        output_path (str): Output file path without extension.
        arch_label (str): Short model-type label for column header, e.g. ``'AFNO'``.

    Returns:
        str: Path to the saved PDF file.

    Example:
        >>> build_figure_rel(results, DEFAULT_LABELS, ranges, config,
        ...                  'results/plots/seasonal_relerr', arch_label='AFNO')
    """
    n = len(results)

    sst_norm  = mcolors.Normalize(ranges['sst_vmin'],    ranges['sst_vmax'])
    sst_enorm = mcolors.TwoSlopeNorm(vcenter=0,
                                     vmin=-ranges['sst_err_lim'],
                                     vmax= ranges['sst_err_lim'])
    sss_norm  = mcolors.Normalize(ranges['sss_vmin'],    ranges['sss_vmax'])
    sss_enorm = mcolors.TwoSlopeNorm(vcenter=0,
                                     vmin=-ranges['sss_err_lim'],
                                     vmax= ranges['sss_err_lim'])

    col_cmaps = [_SST_CMAP, _SST_CMAP, _ERR_CMAP, _SSS_CMAP, _SSS_CMAP, _ERR_CMAP]
    col_norms = [sst_norm,  sst_norm,  sst_enorm, sss_norm,  sss_norm,  sss_enorm]

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
            res['sst_truth'], res['sst_pred'], res['sst_err'],
            res['sss_truth'], res['sss_pred'], res['sss_err'],
        ]
        for c, (data, cmap, norm) in enumerate(zip(panels, col_cmaps, col_norms)):
            draw_panel(axes[r][c], data, lons, lats, cmap, norm)

    # --- RRMSE / RMAE annotations on relative error panels (cols 2 and 5) ---
    for r, res in enumerate(results):
        for err_arr, col in [(res['sst_err'], 2), (res['sss_err'], 5)]:
            rrmse, rmae = _error_metrics_rel(err_arr)
            axes[r][col].text(
                0.03, 0.97,
                f'RRMSE={rrmse:.2f}%\nRMAE ={rmae:.2f}%',
                transform=axes[r][col].transAxes,
                ha='left', va='top', fontsize=7, fontfamily='monospace',
                bbox=dict(facecolor='white', alpha=0.65, edgecolor='none', pad=2),
            )

    # --- column sub-headers (above first row only) ---
    for c, title in enumerate(['GLORYS', arch_label, 'Rel. Error',
                                'GLORYS', arch_label, 'Rel. Error']):
        axes[0][c].annotate(
            title, xy=(0.5, 1.04), xycoords='axes fraction',
            ha='center', va='bottom', fontsize=9, fontweight='bold',
            annotation_clip=False,
        )

    # --- group headers (SST / SSS) above middle column of each group ---
    for col_idx, label in [(1, 'SST  (°C)'), (4, 'SSS  (psu)')]:
        axes[0][col_idx].annotate(
            label, xy=(0.5, 1.18), xycoords='axes fraction',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            annotation_clip=False,
        )

    # --- season labels on the left of each row (vertical, bottom-to-top) ---
    for r, (res, label) in enumerate(zip(results, season_labels)):
        axes[r][0].text(
            -0.14, 0.5, f'{label}  ({res["pred_date"]})',
            transform=axes[r][0].transAxes,
            ha='center', va='center', fontsize=9, fontweight='bold',
            rotation=90,
        )

    # --- colourbar axes (last GridSpec row, merged cells) ---
    cax_sst_f = fig.add_subplot(gs[n, 0:2])
    cax_sst_e = fig.add_subplot(gs[n, 2])
    cax_sss_f = fig.add_subplot(gs[n, 3:5])
    cax_sss_e = fig.add_subplot(gs[n, 5])

    for sm, cax, label in [
        (plt.cm.ScalarMappable(cmap=_SST_CMAP, norm=sst_norm),  cax_sst_f, 'SST (°C)'),
        (plt.cm.ScalarMappable(cmap=_ERR_CMAP,  norm=sst_enorm), cax_sst_e, 'Rel. Error (%)'),
        (plt.cm.ScalarMappable(cmap=_SSS_CMAP,  norm=sss_norm),  cax_sss_f, 'SSS (psu)'),
        (plt.cm.ScalarMappable(cmap=_ERR_CMAP,  norm=sss_enorm), cax_sss_e, 'Rel. Error (%)'),
    ]:
        sm.set_array([])
        fig.colorbar(sm, cax=cax, orientation='horizontal', label=label)

    fig.suptitle(
        f'One-day-ahead SST and SSS forecasts — {arch_label} Model (relative error)',
        fontsize=12, fontweight='bold', y=0.965,
    )

    for ext in ('pdf', 'png'):
        path = f'{output_path}.{ext}'
        fig.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f'Saved: {path}')

    plt.close(fig)
    return f'{output_path}.pdf'


def main():
    """Entry point: parse arguments, load model, run forecasts, build relative-error figure.

    Example:
        >>> # run from project root:
        >>> # python src/visualization/plot_seasonal_forecast_relerr.py --config_file afno_bob_surf_e06p1.yaml
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
    import xarray as xr
    data_dir = Path(config.data.data_dir)
    ocean_ds = xr.open_dataset(data_dir / f'{config.data.file_prefix}.nc')
    atm_ds   = xr.open_dataset(data_dir / f'{config.data.file_prefix_atm}.nc')

    print(f'Relative-error threshold: {args.relerr_threshold}')
    print('=== Running seasonal forecasts ===')
    results = run_season_forecasts_rel(
        config, model, dates, ocean_ds, atm_ds, mean, variance, device,
        threshold=args.relerr_threshold,
    )

    sst_range = (
        tuple(float(v) for v in args.sst_range.split(',')) if args.sst_range else None
    )
    sss_range = (
        tuple(float(v) for v in args.sss_range.split(',')) if args.sss_range else None
    )
    err_range = float(args.err_range) if args.err_range else None

    ranges = compute_color_ranges_rel(results,
                                      sst_range=sst_range,
                                      sss_range=sss_range,
                                      err_range=err_range)

    output_path = (
        args.output or
        f'{config.results.plot_dir}/{config.name}_seasonal_relerr'
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print('=== Building figure ===')
    build_figure_rel(results, season_labels, ranges, config, output_path,
                     arch_label=arch_label)
    print('=== Done ===')


if __name__ == '__main__':
    main()

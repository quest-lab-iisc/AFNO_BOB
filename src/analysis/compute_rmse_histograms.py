"""Compute and plot per-IC RMSE distributions at lead days +1, +3, +5, +7, +9.

For every valid IC date in 2020-2025, runs AFNO RT (E14) autoregressive rollout
and records one RMSE scalar per output variable at each of the five lead days.
With ~2,059 valid IC dates this yields ~2,059 RMSE values per (variable, lead)
cell — small enough to store as raw arrays so the figure can be regenerated and
styled without re-running the model.

The final figure is a 2×5 layout:
  Row 1 — probability density (step-histogram) of per-IC RMSE values.
  Row 2 — cumulative distribution (CDF), showing what percentage of IC dates
           achieve RMSE below a given threshold.

Both rows use a validated ordinal single-blue ramp (light→dark, +1d→+9d) and
box-framed axes with hairline grid lines.

Inputs:
    --config_file (str)    : YAML config in config/ (default: afno_bob_surf_e14.yaml).
    --model_path (str)     : Path to .pth weights (default: results/models/AFNO_BoB_Surf_E14.pth).
    --data_dir (str)       : Primary dataset directory (default: from config).
    --extra_data_dir (str) : 2021-2025 dataset directory
                             (default: data/2021_2025).
    --extra_num_days (int) : IC dates to evaluate in extended period (default: 1711).
    --output_dir (str)     : Directory for rmse_histograms.npz (default:
                             results/AFNO_BoB_Surf_E14/).
    --output_fig (str)     : Stem path for figure outputs, without extension
                             (default: results/figures/fig_rmse_histograms).
    --device (str)         : PyTorch device (default: from config).
    --skip_compute         : If set, load existing .npz and re-plot only.
    --pdf                  : Save an additional PDF alongside the PNG.

Outputs:
    <output_dir>/rmse_histograms.npz   — raw per-IC RMSE arrays per variable/lead.
    <output_fig>.png                   — multi-panel RMSE distribution figure.
    <output_fig>.pdf                   — PDF version (if --pdf is set).
    logs/compute_rmse_histograms.log   — verbatim console log.

Example:
    conda activate BoB_Surf_2
    python src/analysis/compute_rmse_histograms.py \\
        --config_file afno_bob_surf_e14.yaml \\
        --model_path results/models/AFNO_BoB_Surf_E14.pth \\
        --extra_data_dir data/2021_2025 \\
        --device cuda:1 --pdf
"""

import argparse
import importlib
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from configmypy import ConfigPipeline, YamlConfig
from inference.run_metrics import date_to_day_index
from training.utils.experiment_logger import TeeLogger, cleanup_logging


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VAR_ORDER = ['thetao', 'so', 'zos', 'uo', 'vo']

VAR_META = {
    'thetao': dict(label='SST',        unit='°C',   n_bins=60),
    'so':     dict(label='SSS',        unit='psu',  n_bins=60),
    'zos':    dict(label='SSH',        unit='m',    n_bins=60),
    'uo':     dict(label='U current',  unit='m/s',  n_bins=60),
    'vo':     dict(label='V current',  unit='m/s',  n_bins=60),
}

LEAD_DAYS = [1, 3, 5, 7, 9]

# Categorical palette — first 5 slots (blue, orange, aqua, yellow, magenta)
# assigned in fixed order to lead days so each is immediately distinguishable.
LEAD_COLORS = {
    1: '#2a78d6',   # blue
    3: '#eb6834',   # orange
    5: '#1baf7a',   # aqua
    7: '#eda100',   # yellow
    9: '#e87ba4',   # magenta
}
LEAD_LABELS = {1: '+1 day', 3: '+3 days', 5: '+5 days', 7: '+7 days', 9: '+9 days'}

PRIMARY_REF  = '01-01-1993'
EXTENDED_REF = '01-01-2021'

# ---------------------------------------------------------------------------
# Model-type dispatch
# ---------------------------------------------------------------------------

_MODULE_MAP = {
    'afno': 'inference.run_metrics',
    'fno':  'inference.run_metrics_fno',
    'tfno': 'inference.run_metrics_tfno',
    'uno':  'inference.run_metrics_uno',
}


def get_inference_module(model_type: str):
    """Import and return the inference module for the requested model type.

    Args:
        model_type (str): One of 'afno', 'fno', 'tfno', 'uno'.

    Returns:
        module: Imported inference module with load_model, run_autoregressive_forecast,
            postprocess_forecasts, and load_ground_truth_batch.

    Example:
        >>> mod = get_inference_module('fno')
        >>> model = mod.load_model(config, model_path, device)
    """
    return importlib.import_module(_MODULE_MAP[model_type])


# ---------------------------------------------------------------------------
# RMSE accumulation
# ---------------------------------------------------------------------------

def accumulate_rmse(
    config,
    model,
    device,
    primary_data: tuple,
    extended_data,
    extra_num_days: int,
    run_forecast_fn=None,
    postprocess_fn=None,
    load_gt_fn=None,
) -> dict:
    """Run rollouts over 2020-2025 and collect per-IC RMSE at each lead day.

    Args:
        config: configmypy configuration object.
        model (torch.nn.Module): AFNO RT model in eval mode.
        device (torch.device): Compute device.
        primary_data (tuple): (ocean_ds, atm_ds) for the primary 1993-2020 dataset.
        extended_data (tuple | None): (ocean_ds, atm_ds) for the 2021-2025 dataset,
            or None to skip the extended period.
        extra_num_days (int): Number of IC indices to scan in the extended period.
        run_forecast_fn (callable): run_autoregressive_forecast from the model's module.
        postprocess_fn (callable): postprocess_forecasts from the model's module.
        load_gt_fn (callable): load_ground_truth_batch from the model's module.

    Returns:
        dict: ``rmse[var][lead_day]`` = 1-D np.ndarray of RMSE values, one per
            valid IC date.

    Example:
        >>> result = accumulate_rmse(config, model, device, (o20, a20), (oext, aext), 1711)
        >>> result['thetao'][9].shape   # (n_valid_ics,)
    """
    import inference.run_metrics as _default_mod
    if run_forecast_fn is None:
        run_forecast_fn = _default_mod.run_autoregressive_forecast
    if postprocess_fn is None:
        postprocess_fn  = _default_mod.postprocess_forecasts
    if load_gt_fn is None:
        load_gt_fn      = _default_mod.load_ground_truth_batch

    ocean_2020, atm_2020 = primary_data

    # Build IC list
    ic_list = []
    start_2020 = date_to_day_index('01-01-2020', PRIMARY_REF)
    end_2020   = date_to_day_index('31-12-2020', PRIMARY_REF)
    for idx in range(start_2020, end_2020 - max(LEAD_DAYS) + 1):
        d = datetime(1993, 1, 1) + timedelta(days=idx)
        ic_list.append((d, idx, ocean_2020, atm_2020))

    if extended_data is not None:
        ocean_ext, atm_ext = extended_data
        for rel_idx in range(extra_num_days):
            d = datetime(2021, 1, 1) + timedelta(days=rel_idx)
            ic_list.append((d, rel_idx, ocean_ext, atm_ext))

    tqdm.write(f'Total IC dates: {len(ic_list)}')

    # Accumulators: lists that we convert to arrays at the end
    rmse_lists: dict[str, dict[int, list]] = {
        var: {ld: [] for ld in LEAD_DAYS} for var in VAR_ORDER
    }

    skipped = 0
    for d, idx, o_ds, a_ds in tqdm(ic_list, desc='RMSE per IC', unit='IC', file=sys.stdout):
        try:
            forecast_dict, mean_dict = run_forecast_fn(
                config, model, idx, max(LEAD_DAYS), device, o_ds, a_ds)
        except ValueError:
            tqdm.write(f'  Skip {d.date()}: corrupted input')
            skipped += 1
            continue

        preds  = postprocess_fn(forecast_dict, mean_dict, config)
        truths = load_gt_fn(config, idx, max(LEAD_DAYS), o_ds)

        for ld in LEAD_DAYS:
            for var in VAR_ORDER:
                pred_field  = preds[ld][var]
                truth_field = truths[ld][var]
                valid = ~np.isnan(truth_field) & ~np.isnan(pred_field)
                if valid.sum() == 0:
                    continue
                rmse = float(np.sqrt(np.mean(
                    (pred_field[valid] - truth_field[valid]) ** 2)))
                rmse_lists[var][ld].append(rmse)

    tqdm.write(f'\nDone. Skipped {skipped} IC dates (corrupted).')
    rmse = {var: {ld: np.array(rmse_lists[var][ld]) for ld in LEAD_DAYS}
            for var in VAR_ORDER}
    return rmse


def save_rmse(rmse: dict, out_path: Path) -> None:
    """Save per-IC RMSE arrays to a compressed .npz file.

    Args:
        rmse (dict): ``rmse[var][lead_day]`` = 1-D array of RMSE values.
        out_path (Path): Destination .npz file path.

    Returns:
        None

    Example:
        >>> save_rmse(rmse, Path('results/.../rmse_histograms.npz'))
    """
    flat = {}
    for var in VAR_ORDER:
        for ld in LEAD_DAYS:
            flat[f'rmse_{var}_{ld}'] = rmse[var][ld]
    np.savez_compressed(out_path, **flat)
    tqdm.write(f'Saved RMSE data: {out_path}')


def load_rmse(npz_path: Path) -> dict:
    """Load per-IC RMSE arrays from a previously saved .npz file.

    Args:
        npz_path (Path): Path to .npz file written by save_rmse.

    Returns:
        dict: ``rmse[var][lead_day]`` = 1-D np.ndarray of RMSE values.

    Example:
        >>> rmse = load_rmse(Path('results/.../rmse_histograms.npz'))
        >>> rmse['thetao'][9].mean()
    """
    data = np.load(npz_path)
    return {var: {ld: data[f'rmse_{var}_{ld}'] for ld in LEAD_DAYS}
            for var in VAR_ORDER}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _style_ax(ax: plt.Axes) -> None:
    """Apply shared axis styling: box frame, subtle grid, muted tick colours.

    Args:
        ax (plt.Axes): Axes object to style in-place.

    Returns:
        None

    Example:
        >>> _style_ax(ax)
    """
    ax.set_facecolor('#fcfcfb')
    for spine in ax.spines.values():
        spine.set_color('#c3c2b7')
        spine.set_linewidth(0.8)
    ax.tick_params(colors='#52514e', labelsize=8, length=3)
    ax.grid(True, color='#e1e0d9', linewidth=0.6, linestyle='-', zorder=0)
    ax.set_axisbelow(True)


def plot_rmse_histograms(rmse: dict, fig_stem: str, save_pdf: bool = False,
                         model_label: str = 'AFNO RT') -> None:
    """Render a 2×5 per-IC RMSE distribution figure and save to disk.

    Row 1 — probability density of per-IC RMSE values at each lead day.
    Row 2 — CDF: what percentage of IC dates achieve RMSE below a threshold.

    Bin edges for each variable are derived from the data (0 to the 99th
    percentile of the +9d distribution, the widest) so the x-axis is always
    well-utilised without hard-coding physical units.

    Args:
        rmse (dict): Output of accumulate_rmse or load_rmse.
        fig_stem (str): File path without extension; PNG always written, PDF optional.
        save_pdf (bool): If True, also write a PDF.

    Returns:
        None

    Example:
        >>> plot_rmse_histograms(rmse, 'results/figures/fig_rmse_histograms', save_pdf=True)
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharex='col')
    fig.patch.set_facecolor('#fcfcfb')

    cdf_ref_pcts = [25, 50, 75]

    for col, var in enumerate(VAR_ORDER):
        meta = VAR_META[var]
        ax_pdf = axes[0, col]
        ax_cdf = axes[1, col]

        # Derive x-range from data: 0 → 99th-pct of the +9d (widest) distribution
        all_vals = np.concatenate([rmse[var][ld] for ld in LEAD_DAYS])
        x_max = float(np.percentile(all_vals, 99.5))
        bin_edges = np.linspace(0.0, x_max, meta['n_bins'] + 1)
        bin_width = bin_edges[1] - bin_edges[0]
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        for ld in LEAD_DAYS:
            vals  = rmse[var][ld]
            total = len(vals)
            c, _  = np.histogram(vals, bins=bin_edges)
            c     = c.astype(float)

            # density
            density = c / (total * bin_width) if total > 0 else c
            ax_pdf.step(centers, density, where='mid',
                        color=LEAD_COLORS[ld], linewidth=1.4,
                        label=LEAD_LABELS[ld], alpha=0.92)

            # CDF (%)
            cdf = np.cumsum(c) / total * 100 if total > 0 else np.zeros_like(c)
            ax_cdf.step(centers, cdf, where='mid',
                        color=LEAD_COLORS[ld], linewidth=1.4,
                        label=LEAD_LABELS[ld], alpha=0.92)

        # ---- PDF styling ----
        _style_ax(ax_pdf)
        ax_pdf.set_xlim(0, x_max)
        ax_pdf.set_title(meta['label'], fontsize=10, fontweight='bold',
                         color='#0b0b0b', pad=6)
        if col == 0:
            ax_pdf.set_ylabel('Density', fontsize=9, color='#52514e')
        ax_pdf.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

        # ---- CDF styling ----
        _style_ax(ax_cdf)
        ax_cdf.set_xlim(0, x_max)
        ax_cdf.set_ylim(0, 100)
        ax_cdf.set_xlabel(f"RMSE ({meta['unit']})", fontsize=9, color='#52514e')
        for pct in cdf_ref_pcts:
            ax_cdf.axhline(pct, color='#898781', linewidth=0.6, linestyle=':', zorder=1)
        ax_cdf.yaxis.set_major_locator(mticker.MultipleLocator(25))
        ax_cdf.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        if col == 0:
            ax_cdf.set_ylabel('Cumulative %', fontsize=9, color='#52514e')

    # Row labels
    axes[0, 0].annotate('PDF', xy=(-0.22, 0.5), xycoords='axes fraction',
                         fontsize=9, color='#52514e', rotation=90,
                         va='center', ha='right')
    axes[1, 0].annotate('CDF', xy=(-0.22, 0.5), xycoords='axes fraction',
                         fontsize=9, color='#52514e', rotation=90,
                         va='center', ha='right')

    # Shared legend on top-right panel
    handles, labels = axes[0, -1].get_legend_handles_labels()
    axes[0, -1].legend(handles, labels, fontsize=8, frameon=False,
                       loc='upper right', handlelength=1.2,
                       labelcolor='#0b0b0b')

    fig.suptitle(f'Per-IC RMSE Distributions — {model_label} (2020-2025)',
                 fontsize=11, y=1.01, color='#0b0b0b', fontweight='bold')
    fig.tight_layout(h_pad=2.5)

    png_path = f'{fig_stem}.png'
    fig.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='#fcfcfb')
    print(f'Saved: {png_path}')

    if save_pdf:
        pdf_path = f'{fig_stem}.pdf'
        fig.savefig(pdf_path, bbox_inches='tight', facecolor='#fcfcfb')
        print(f'Saved: {pdf_path}')

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the per-IC RMSE histogram pipeline.

    Returns:
        argparse.Namespace: Parsed arguments.

    Example:
        >>> args = parse_args()
    """
    p = argparse.ArgumentParser(
        description='Compute and plot per-IC RMSE distributions for any publication model')
    p.add_argument('--model_type',     default='afno',
                   choices=['afno', 'fno', 'tfno', 'uno'],
                   help='Model architecture (determines which inference module to use)')
    p.add_argument('--model_name',     default=None,
                   help='Label for figure title and log filename '
                        '(default: derived from --model_path stem)')
    p.add_argument('--config_file',    default='afno_bob_surf_e14.yaml')
    p.add_argument('--model_path',     default='results/models/AFNO_BoB_Surf_E14.pth')
    p.add_argument('--data_dir',       default=None,
                   help='Primary dataset dir (default: from config)')
    p.add_argument('--extra_data_dir', default='data/2021_2025')
    p.add_argument('--extra_num_days', type=int, default=1711)
    p.add_argument('--output_dir',     default=None,
                   help='Directory for rmse_histograms.npz '
                        '(default: results/<model_name>/)')
    p.add_argument('--output_fig',     default=None,
                   help='Figure stem without extension '
                        '(default: results/figures/fig_rmse_histograms_<model_name>)')
    p.add_argument('--device',         default=None)
    p.add_argument('--skip_compute',   action='store_true',
                   help='Skip rollout; load existing .npz and re-plot only')
    p.add_argument('--pdf',            action='store_true')
    return p.parse_args()


def main() -> None:
    """Entry point: run rollout loop, collect per-IC RMSEs, and plot.

    Example:
        >>> # python src/analysis/compute_rmse_histograms.py --device cuda:1 --pdf
    """
    args = parse_args()

    model_name = args.model_name or Path(args.model_path).stem
    output_dir = Path(args.output_dir or f'results/{model_name}')
    output_fig = args.output_fig or f'results/figures/fig_rmse_histograms_{model_name}'

    log_path = Path('logs') / f'compute_rmse_histograms_{model_name}.log'
    log_path.parent.mkdir(exist_ok=True)
    tee = TeeLogger(str(log_path))
    sys.stdout = tee
    sys.stderr = tee

    try:
        print(f'Model type  : {args.model_type}')
        print(f'Model name  : {model_name}')
        print(f'Config      : {args.config_file}')
        print(f'Model       : {args.model_path}')
        print(f'Output dir  : {output_dir}')
        print(f'Output fig  : {output_fig}')

        output_dir.mkdir(parents=True, exist_ok=True)
        npz_path = output_dir / 'rmse_histograms.npz'
        Path(output_fig).parent.mkdir(parents=True, exist_ok=True)

        mod = get_inference_module(args.model_type)

        if args.skip_compute and npz_path.exists():
            print(f'--skip_compute: loading existing {npz_path}')
            rmse = load_rmse(npz_path)
        else:
            pipe   = ConfigPipeline([YamlConfig(args.config_file, config_name='default',
                                                config_folder='config/')])
            config = pipe.read_conf()

            device_str = args.device or config.device
            device = torch.device(
                device_str if (torch.cuda.is_available() or 'cpu' in device_str) else 'cpu')
            print(f'Device      : {device}')

            model = mod.load_model(config, args.model_path, device)

            data_dir = Path(args.data_dir or config.data.data_dir)
            print(f'Primary data: {data_dir}')
            ocean_2020 = xr.open_dataset(data_dir / f'{config.data.file_prefix}.nc')
            atm_2020   = xr.open_dataset(data_dir / f'{config.data.file_prefix_atm}.nc')

            extended_data = None
            if args.extra_data_dir:
                ext_dir = Path(args.extra_data_dir)
                print(f'Extended    : {ext_dir}  ({args.extra_num_days} IC days)')
                ocean_ext = xr.open_dataset(ext_dir / 'ocean_2021_2025.nc')
                atm_ext   = xr.open_dataset(ext_dir / 'atm_2021_2025.nc')
                extended_data = (ocean_ext, atm_ext)

            rmse = accumulate_rmse(
                config, model, device,
                (ocean_2020, atm_2020), extended_data, args.extra_num_days,
                run_forecast_fn=mod.run_autoregressive_forecast,
                postprocess_fn=mod.postprocess_forecasts,
                load_gt_fn=mod.load_ground_truth_batch)
            save_rmse(rmse, npz_path)

        # Print summary statistics
        print('\nSummary — median RMSE per variable at each lead day:')
        hdr = f"  {'Variable':<15}" + ''.join(f"  {f'+{ld}d':>8}" for ld in LEAD_DAYS)
        print(hdr)
        for var in VAR_ORDER:
            row = f"  {VAR_META[var]['label']:<15}"
            for ld in LEAD_DAYS:
                row += f"  {np.median(rmse[var][ld]):>8.4f}"
            print(row)

        plot_rmse_histograms(rmse, output_fig, save_pdf=args.pdf,
                             model_label=model_name.replace('_', ' '))

    except Exception as exc:
        print(f'\nERROR: {exc}')
        raise
    finally:
        cleanup_logging(tee)


if __name__ == '__main__':
    main()

"""Compute and plot forecast error distributions at lead days +1, +3, +5, +7, +9.

For every valid IC date in 2020-2025, runs AFNO RT (E14) autoregressive rollout
and accumulates per-pixel forecast errors (pred - truth) into histogram bins for
each output variable at the five requested lead days.  Histogram counts are written
to an .npz file so the figure can be regenerated without re-running the model.  The
final figure is a 1×5 multi-panel plot with overlaid lead-day distributions using a
validated ordinal single-blue ramp (light→dark).

Inputs:
    --config_file (str)    : YAML config in config/ (default: afno_bob_surf_e14.yaml).
    --model_path (str)     : Path to .pth weights (default: results/models/AFNO_BoB_Surf_E14.pth).
    --data_dir (str)       : Primary dataset directory (default: from config).
    --extra_data_dir (str) : 2021-2025 dataset directory
                             (default: data/2021_2025).
    --extra_num_days (int) : IC dates to evaluate in extended period (default: 1711).
    --output_dir (str)     : Directory for error_histograms.npz (default:
                             results/AFNO_BoB_Surf_E14/).
    --output_fig (str)     : Stem path for figure outputs, without extension
                             (default: results/figures/fig_error_histograms).
    --device (str)         : PyTorch device (default: from config).
    --skip_compute         : If set, load existing .npz and re-plot only.
    --pdf                  : Save an additional PDF alongside the PNG.

Outputs:
    <output_dir>/error_histograms.npz  — histogram counts + bin edges per variable/lead.
    <output_fig>.png                   — multi-panel error distribution figure.
    <output_fig>.pdf                   — PDF version (if --pdf is set).
    logs/compute_error_histograms.log  — verbatim console log.

Example:
    conda activate BoB_Surf_2
    python src/analysis/compute_error_histograms.py \\
        --config_file afno_bob_surf_e14.yaml \\
        --model_path results/models/AFNO_BoB_Surf_E14.pth \\
        --extra_data_dir data/2021_2025 \\
        --device cuda:1 --pdf
"""

import argparse
import importlib
import sys
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
# Constants
# ---------------------------------------------------------------------------

VAR_ORDER = ['thetao', 'so', 'zos', 'uo', 'vo']

VAR_META = {
    'thetao': dict(label='SST', unit='°C',   xlim=(-2.5, 2.5),  n_bins=200),
    'so':     dict(label='SSS', unit='psu',  xlim=(-5.0, 5.0),  n_bins=200),
    'zos':    dict(label='SSH', unit='m',    xlim=(-0.4, 0.4),  n_bins=200),
    'uo':     dict(label='Zonal current',      unit='m/s', xlim=(-1.0, 1.0), n_bins=200),
    'vo':     dict(label='Meridional current', unit='m/s', xlim=(-1.0, 1.0), n_bins=200),
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
# Histogram accumulation
# ---------------------------------------------------------------------------

def make_bin_edges(var: str) -> np.ndarray:
    """Build histogram bin edges for a variable from its VAR_META entry.

    Args:
        var (str): Variable name (key in VAR_ORDER).

    Returns:
        np.ndarray: 1-D array of (n_bins+1) bin-edge values.

    Example:
        >>> edges = make_bin_edges('thetao')
        >>> edges.shape
        (201,)
    """
    m = VAR_META[var]
    return np.linspace(m['xlim'][0], m['xlim'][1], m['n_bins'] + 1)


def accumulate_errors(
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
    """Run rollouts over 2020-2025 and accumulate per-pixel errors into histograms.

    Args:
        config: configmypy configuration object.
        model (torch.nn.Module): AFNO RT model in eval mode.
        device (torch.device): Compute device.
        primary_data (tuple): (ocean_ds, atm_ds) for the primary 1993-2020 dataset.
        extended_data (tuple | None): (ocean_ds, atm_ds) for the 2021-2025 dataset,
            or None to skip the extended period.
        extra_num_days (int): Number of IC indices to scan in the extended period.
        run_forecast_fn (callable): run_autoregressive_forecast from the model's
            inference module (default: from inference.run_metrics).
        postprocess_fn (callable): postprocess_forecasts from the model's module.
        load_gt_fn (callable): load_ground_truth_batch from the model's module.

    Returns:
        dict: Nested dict ``counts[var][lead_day]`` = np.ndarray of histogram counts
            (length n_bins) and ``edges[var]`` = np.ndarray of bin edges.

    Example:
        >>> result = accumulate_errors(config, model, device, (o20, a20), (oext, aext), 1711)
        >>> result['counts']['thetao'][9].sum()  # total pixels accumulated at +9d
        ...
    """
    import inference.run_metrics as _default_mod
    if run_forecast_fn is None:
        run_forecast_fn = _default_mod.run_autoregressive_forecast
    if postprocess_fn is None:
        postprocess_fn  = _default_mod.postprocess_forecasts
    if load_gt_fn is None:
        load_gt_fn      = _default_mod.load_ground_truth_batch

    ocean_2020, atm_2020 = primary_data

    # Build IC list: (date, day_idx, ocean_ds, atm_ds)
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

    # Initialise histogram accumulators
    bin_edges = {var: make_bin_edges(var) for var in VAR_ORDER}
    counts: dict[str, dict[int, np.ndarray]] = {
        var: {ld: np.zeros(VAR_META[var]['n_bins'], dtype=np.int64) for ld in LEAD_DAYS}
        for var in VAR_ORDER
    }

    skipped = 0
    for d, idx, o_ds, a_ds in tqdm(ic_list, desc='Error histograms', unit='IC', file=sys.stdout):
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
                errors = pred_field[valid] - truth_field[valid]
                hist, _ = np.histogram(errors, bins=bin_edges[var])
                counts[var][ld] += hist

    tqdm.write(f'\nDone. Skipped {skipped} IC dates (corrupted).')
    return {'counts': counts, 'edges': bin_edges}


def save_histograms(result: dict, out_path: Path) -> None:
    """Save histogram counts and edges to a compressed .npz file.

    Args:
        result (dict): Output of accumulate_errors — contains 'counts' and 'edges'.
        out_path (Path): Destination .npz file path.

    Returns:
        None

    Example:
        >>> save_histograms(result, Path('results/.../error_histograms.npz'))
    """
    flat = {}
    for var in VAR_ORDER:
        flat[f'edges_{var}'] = result['edges'][var]
        for ld in LEAD_DAYS:
            flat[f'counts_{var}_{ld}'] = result['counts'][var][ld]
    np.savez_compressed(out_path, **flat)
    tqdm.write(f'Saved histogram data: {out_path}')


def load_histograms(npz_path: Path) -> dict:
    """Load histogram counts and edges from a previously saved .npz file.

    Args:
        npz_path (Path): Path to the .npz file written by save_histograms.

    Returns:
        dict: Same structure as accumulate_errors output.

    Example:
        >>> result = load_histograms(Path('results/.../error_histograms.npz'))
        >>> result['counts']['thetao'][9].sum()
    """
    data = np.load(npz_path)
    edges   = {var: data[f'edges_{var}'] for var in VAR_ORDER}
    counts  = {
        var: {ld: data[f'counts_{var}_{ld}'] for ld in LEAD_DAYS}
        for var in VAR_ORDER
    }
    return {'counts': counts, 'edges': edges}


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
    surf = '#fcfcfb'
    spine_col = '#c3c2b7'
    grid_col  = '#e1e0d9'

    ax.set_facecolor(surf)
    for spine in ax.spines.values():
        spine.set_color(spine_col)
        spine.set_linewidth(0.8)
    ax.tick_params(colors='#52514e', labelsize=10, length=3)
    ax.grid(True, color=grid_col, linewidth=0.6, linestyle='-', zorder=0)
    ax.set_axisbelow(True)


def plot_histograms(result: dict, fig_stem: str, save_pdf: bool = False,
                    model_label: str = 'AFNO RT') -> None:
    """Render a 2×5 multi-panel error-distribution figure and save to disk.

    Row 1 — probability density: overlaid step-plot histograms per lead day.
    Row 2 — cumulative distribution (CDF): shows what percentage of IC × pixel
    samples have an error smaller than a given threshold.  Both rows use a
    validated ordinal single-blue ramp (light→dark, +1d→+9d).  All axes are
    framed as boxes with hairline grid lines.

    Args:
        result (dict): Output of accumulate_errors or load_histograms.
        fig_stem (str): File path without extension; PNG (always) and PDF (optional)
            are written there.
        save_pdf (bool): If True, also write a PDF.

    Returns:
        None

    Example:
        >>> plot_histograms(result, 'results/figures/fig_error_histograms', save_pdf=True)
    """
    counts = result['counts']
    edges  = result['edges']

    fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharex='col')
    fig.patch.set_facecolor('#fcfcfb')

    cdf_ref_pcts = [25, 50, 75]   # reference lines on CDF panels

    for col, var in enumerate(VAR_ORDER):
        meta     = VAR_META[var]
        edges_v  = edges[var]
        bin_width = edges_v[1] - edges_v[0]
        centers  = 0.5 * (edges_v[:-1] + edges_v[1:])

        ax_pdf = axes[0, col]
        ax_cdf = axes[1, col]

        for ld in LEAD_DAYS:
            c     = counts[var][ld].astype(float)
            total = c.sum()

            # ---- Row 1: density ----
            density = c / (total * bin_width) if total > 0 else c
            ax_pdf.step(centers, density, where='mid',
                        color=LEAD_COLORS[ld], linewidth=1.4,
                        label=LEAD_LABELS[ld], alpha=0.92)

            # ---- Row 2: CDF (%) ----
            cdf = np.cumsum(c) / total * 100 if total > 0 else np.zeros_like(c)
            ax_cdf.step(centers, cdf, where='mid',
                        color=LEAD_COLORS[ld], linewidth=1.4,
                        label=LEAD_LABELS[ld], alpha=0.92)

        # ---- PDF styling ----
        _style_ax(ax_pdf)
        ax_pdf.set_xlim(meta['xlim'])
        ax_pdf.axvline(0, color='#898781', linewidth=0.9, linestyle='--', zorder=1)
        ax_pdf.set_title(meta['label'], fontsize=13, fontweight='bold',
                         color='#0b0b0b', pad=6)
        if col == 0:
            ax_pdf.set_ylabel('Density', fontsize=12, color='#52514e')
        ax_pdf.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

        # ---- CDF styling ----
        _style_ax(ax_cdf)
        ax_cdf.set_xlim(meta['xlim'])
        ax_cdf.set_ylim(0, 100)
        ax_cdf.set_xlabel(f"Error ({meta['unit']})", fontsize=12, color='#52514e')
        ax_cdf.axvline(0, color='#898781', linewidth=0.9, linestyle='--', zorder=1)
        for pct in cdf_ref_pcts:
            ax_cdf.axhline(pct, color='#898781', linewidth=0.6,
                           linestyle=':', zorder=1)
        ax_cdf.yaxis.set_major_locator(mticker.MultipleLocator(25))
        ax_cdf.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        if col == 0:
            ax_cdf.set_ylabel('Cumulative %', fontsize=12, color='#52514e')

    # Row labels (left margin)
    axes[0, 0].annotate('PDF', xy=(-0.22, 0.5), xycoords='axes fraction',
                         fontsize=12, color='#52514e', rotation=90, va='center',
                         ha='right')
    axes[1, 0].annotate('CDF', xy=(-0.22, 0.5), xycoords='axes fraction',
                         fontsize=12, color='#52514e', rotation=90, va='center',
                         ha='right')

    # Shared legend on top-right panel
    handles, labels = axes[0, -1].get_legend_handles_labels()
    axes[0, -1].legend(handles, labels, fontsize=10, frameon=False,
                       loc='upper right', handlelength=1.2,
                       labelcolor='#0b0b0b')

    fig.suptitle(f'Forecast Error Distributions — {model_label} (2020-2025)',
                 fontsize=14, y=1.01, color='#0b0b0b', fontweight='bold')

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
    """Parse command-line arguments for the error histogram pipeline.

    Returns:
        argparse.Namespace: Parsed arguments.

    Example:
        >>> args = parse_args()
    """
    p = argparse.ArgumentParser(
        description='Compute and plot forecast error distributions for any publication model')
    p.add_argument('--model_type',     default='afno',
                   choices=['afno', 'fno', 'tfno', 'uno'],
                   help='Model architecture (determines which inference module to use)')
    p.add_argument('--model_name',     default=None,
                   help='Label used in figure title and log filename '
                        '(default: derived from --model_path stem)')
    p.add_argument('--model_label',    default=None,
                   help='Display name shown in the figure suptitle '
                        '(default: model_name with underscores replaced by spaces)')
    p.add_argument('--config_file',    default='afno_bob_surf_e14.yaml')
    p.add_argument('--model_path',     default='results/models/AFNO_BoB_Surf_E14.pth')
    p.add_argument('--data_dir',       default=None,
                   help='Primary dataset dir (default: from config)')
    p.add_argument('--extra_data_dir', default='data/2021_2025')
    p.add_argument('--extra_num_days', type=int, default=1711)
    p.add_argument('--output_dir',     default=None,
                   help='Directory for error_histograms.npz '
                        '(default: results/<model_name>/)')
    p.add_argument('--output_fig',     default=None,
                   help='Figure stem without extension '
                        '(default: results/figures/fig_error_histograms_<model_name>)')
    p.add_argument('--device',         default=None)
    p.add_argument('--skip_compute',   action='store_true',
                   help='Skip rollout; load existing .npz and re-plot only')
    p.add_argument('--pdf',            action='store_true')
    return p.parse_args()


def main() -> None:
    """Entry point: run rollout loop, accumulate histogram counts, and plot.

    Example:
        >>> # python src/analysis/compute_error_histograms.py --device cuda:1 --pdf
    """
    args = parse_args()

    model_name = args.model_name or Path(args.model_path).stem
    output_dir = Path(args.output_dir or f'results/{model_name}')
    output_fig = args.output_fig or f'results/figures/fig_error_histograms_{model_name}'

    log_path = Path('logs') / f'compute_error_histograms_{model_name}.log'
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
        npz_path = output_dir / 'error_histograms.npz'
        Path(output_fig).parent.mkdir(parents=True, exist_ok=True)

        mod = get_inference_module(args.model_type)

        if args.skip_compute and npz_path.exists():
            print(f'--skip_compute: loading existing {npz_path}')
            result = load_histograms(npz_path)
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

            result = accumulate_errors(
                config, model, device,
                (ocean_2020, atm_2020), extended_data, args.extra_num_days,
                run_forecast_fn=mod.run_autoregressive_forecast,
                postprocess_fn=mod.postprocess_forecasts,
                load_gt_fn=mod.load_ground_truth_batch)
            save_histograms(result, npz_path)

        display_label = args.model_label or model_name.replace('_', ' ')
        plot_histograms(result, output_fig, save_pdf=args.pdf,
                        model_label=display_label)

    except Exception as exc:
        print(f'\nERROR: {exc}')
        raise
    finally:
        cleanup_logging(tee)


if __name__ == '__main__':
    main()

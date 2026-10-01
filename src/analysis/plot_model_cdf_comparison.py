"""Compare per-IC RMSE CDFs across all publication models.

Loads pre-computed rmse_histograms.npz for each model and plots a 5×5 grid:
rows = lead days (+1, +3, +5, +7, +9), columns = variables (SST, SSS, SSH, U, V).
Each panel overlays the empirical CDF of per-IC RMSE for every model, using the
categorical palette (one distinct hue per model).

Empirical CDFs are derived directly from the sorted raw RMSE arrays (no binning),
giving exact percentile curves from ~2,059 values per model.

Inputs:
    --models (str, repeatable): One or more "Label:results_dir" pairs.  Each
        results_dir must contain rmse_histograms.npz produced by
        compute_rmse_histograms.py.
    --output_fig (str): Figure stem without extension
        (default: results/figures/fig_model_cdf_comparison).
    --pdf: Also save a PDF alongside the PNG.

Outputs:
    <output_fig>.png  — 5×5 CDF comparison figure.
    <output_fig>.pdf  — PDF version (if --pdf is set).

Example:
    conda activate BoB_Surf_2
    python src/analysis/plot_model_cdf_comparison.py \\
        --models "AFNO 1T:results/AFNO_BoB_Surf_E13" \\
                 "AFNO RT:results/AFNO_BoB_Surf_E14" \\
                 "FNO 1T:results/FNO_BoB_Surf_E03" \\
                 "FNO RT:results/FNO_BoB_Surf_E04" \\
                 "TFNO 1T:results/TFNO_BoB_Surf_E03" \\
                 "TFNO RT:results/TFNO_BoB_Surf_E04" \\
                 "UNO RT:results/UNO_BoB_Surf_E04" \\
        --output_fig results/figures/fig_model_cdf_comparison --pdf
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VAR_ORDER = ['thetao', 'so', 'zos', 'uo', 'vo']

VAR_META = {
    'thetao': dict(label='SST',       unit='°C'),
    'so':     dict(label='SSS',       unit='psu'),
    'zos':    dict(label='SSH',       unit='m'),
    'uo':     dict(label='U current', unit='m/s'),
    'vo':     dict(label='V current', unit='m/s'),
}

LEAD_DAYS = [1, 3, 5, 7, 9]

# Model styles — mirrors fig1 MODEL_STYLES_MARKER:
# color encodes architecture family; ^ marker encodes RT training regime.
_C = {
    'Persistence': '#888888',
    'FNO':         '#4363d8',
    'UNO':         '#3cb44b',
    'TFNO':        '#f58231',
    'AFNO':        '#9467bd',
}

MODEL_STYLES = {
    'Persistence': dict(color=_C['Persistence'], ls='--', lw=1.4, marker='',  ms=0),
    'FNO 1T':      dict(color=_C['FNO'],         ls='-',  lw=1.4, marker='',  ms=0),
    'FNO RT':      dict(color=_C['FNO'],         ls='-',  lw=1.4, marker='^', ms=6),
    'UNO 1T':      dict(color=_C['UNO'],         ls='-',  lw=1.4, marker='',  ms=0),
    'UNO RT':      dict(color=_C['UNO'],         ls='-',  lw=1.4, marker='^', ms=6),
    'TFNO 1T':     dict(color=_C['TFNO'],        ls='-',  lw=1.4, marker='',  ms=0),
    'TFNO RT':     dict(color=_C['TFNO'],        ls='-',  lw=1.4, marker='^', ms=6),
    'AFNO 1T':     dict(color=_C['AFNO'],        ls='-',  lw=1.6, marker='',  ms=0),
    'AFNO RT':     dict(color=_C['AFNO'],        ls='-',  lw=2.2, marker='^', ms=6),
}
_DEFAULT_STYLE = dict(color='#aaaaaa', ls='-', lw=1.2, marker='', ms=0)

CDF_REF_PCTS = [25, 50, 75]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_model_rmse(results_dir: Path) -> dict | None:
    """Load per-IC RMSE arrays from a model's rmse_histograms.npz.

    Args:
        results_dir (Path): Directory containing rmse_histograms.npz.

    Returns:
        dict | None: ``rmse[var][lead_day]`` = 1-D np.ndarray, or None if the
            file does not exist.

    Example:
        >>> rmse = load_model_rmse(Path('results/AFNO_BoB_Surf_E14'))
        >>> rmse['thetao'][9].shape
        (2059,)
    """
    npz_path = results_dir / 'rmse_histograms.npz'
    if not npz_path.exists():
        print(f'  WARNING: not found — {npz_path}', file=sys.stderr)
        return None
    data = np.load(npz_path)
    return {var: {ld: data[f'rmse_{var}_{ld}'] for ld in LEAD_DAYS}
            for var in VAR_ORDER}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _style_ax(ax: plt.Axes) -> None:
    """Apply shared axis styling: box frame, subtle grid, muted ticks.

    Args:
        ax (plt.Axes): Axes to style in-place.

    Returns:
        None

    Example:
        >>> _style_ax(ax)
    """
    ax.set_facecolor('#fcfcfb')
    for spine in ax.spines.values():
        spine.set_color('#c3c2b7')
        spine.set_linewidth(0.8)
    ax.tick_params(colors='#52514e', labelsize=9, length=3)
    ax.grid(True, color='#e1e0d9', linewidth=0.5, linestyle='-', zorder=0)
    ax.set_axisbelow(True)


def plot_cdf_comparison(
    models: list[tuple[str, dict]],
    output_fig: str,
    save_pdf: bool = False,
) -> None:
    """Render a 5×5 model CDF comparison figure.

    Layout: rows = lead days (+1/+3/+5/+7/+9), columns = variables.
    Each panel overlays empirical CDFs for all models.  X-axis is shared within
    each column so the RMSE scale is consistent across lead days.

    Args:
        models (list[tuple[str, dict]]): List of (label, rmse_dict) pairs, one
            per model.  rmse_dict has the same structure as load_model_rmse output.
        output_fig (str): File stem (no extension); PNG always written, PDF optional.
        save_pdf (bool): Also write a PDF.

    Returns:
        None

    Example:
        >>> plot_cdf_comparison(model_list, 'results/figures/fig_model_cdf_comparison')
    """
    n_models = len(models)

    fig, axes = plt.subplots(
        len(LEAD_DAYS), len(VAR_ORDER),
        figsize=(20, 14),
        sharex='col', sharey=True,
    )
    fig.patch.set_facecolor('#fcfcfb')

    # Per-column x-limit: 99.5th pct of +9d RMSE across all models
    col_xlim = {}
    for var in VAR_ORDER:
        all_vals = np.concatenate([rmse[var][9] for _, rmse in models
                                   if rmse is not None and len(rmse[var][9]) > 0])
        col_xlim[var] = float(np.percentile(all_vals, 99.5))

    for row, ld in enumerate(LEAD_DAYS):
        for col, var in enumerate(VAR_ORDER):
            ax = axes[row, col]
            _style_ax(ax)

            for i, (label, rmse) in enumerate(models):
                if rmse is None:
                    continue
                vals = rmse[var][ld]
                if len(vals) == 0:
                    continue
                sty = MODEL_STYLES.get(label, _DEFAULT_STYLE)
                vals_sorted = np.sort(vals)
                cdf_pct = np.arange(1, len(vals_sorted) + 1) / len(vals_sorted) * 100
                # Stagger marker start per model so ^ symbols don't overlap
                markevery = (i * 30, 220) if sty['marker'] else None
                ax.plot(vals_sorted, cdf_pct,
                        color=sty['color'], ls=sty['ls'], lw=sty['lw'],
                        marker=sty['marker'] or None, ms=sty['ms'],
                        markevery=markevery,
                        alpha=0.92, label=label)

            # Reference lines
            for pct in CDF_REF_PCTS:
                ax.axhline(pct, color='#898781', linewidth=0.5,
                           linestyle=':', zorder=1)

            ax.set_xlim(0, col_xlim[var])
            ax.set_ylim(0, 100)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

            # Column header (top row only)
            if row == 0:
                ax.set_title(
                    f"{VAR_META[var]['label']} ({VAR_META[var]['unit']})",
                    fontsize=12, fontweight='bold', color='#0b0b0b', pad=6)

            # X-label (bottom row only)
            if row == len(LEAD_DAYS) - 1:
                ax.set_xlabel('RMSE', fontsize=10, color='#52514e')

            # Y-label (left column only)
            if col == 0:
                ax.set_ylabel('Cumulative %', fontsize=10, color='#52514e',
                              labelpad=4)

    # Bold row labels (lead-day) — placed to the left of each row's leftmost
    # axis, rotated bottom-to-top matching the Cumulative % y-axis orientation.
    for row, ld in enumerate(LEAD_DAYS):
        axes[row, 0].annotate(
            f'+{ld}d',
            xy=(-0.22, 0.5), xycoords='axes fraction',
            fontsize=13, fontweight='bold', color='#0b0b0b',
            rotation=90, va='center', ha='center',
            annotation_clip=False)

    # Shared legend — below the figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center', ncol=n_models,
               fontsize=12, frameon=False,
               bbox_to_anchor=(0.5, -0.02),
               handlelength=1.5, labelcolor='#0b0b0b')

    fig.suptitle('Per-IC RMSE CDF Comparison — All Models (2020-2025)',
                 fontsize=14, y=1.01, color='#0b0b0b', fontweight='bold')

    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    fig.subplots_adjust(left=0.11)

    png_path = f'{output_fig}.png'
    fig.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='#fcfcfb')
    print(f'Saved: {png_path}')

    if save_pdf:
        pdf_path = f'{output_fig}.pdf'
        fig.savefig(pdf_path, bbox_inches='tight', facecolor='#fcfcfb')
        print(f'Saved: {pdf_path}')

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the CDF comparison figure.

    Returns:
        argparse.Namespace: Parsed arguments.

    Example:
        >>> args = parse_args()
    """
    p = argparse.ArgumentParser(
        description='Plot per-IC RMSE CDF comparison across publication models')
    p.add_argument('--models', nargs='+', required=True,
                   metavar='LABEL:RESULTS_DIR',
                   help='One or more "Label:results_dir" pairs, each results_dir '
                        'must contain rmse_histograms.npz')
    p.add_argument('--output_fig', default='results/figures/fig_model_cdf_comparison',
                   help='Figure stem without extension')
    p.add_argument('--pdf', action='store_true')
    return p.parse_args()


def main() -> None:
    """Entry point: load model RMSE data and render CDF comparison figure.

    Example:
        >>> # python src/analysis/plot_model_cdf_comparison.py \\
        ... #     --models "AFNO 1T:results/AFNO_BoB_Surf_E13" \\
        ... #              "AFNO RT:results/AFNO_BoB_Surf_E14" \\
        ... #     --pdf
    """
    args = parse_args()

    if len(args.models) > len(MODEL_STYLES):
        print(f'ERROR: at most {len(MODEL_STYLES)} models supported '
              f'(got {len(args.models)})', file=sys.stderr)
        sys.exit(1)

    # Parse "Label:path" pairs
    models = []
    for entry in args.models:
        if ':' not in entry:
            print(f'ERROR: expected "Label:results_dir", got "{entry}"',
                  file=sys.stderr)
            sys.exit(1)
        label, results_dir = entry.split(':', 1)
        print(f'Loading {label:20s} from {results_dir}')
        rmse = load_model_rmse(Path(results_dir))
        models.append((label, rmse))

    available = [(lbl, r) for lbl, r in models if r is not None]
    if not available:
        print('ERROR: no valid model data found.', file=sys.stderr)
        sys.exit(1)

    Path(args.output_fig).parent.mkdir(parents=True, exist_ok=True)
    plot_cdf_comparison(available, args.output_fig, save_pdf=args.pdf)


if __name__ == '__main__':
    main()

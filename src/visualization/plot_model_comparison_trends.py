"""Plot RMSE and Pearson correlation trends over lead days 1-9 for multiple models.

Produces a 5-row × 2-column publication figure (8 inches wide).  Each row is one
ocean variable; left column shows RMSE (lower = better) and right column shows
Pearson correlation r (higher = better).  Seven models are overlaid per panel with
distinct colours and line styles.

Inputs:
    --models (str, repeatable): ``"Label:results_dir"`` pairs in legend order.
    --leads (int list): Lead days to plot (default 1–9).
    --output (str): Output figure path (default: results/figures/model_comparison_trends.pdf).
    --dpi (int): DPI for raster export (default 300).
    --width (float): Figure width in inches (default 8.0).

Outputs:
    <output>  PDF (and PNG at same path with .png extension).

Example:
    python src/visualization/plot_model_comparison_trends.py \\
        --models "Persistence:results/Persistence_Baseline" \\
                 "ConvLSTM:results/ConvLSTM_BoB_Surf_E01" \\
                 "FNO:results/FNO_BoB_Surf_E01" \\
                 "TFNO:results/TFNO_BoB_Surf_E01" \\
                 "UNO:results/UNO_BoB_Surf_E01" \\
                 "AFNO 1T:results/AFNO_BoB_Surf_E12p1" \\
                 "AFNO RT:results/AFNO_BoB_Surf_E11p1"
"""

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))


# ── variable metadata ────────────────────────────────────────────────────────

VAR_ORDER = ['thetao', 'so', 'zos', 'uo', 'vo']

VAR_META = {
    'thetao': dict(label='SST',               rmse_unit='°C',  r_ylim=(0.75, 1.01)),
    'so':     dict(label='SSS',               rmse_unit='psu', r_ylim=(0.75, 1.01)),
    'zos':    dict(label='SSH',               rmse_unit='m',   r_ylim=(0.75, 1.01)),
    'uo':     dict(label='Zonal current',     rmse_unit='m/s', r_ylim=(0.55, 1.01)),
    'vo':     dict(label='Meridional current',rmse_unit='m/s', r_ylim=(0.55, 1.01)),
}

METRICS_FILENAMES = [
    'forecast_metrics_combined.txt',
    'forecast_metrics_extended.txt',
    'forecast_metrics.txt',
]

# ── model style palettes ──────────────────────────────────────────────────────
# Color encodes architecture family; line style / marker encodes training regime.

_C = {
    'Persistence': '#888888',
    'FNO':         '#4363d8',
    'UNO':         '#3cb44b',
    'TFNO':        '#f58231',
    'AFNO':        '#9467bd',
}

# Variant A: solid = 1T, dashed = RT (same color per architecture)
MODEL_STYLES_LINESTYLE = {
    'Persistence': dict(color=_C['Persistence'], ls='--', lw=1.4, marker='',  ms=0),
    'FNO 1T':      dict(color=_C['FNO'],         ls='-',  lw=1.4, marker='',  ms=0),
    'FNO RT':      dict(color=_C['FNO'],         ls='--', lw=1.4, marker='',  ms=0),
    'UNO 1T':      dict(color=_C['UNO'],         ls='-',  lw=1.4, marker='',  ms=0),
    'UNO RT':      dict(color=_C['UNO'],         ls='--', lw=1.4, marker='',  ms=0),
    'TFNO 1T':     dict(color=_C['TFNO'],        ls='-',  lw=1.4, marker='',  ms=0),
    'TFNO RT':     dict(color=_C['TFNO'],        ls='--', lw=1.4, marker='',  ms=0),
    'AFNO 1T':     dict(color=_C['AFNO'],        ls='-',  lw=1.6, marker='',  ms=0),
    'AFNO RT':     dict(color=_C['AFNO'],        ls='-',  lw=2.2, marker='o', ms=3),
}

# Variant B: no marker = 1T, triangle marker = RT (all solid, same color per architecture)
MODEL_STYLES_MARKER = {
    'Persistence': dict(color=_C['Persistence'], ls='--', lw=1.4, marker='',  ms=0),
    'FNO 1T':      dict(color=_C['FNO'],         ls='-',  lw=1.4, marker='',  ms=0),
    'FNO RT':      dict(color=_C['FNO'],         ls='-',  lw=1.4, marker='^', ms=4),
    'UNO 1T':      dict(color=_C['UNO'],         ls='-',  lw=1.4, marker='',  ms=0),
    'UNO RT':      dict(color=_C['UNO'],         ls='-',  lw=1.4, marker='^', ms=4),
    'TFNO 1T':     dict(color=_C['TFNO'],        ls='-',  lw=1.4, marker='',  ms=0),
    'TFNO RT':     dict(color=_C['TFNO'],        ls='-',  lw=1.4, marker='^', ms=4),
    'AFNO 1T':     dict(color=_C['AFNO'],        ls='-',  lw=1.6, marker='',  ms=0),
    'AFNO RT':     dict(color=_C['AFNO'],        ls='-',  lw=2.2, marker='^', ms=4),
}

_DEFAULT_STYLE = dict(color='#aaaaaa', ls='-', lw=1.2, marker='', ms=0)


# ── I/O helpers ──────────────────────────────────────────────────────────────

def find_metrics_file(results_dir: Path) -> Path | None:
    """Return the best available metrics file in ``results_dir``.

    Args:
        results_dir (Path): Model results directory.

    Returns:
        Path | None: Path to the metrics file, or None if not found.

    Example:
        >>> find_metrics_file(Path('results/AFNO_BoB_Surf_E11p1'))
        PosixPath('results/AFNO_BoB_Surf_E11p1/forecast_metrics_combined.txt')
    """
    for fname in METRICS_FILENAMES:
        p = results_dir / fname
        if p.exists():
            return p
    return None


def parse_lead_metrics(filepath: Path, leads: list[int]) -> dict[str, dict[int, dict[str, float]]]:
    """Parse per-lead RMSE and Pearson r for each variable from a metrics file.

    Args:
        filepath (Path): Path to a forecast_metrics*.txt file.
        leads (list[int]): Lead days to extract.

    Returns:
        dict: var → {lead → {'rmse': float, 'pearson': float}}.
            Missing values are NaN.

    Example:
        >>> data = parse_lead_metrics(Path('results/.../forecast_metrics_combined.txt'), list(range(1,10)))
        >>> data['thetao'][3]['rmse']
        0.202
    """
    text = filepath.read_text()
    result: dict[str, dict[int, dict[str, float]]] = {v: {} for v in VAR_ORDER}

    blocks = re.split(r'LEAD TIME: \+(\d+) day\(s\)', text)
    for i in range(1, len(blocks), 2):
        lt   = int(blocks[i])
        if lt not in leads:
            continue
        body = blocks[i + 1]
        for var in VAR_ORDER:
            rmse_m    = re.search(rf'{var.upper()}:\s+RMSE:\s+([\d.]+)', body, re.IGNORECASE)
            pearson_m = re.search(rf'{var.upper()}:.*?Pearson Corr:\s+([\d.]+)', body,
                                  re.IGNORECASE | re.DOTALL)
            result[var][lt] = {
                'rmse':    float(rmse_m.group(1))    if rmse_m    else float('nan'),
                'pearson': float(pearson_m.group(1)) if pearson_m else float('nan'),
            }

    return result


# ── plotting ─────────────────────────────────────────────────────────────────

def make_figure(all_data: dict[str, dict],
                model_names: list[str],
                leads: list[int],
                width: float,
                model_styles: dict | None = None) -> plt.Figure:
    """Build the 5-row × 2-col comparison figure.

    Args:
        all_data (dict): model_label → {var → {lead → {'rmse', 'pearson'}}}.
        model_names (list[str]): Ordered model labels.
        leads (list[int]): Lead days (x-axis).
        width (float): Figure width in inches.
        model_styles (dict | None): Style dict keyed by model label.
            Defaults to MODEL_STYLES_LINESTYLE.

    Returns:
        plt.Figure: The completed figure.

    Example:
        >>> fig = make_figure(all_data, model_names, list(range(1,10)), 8.0)
        >>> fig.savefig('out.pdf')
    """
    if model_styles is None:
        model_styles = MODEL_STYLES_LINESTYLE
    n_vars = len(VAR_ORDER)
    row_h  = 1.9          # inches per row
    top    = 0.25         # space for col labels
    bottom = 0.80         # space for 2-row legend
    fig_h  = n_vars * row_h + top + bottom

    fig, axes = plt.subplots(
        n_vars, 2,
        figsize=(width, fig_h),
        constrained_layout=False,
    )

    # Manual margins
    fig.subplots_adjust(
        left=0.10, right=0.98,
        top=1 - top / fig_h,
        bottom=bottom / fig_h,
        hspace=0.15, wspace=0.30,
    )

    x = np.array(leads)

    for row, var in enumerate(VAR_ORDER):
        meta      = VAR_META[var]
        ax_rmse   = axes[row, 0]
        ax_r      = axes[row, 1]

        for name in model_names:
            sty  = model_styles.get(name, _DEFAULT_STYLE)
            data = all_data[name][var]

            rmse_vals    = np.array([data.get(lt, {}).get('rmse',    float('nan')) for lt in leads])
            pearson_vals = np.array([data.get(lt, {}).get('pearson', float('nan')) for lt in leads])

            kw = dict(color=sty['color'], linestyle=sty['ls'],
                      linewidth=sty['lw'], label=name,
                      marker=sty['marker'] if sty['marker'] else None,
                      markersize=sty['ms'])

            ax_rmse.plot(x, rmse_vals, **kw)
            ax_r.plot(x, pearson_vals, **kw)

        # ── RMSE panel styling ────────────────────────────────────────────
        ax_rmse.set_ylabel(f'RMSE ({meta["rmse_unit"]})', fontsize=9)
        ax_rmse.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        ax_rmse.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, min_n_ticks=3))
        ax_rmse.set_ylim(bottom=0)

        # ── Pearson r panel styling ───────────────────────────────────────
        ax_r.set_ylabel('Pearson $r$', fontsize=9)
        ax_r.set_ylim(meta['r_ylim'])
        ax_r.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax_r.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, min_n_ticks=3))
        ax_r.axhline(1.0, color='#cccccc', lw=0.6, zorder=0)

        for ax in (ax_rmse, ax_r):
            ax.set_xticks(leads)
            ax.tick_params(axis='both', labelsize=9)
            ax.grid(axis='both', lw=0.4, color='#e0e0e0', zorder=0)
            for spine in ax.spines.values():
                spine.set_visible(True)
            if row < n_vars - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Lead day', fontsize=9)

        # Variable label on left margin
        ax_rmse.annotate(
            meta['label'],
            xy=(-0.28, 0.5), xycoords='axes fraction',
            fontsize=10, fontweight='bold',
            ha='center', va='center', rotation=90,
        )

    # ── Column titles ─────────────────────────────────────────────────────
    axes[0, 0].set_title('RMSE', fontsize=10, fontweight='bold', pad=4)
    axes[0, 1].set_title('Pearson $r$', fontsize=10, fontweight='bold', pad=4)

    # ── Shared legend at bottom ───────────────────────────────────────────
    # matplotlib fills legend column-major; reorder to get row-major visual layout
    handles, labels = axes[0, 0].get_legend_handles_labels()
    ncol_leg = math.ceil(len(handles) / 2)
    nrows_leg = math.ceil(len(handles) / ncol_leg)
    row_major_idx = [
        row * ncol_leg + col
        for col in range(ncol_leg)
        for row in range(nrows_leg)
        if row * ncol_leg + col < len(handles)
    ]
    handles = [handles[i] for i in row_major_idx]
    labels  = [labels[i]  for i in row_major_idx]
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=ncol_leg,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.54, 0.0),
        columnspacing=1.0,
        handlelength=2.0,
    )

    return fig


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point: load metrics, build figure, save PDF and PNG.

    Example:
        python src/visualization/plot_model_comparison_trends.py \\
            --models "Persistence:results/Persistence_Baseline" \\
                     "AFNO RT:results/AFNO_BoB_Surf_E11p1"
    """
    parser = argparse.ArgumentParser(description='Plot 9-day RMSE and Pearson r trends')
    parser.add_argument('--models', nargs='+', required=True,
                        help='"Label:results_dir" pairs in legend order')
    parser.add_argument('--leads', nargs='+', type=int, default=list(range(1, 10)))
    parser.add_argument('--output', default='results/figures/model_comparison_trends.pdf')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--width', type=float, default=8.0)
    parser.add_argument('--line-variant', choices=['linestyle', 'marker'], default='linestyle',
                        help='Style variant: linestyle (solid=1T, dashed=RT) or '
                             'marker (no-marker=1T, triangle=RT)')
    args = parser.parse_args()

    plt.rcParams.update({
        'font.family':     'sans-serif',
        'font.size':       9,
        'axes.linewidth':  0.6,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'lines.linewidth': 1.4,
        'pdf.fonttype':    42,   # embeds fonts for journal submission
        'ps.fonttype':     42,
    })

    model_names: list[str] = []
    all_data: dict[str, dict] = {}

    print('=== Loading metrics ===')
    for entry in args.models:
        label, results_dir = entry.split(':', 1)
        mfile = find_metrics_file(Path(results_dir))
        if mfile is None:
            print(f'  WARNING: no metrics file in {results_dir} — skipping')
            continue
        all_data[label] = parse_lead_metrics(mfile, args.leads)
        model_names.append(label)
        print(f'  {label:20s}  {mfile.name}')

    if not model_names:
        print('No valid models. Exiting.')
        sys.exit(1)

    styles = MODEL_STYLES_MARKER if args.line_variant == 'marker' else MODEL_STYLES_LINESTYLE
    print(f'=== Building figure (variant: {args.line_variant}) ===')
    fig = make_figure(all_data, model_names, args.leads, args.width, model_styles=styles)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
    print(f'Saved: {out}')

    png_out = out.with_suffix('.png')
    fig.savefig(png_out, dpi=args.dpi, bbox_inches='tight')
    print(f'Saved: {png_out}')
    plt.close(fig)


if __name__ == '__main__':
    main()

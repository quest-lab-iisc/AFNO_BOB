"""Generate LaTeX comparison tables of average metrics across lead days 1-9.

Reads ``forecast_metrics_combined.txt`` (preferred), ``forecast_metrics_extended.txt``,
or ``forecast_metrics.txt`` from each model's results directory.  Extracts per-lead
RMSE, MAE, R², and Pearson correlation, averages over the requested leads, and writes
one of three publication-ready LaTeX table layouts:

  * ``multirow`` (default): variables in rows, one column per model, two sub-rows
    per variable (RMSE and Pearson r).
  * ``multicol``: variables in rows, two sub-columns per model (RMSE and r).
  * ``transposed``: models in rows, variables in columns, four sub-columns per
    variable (RMSE, MAE, R², r).

Inputs:
    --models (str, repeatable): One or more ``label:results_dir`` pairs, e.g.
        ``"AFNO RT:results/AFNO_BoB_Surf_E11p1"``.
        Order determines left-to-right / top-to-bottom ordering.
    --leads (int list): Lead days to average over (default 1 2 3 4 5 6 7 8 9).
    --output (str): Output .tex file path (default: results/tables/model_comparison_table.tex).
    --caption (str): LaTeX table caption.
    --label (str): LaTeX label.
    --style (str): One of ``multirow``, ``multicol``, ``transposed``.

Outputs:
    <output>  LaTeX table source file.

Example:
    python src/inference/run_model_comparison_table.py \\
        --models "Persistence:results/Persistence_Baseline" \\
                 "FNO:results/FNO_BoB_Surf_E01" \\
                 "AFNO RT:results/AFNO_BoB_Surf_E11p1" \\
        --style transposed \\
        --output results/tables/model_comparison_transposed.tex
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


VAR_LABELS = {
    'thetao': 'SST',
    'so':     'SSS',
    'zos':    'SSH',
    'uo':     'Zonal current',
    'vo':     'Meridional current',
}
VAR_ORDER = ['thetao', 'so', 'zos', 'uo', 'vo']
METRICS_FILENAMES = [
    'forecast_metrics_combined.txt',
    'forecast_metrics_extended.txt',
    'forecast_metrics.txt',
]


def find_metrics_file(results_dir: Path) -> Path | None:
    """Return the best available metrics file in ``results_dir``.

    Preference order: combined → extended → basic.

    Args:
        results_dir (Path): Model results directory.

    Returns:
        Path | None: Path to the metrics file, or None if none found.

    Example:
        >>> find_metrics_file(Path('results/AFNO_BoB_Surf_E11p1'))
        PosixPath('results/AFNO_BoB_Surf_E11p1/forecast_metrics_combined.txt')
    """
    for fname in METRICS_FILENAMES:
        p = results_dir / fname
        if p.exists():
            return p
    return None


def parse_metrics(filepath: Path, leads: list[int]) -> dict[str, dict[str, float]]:
    """Parse per-lead RMSE and Pearson correlation from a metrics file and average over leads.

    Args:
        filepath (Path): Path to a forecast_metrics*.txt file.
        leads (list[int]): Lead days to include in the average.

    Returns:
        dict[str, dict[str, float]]: Variable name → {'rmse': float, 'pearson': float},
            averaged over ``leads``.  Missing values default to NaN.

    Example:
        >>> m = parse_metrics(Path('results/.../forecast_metrics_combined.txt'), [1,3,5,7,9])
        >>> m['thetao']['rmse']
        0.243
        >>> m['thetao']['pearson']
        0.961
    """
    text = filepath.read_text()

    # Map lead time → {var → {rmse, mae, r2, pearson}}
    lead_data: dict[int, dict[str, dict[str, float]]] = {}

    blocks = re.split(r'LEAD TIME: \+(\d+) day\(s\)', text)
    for i in range(1, len(blocks), 2):
        lt   = int(blocks[i])
        body = blocks[i + 1]
        lead_data[lt] = {}
        for var in VAR_ORDER:
            rmse_m    = re.search(rf'{var.upper()}:\s+RMSE:\s+([\d.]+)', body, re.IGNORECASE)
            mae_m     = re.search(rf'{var.upper()}:.*?MAE:\s+([\d.]+)', body,
                                  re.IGNORECASE | re.DOTALL)
            r2_m      = re.search(rf'{var.upper()}:.*?R²:\s+([\d.eE+\-]+)', body,
                                  re.IGNORECASE | re.DOTALL)
            pearson_m = re.search(rf'{var.upper()}:.*?Pearson Corr:\s+([\d.]+)', body,
                                  re.IGNORECASE | re.DOTALL)
            if rmse_m or pearson_m:
                lead_data[lt][var] = {
                    'rmse':    float(rmse_m.group(1))    if rmse_m    else float('nan'),
                    'mae':     float(mae_m.group(1))     if mae_m     else float('nan'),
                    'r2':      float(r2_m.group(1))      if r2_m      else float('nan'),
                    'pearson': float(pearson_m.group(1)) if pearson_m else float('nan'),
                }

    result: dict[str, dict[str, float]] = {}
    for var in VAR_ORDER:
        def _avg(key: str) -> float:
            vals = [lead_data[lt][var][key] for lt in leads
                    if lt in lead_data and var in lead_data[lt]]
            return float(sum(vals) / len(vals)) if vals else float('nan')
        result[var] = {
            'rmse':    _avg('rmse'),
            'mae':     _avg('mae'),
            'r2':      _avg('r2'),
            'pearson': _avg('pearson'),
        }

    return result


def write_latex_table_multicol(model_metrics: dict[str, dict[str, dict[str, float]]],
                               model_names: list[str],
                               output_path: Path,
                               caption: str,
                               label: str,
                               leads: list[int]) -> None:
    """Write a LaTeX table where each model occupies two sub-columns (RMSE and r).

    Each ocean variable has a single row.  Model names span two sub-columns via
    ``\\multicolumn``.  Bold marks the best value across models for each metric;
    underline marks the second best.  Requires ``booktabs``, ``multicolumn``, and
    ``graphicx`` packages.

    Args:
        model_metrics (dict): model_label → {var → {'rmse': float, 'pearson': float}}.
        model_names (list[str]): Ordered list of model labels (column headers).
        output_path (Path): Destination .tex file.
        caption (str): LaTeX caption string.
        label (str): LaTeX label string.
        leads (list[int]): Lead days averaged (shown in caption).

    Returns:
        None: Writes .tex to output_path.

    Example:
        >>> write_latex_table_multicol(metrics, names, Path('table.tex'), 'cap', 'tab:cmp', [1,9])
    """
    n = len(model_names)
    # Column spec: variable column, then two sub-columns per model separated by vertical rules
    col_spec = 'l|' + '|'.join('cc' for _ in model_names)

    def fmt(val: float) -> str:
        """Format a metric value to 3 decimal places, or '---' for NaN."""
        return f'{val:.3f}' if val == val else '---'

    def mark_row(vals: list[float], higher_is_better: bool) -> list[str]:
        """Return formatted cells with bold (best) and underline (second best).

        Args:
            vals (list[float]): One value per model.
            higher_is_better (bool): Direction of optimality.

        Returns:
            list[str]: LaTeX-formatted cell strings, one per model.
        """
        # Rank on displayed (3-dp) values so ties at the printed precision share a mark.
        finite = sorted(set(round(v, 3) for v in vals if v == v), reverse=higher_is_better)
        best   = finite[0] if len(finite) >= 1 else None
        second = finite[1] if len(finite) >= 2 else None
        cells = []
        for v in vals:
            s = fmt(v)
            if best is not None and v == v and abs(round(v, 3) - best) < 1e-9:
                s = f'\\textbf{{{s}}}'
            elif second is not None and v == v and abs(round(v, 3) - second) < 1e-9:
                s = f'\\underline{{{s}}}'
            cells.append(s)
        return cells

    lines: list[str] = []
    leads_str = ', '.join(str(l) for l in leads)

    lines.append(r'\begin{table}[ht]')
    lines.append(f'  \\caption{{{caption} Lead days averaged: {leads_str}.}}')
    lines.append(f'  \\label{{{label}}}')
    lines.append(r'  \centering')
    lines.append(r'  \resizebox{\textwidth}{!}{')
    lines.append(f'  \\begin{{tabular}}{{{col_spec}}}')
    lines.append(r'    \toprule')

    # Row 1: model names spanning two sub-columns each
    mcols = []
    for i, name in enumerate(model_names):
        sep = 'c|' if i < n - 1 else 'c'
        mcols.append(f'\\multicolumn{{2}}{{{sep}}}{{\\textbf{{{name}}}}}')
    lines.append(f'    \\textbf{{Variable}} & ' + ' & '.join(mcols) + r' \\')

    # Row 2: RMSE / r sub-headers
    sub = ' & '.join(['RMSE & $r$'] * n)
    lines.append(f'    & {sub} \\\\')
    lines.append(r'    \midrule')

    # Data rows — one per variable
    for var in VAR_ORDER:
        var_label = VAR_LABELS.get(var, var)
        rmse_vals    = [model_metrics[m][var]['rmse']    for m in model_names]
        pearson_vals = [model_metrics[m][var]['pearson'] for m in model_names]

        rmse_cells    = mark_row(rmse_vals,    higher_is_better=False)
        pearson_cells = mark_row(pearson_vals, higher_is_better=True)

        # Interleave: rmse_0, r_0, rmse_1, r_1, ...
        interleaved = []
        for r, p in zip(rmse_cells, pearson_cells):
            interleaved.extend([r, p])

        lines.append(f'    {var_label} & ' + ' & '.join(interleaved) + r' \\')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}}')
    lines.append(r'\end{table}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines) + '\n')
    print(f'Saved: {output_path}')


def write_latex_table(model_metrics: dict[str, dict[str, dict[str, float]]],
                      model_names: list[str],
                      output_path: Path,
                      caption: str,
                      label: str,
                      leads: list[int]) -> None:
    """Write a LaTeX table comparing average RMSE and Pearson r across models.

    Each ocean variable occupies two rows: one for RMSE (lower = better) and one
    for Pearson correlation (higher = better).  The variable label spans both rows
    via ``\\multirow``.  Bold marks the best value per row; underline marks the
    second best.  Requires the ``booktabs``, ``multirow``, and ``graphicx`` packages.

    Args:
        model_metrics (dict): model_label → {var → {'rmse': float, 'pearson': float}}.
        model_names (list[str]): Ordered list of model labels (column headers).
        output_path (Path): Destination .tex file.
        caption (str): LaTeX caption string.
        label (str): LaTeX label string.
        leads (list[int]): Lead days averaged (shown in caption).

    Returns:
        None: Writes .tex to output_path.

    Example:
        >>> write_latex_table(metrics, names, Path('table.tex'), 'My caption', 'tab:cmp', [1,9])
    """
    n_models = len(model_names)
    col_spec = 'll|' + 'c' * n_models

    def fmt(val: float) -> str:
        """Format a metric value to 3 decimal places, or '---' for NaN."""
        return f'{val:.3f}' if val == val else '---'

    def mark_row(vals: list[float], higher_is_better: bool) -> list[str]:
        """Return formatted cells with bold (best) and underline (second best).

        Args:
            vals (list[float]): One value per model.
            higher_is_better (bool): If True, largest value is best; else smallest.

        Returns:
            list[str]: LaTeX-formatted cell strings.
        """
        # Rank on displayed (3-dp) values so ties at the printed precision share a mark.
        finite = sorted(set(round(v, 3) for v in vals if v == v),
                        reverse=higher_is_better)
        best   = finite[0] if len(finite) >= 1 else None
        second = finite[1] if len(finite) >= 2 else None

        cells = []
        for v in vals:
            s = fmt(v)
            if best is not None and v == v and abs(round(v, 3) - best) < 1e-9:
                s = f'\\textbf{{{s}}}'
            elif second is not None and v == v and abs(round(v, 3) - second) < 1e-9:
                s = f'\\underline{{{s}}}'
            cells.append(s)
        return cells

    lines: list[str] = []
    leads_str = ', '.join(str(l) for l in leads)

    lines.append(r'\begin{table}[ht]')
    lines.append(f'  \\caption{{{caption} Lead days averaged: {leads_str}.}}')
    lines.append(f'  \\label{{{label}}}')
    lines.append(r'  \centering')
    lines.append(r'  \resizebox{\textwidth}{!}{')
    lines.append(f'  \\begin{{tabular}}{{{col_spec}}}')
    lines.append(r'    \toprule')

    # Header row
    header_cols = ' & '.join(f'\\textbf{{{n}}}' for n in model_names)
    lines.append(f'    \\textbf{{Variable}} & \\textbf{{Metric}} & {header_cols} \\\\')
    lines.append(r'    \midrule')

    for idx, var in enumerate(VAR_ORDER):
        var_label = VAR_LABELS.get(var, var)

        rmse_vals    = [model_metrics[m][var]['rmse']    for m in model_names]
        pearson_vals = [model_metrics[m][var]['pearson'] for m in model_names]

        rmse_cells    = mark_row(rmse_vals,    higher_is_better=False)
        pearson_cells = mark_row(pearson_vals, higher_is_better=True)

        # Variable label spans two rows
        lines.append(
            f'    \\multirow{{2}}{{*}}{{{var_label}}} & RMSE & '
            + ' & '.join(rmse_cells) + r' \\'
        )
        lines.append(
            f'     & $r$ & '
            + ' & '.join(pearson_cells) + r' \\'
        )

        # Separator between variable groups (not after the last one)
        if idx < len(VAR_ORDER) - 1:
            lines.append(r'    \midrule')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}}')
    lines.append(r'\end{table}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines) + '\n')
    print(f'Saved: {output_path}')


def write_latex_table_transposed(model_metrics: dict[str, dict[str, dict[str, float]]],
                                 model_names: list[str],
                                 output_path: Path,
                                 caption: str,
                                 label: str,
                                 leads: list[int]) -> None:
    """Write a transposed LaTeX table: models in rows, variable×metric sub-columns.

    Each ocean variable occupies two sub-columns (RMSE, r).  Model names form the
    row labels.  Bold marks the best value per column; underline marks the second
    best.  Requires ``booktabs``, ``multicolumn``, and ``graphicx`` packages.

    Args:
        model_metrics (dict): model_label → {var → {'rmse', 'pearson'}}.
        model_names (list[str]): Ordered list of model labels (row labels).
        output_path (Path): Destination .tex file.
        caption (str): LaTeX caption string.
        label (str): LaTeX label string.
        leads (list[int]): Lead days averaged (shown in caption).

    Returns:
        None: Writes .tex to output_path.

    Example:
        >>> write_latex_table_transposed(metrics, names, Path('t.tex'), 'cap', 'tab:t', [1,9])
    """
    n_vars = len(VAR_ORDER)
    metrics_order = ['rmse', 'pearson']
    metric_labels = {'rmse': 'RMSE', 'pearson': '$r$'}
    higher_is_better = {'rmse': False, 'pearson': True}

    # col spec: model name | then for each var: 2 metric sub-cols separated by |
    col_spec = 'l|' + '|'.join('cc' for _ in VAR_ORDER)

    def fmt(val: float) -> str:
        """Format a metric value to 3 decimal places, or '---' for NaN."""
        return f'{val:.3f}' if val == val else '---'

    def mark_col(vals: list[float], hib: bool) -> list[str]:
        """Return formatted cells with bold (best) and underline (second best).

        Args:
            vals (list[float]): One value per model.
            hib (bool): True if higher is better.

        Returns:
            list[str]: LaTeX-formatted cell strings.
        """
        # Rank on displayed (3-dp) values so ties at the printed precision share a mark.
        finite = sorted(set(round(v, 3) for v in vals if v == v), reverse=hib)
        best   = finite[0] if len(finite) >= 1 else None
        second = finite[1] if len(finite) >= 2 else None
        cells = []
        for v in vals:
            s = fmt(v)
            if best is not None and v == v and abs(round(v, 3) - best) < 1e-9:
                s = f'\\textbf{{{s}}}'
            elif second is not None and v == v and abs(round(v, 3) - second) < 1e-9:
                s = f'\\underline{{{s}}}'
            cells.append(s)
        return cells

    # Pre-compute per-column markings (columns = variable × metric combinations)
    marked: dict[tuple[str, str], list[str]] = {}
    for var in VAR_ORDER:
        for metric in metrics_order:
            vals = [model_metrics[m][var][metric] for m in model_names]
            marked[(var, metric)] = mark_col(vals, higher_is_better[metric])

    lines: list[str] = []
    leads_str = ', '.join(str(l) for l in leads)

    lines.append(r'\begin{table}[ht]')
    lines.append(f'  \\caption{{{caption} Lead days averaged: {leads_str}.}}')
    lines.append(f'  \\label{{{label}}}')
    lines.append(r'  \centering')
    lines.append(r'  \resizebox{\textwidth}{!}{')
    lines.append(f'  \\begin{{tabular}}{{{col_spec}}}')
    lines.append(r'    \toprule')

    # Header row 1: variable labels spanning 2 sub-cols each
    var_mcols = []
    for i, var in enumerate(VAR_ORDER):
        sep = 'c|' if i < n_vars - 1 else 'c'
        var_mcols.append(f'\\multicolumn{{2}}{{{sep}}}{{\\textbf{{{VAR_LABELS[var]}}}}}')
    lines.append(r'    \textbf{Model} & ' + ' & '.join(var_mcols) + r' \\')

    # Header row 2: metric sub-headers repeated for each variable
    sub_headers = ' & '.join(
        ' & '.join(metric_labels[m] for m in metrics_order)
        for _ in VAR_ORDER
    )
    lines.append(f'    & {sub_headers} \\\\')
    lines.append(r'    \midrule')

    # Data rows — one per model
    for row_idx, model in enumerate(model_names):
        cells = []
        for var in VAR_ORDER:
            for met_idx, metric in enumerate(metrics_order):
                col_cells = marked[(var, metric)]
                cells.append(col_cells[row_idx])
        lines.append(f'    {model} & ' + ' & '.join(cells) + r' \\')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}}')
    lines.append(r'\end{table}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines) + '\n')
    print(f'Saved: {output_path}')


def main() -> None:
    """Entry point: parse model list, read metrics, write LaTeX table.

    Example:
        python src/inference/run_model_comparison_table.py \\
            --models "Persistence:results/Persistence_Baseline" \\
                     "AFNO RT:results/AFNO_BoB_Surf_E11p1"
    """
    parser = argparse.ArgumentParser(
        description='Generate model comparison RMSE + Pearson r table')
    parser.add_argument('--models', nargs='+', required=True,
                        help='"Label:results_dir" pairs in display order')
    parser.add_argument('--leads', nargs='+', type=int,
                        default=list(range(1, 10)))
    parser.add_argument('--output',
                        default='results/tables/model_comparison_table.tex')
    parser.add_argument('--caption',
                        default=(
                            'Comparison of average RMSE and Pearson correlation ($r$) '
                            'across the 2020--2025 test period for all ocean variables. '
                            'Bold values indicate the best model per metric row; '
                            'underlined values indicate the second best; values tied at the displayed precision share a mark.'
                        ))
    parser.add_argument('--label', default='tab:model_comparison')
    parser.add_argument('--style', choices=['multirow', 'multicol', 'transposed'],
                        default='multirow',
                        help='Table layout: multirow (one col per model, two rows per var); '
                             'multicol (two sub-cols per model, one row per var); '
                             'transposed (models in rows, variable×metric sub-columns)')
    args = parser.parse_args()

    model_names:   list[str]       = []
    model_metrics: dict[str, dict] = {}

    print('=== Model comparison table ===')
    for entry in args.models:
        label, results_dir = entry.split(':', 1)
        mfile = find_metrics_file(Path(results_dir))
        if mfile is None:
            print(f'  WARNING: no metrics file found in {results_dir} — skipping')
            continue
        metrics = parse_metrics(mfile, args.leads)
        model_names.append(label)
        model_metrics[label] = metrics
        print(f'  {label:20s}  ({mfile.name}):')
        for v in VAR_ORDER:
            print(f'    {v}: RMSE={metrics[v]["rmse"]:.3f}  r={metrics[v]["pearson"]:.3f}')

    if not model_names:
        print('No valid models found. Exiting.')
        sys.exit(1)

    if args.style == 'multicol':
        writer = write_latex_table_multicol
    elif args.style == 'transposed':
        writer = write_latex_table_transposed
    else:
        writer = write_latex_table
    writer(
        model_metrics, model_names,
        Path(args.output),
        args.caption,
        args.label,
        args.leads,
    )


if __name__ == '__main__':
    main()

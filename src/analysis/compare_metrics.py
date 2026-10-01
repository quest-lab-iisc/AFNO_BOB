"""Compare forecast metrics at key lead times across experiments.

Reads forecast_metrics.txt, forecast_metrics_extended.txt, and
forecast_metrics_combined.txt for each specified experiment and writes
Unicode box-drawing comparison tables — one per metric file, plus a summary
table averaged across all lead times.  The first experiment is the baseline;
all others display values with a percentage-change annotation.

Inputs:
    --experiments (list[str]): Space-separated experiment names.  The first
        entry is the baseline.
        E.g. AFNO_BoB_Surf_E00 AFNO_BoB_Surf_E11
    --results_dir (str): Root results directory (default: results/).
    --leads (list[int]): Lead days to show as columns (default: 1 5 9).
    --metric (str): rmse | mae | r2 | pearson (default: rmse).
    --output_dir (str): Directory to write .txt table files
        (default: results/tables/).

Outputs:
    {output_dir}/compare_{metric}_2020.txt
    {output_dir}/compare_{metric}_extended.txt
    {output_dir}/compare_{metric}_combined.txt
    All tables are also printed to stdout.

Example:
    conda activate BoB_Surf_2
    python src/analysis/compare_metrics.py \\
        --experiments AFNO_BoB_Surf_E00 AFNO_BoB_Surf_E11 \\
        --leads 1 5 9
"""

import argparse
import os
import re
from pathlib import Path


METRIC_LABEL = {
    'rmse':    'RMSE',
    'mae':     'MAE',
    'r2':      'R²',
    'pearson': 'Pearson Corr',
}

VARIABLES = ['THETAO', 'SO', 'UO', 'VO', 'ZOS']

FILE_TAGS = [
    ('forecast_metrics.txt',          '2020'),
    ('forecast_metrics_extended.txt', 'extended'),
    ('forecast_metrics_combined.txt', 'combined'),
]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_metrics_file(path: Path, metric_key: str) -> dict:
    """Parse a forecast_metrics*.txt file and extract values for one metric.

    Args:
        path (Path): Path to the metrics text file.
        metric_key (str): One of 'rmse', 'mae', 'r2', 'pearson'.

    Returns:
        dict: Nested dict mapping lead_day (int) or 'summary' to
            {variable_name (str): metric_value (float)}.

    Example:
        >>> data = parse_metrics_file(Path('results/E00/forecast_metrics.txt'), 'rmse')
        >>> data[1]['THETAO']
        0.126493
    """
    label = METRIC_LABEL[metric_key]
    data: dict = {}
    current_lead = None
    current_var = None
    in_summary = False

    for line in path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()

        m = re.match(r'LEAD TIME: \+(\d+) day\(s\)', stripped)
        if m:
            current_lead = int(m.group(1))
            in_summary = False
            data[current_lead] = {}
            current_var = None
            continue

        if re.match(r'SUMMARY \(Averaged', stripped):
            in_summary = True
            current_lead = 'summary'
            data['summary'] = {}
            current_var = None
            continue

        if current_lead is None:
            continue

        if in_summary:
            m = re.match(r'^([A-Z0-9]+):$', stripped)
        else:
            m = re.match(r'^  ([A-Z0-9]+):$', line.rstrip())
        if m:
            current_var = m.group(1)
            continue

        if current_var is not None:
            m = re.match(rf'\s+{re.escape(label)}:\s+([\d.]+)', line)
            if m:
                data[current_lead][current_var] = float(m.group(1))

    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_names(experiments: list) -> list:
    """Strip the common underscore-delimited prefix from experiment names.

    Args:
        experiments (list): Full experiment names, e.g.
            ['AFNO_BoB_Surf_E00', 'AFNO_BoB_Surf_E11'].

    Returns:
        list: Shortened names, e.g. ['E00', 'E11'].

    Example:
        >>> _short_names(['AFNO_BoB_Surf_E00', 'AFNO_BoB_Surf_E11'])
        ['E00', 'E11']
    """
    if len(experiments) <= 1:
        return list(experiments)
    prefix = os.path.commonprefix(experiments)
    if '_' in prefix:
        prefix = prefix[: prefix.rfind('_') + 1]
    return [e[len(prefix):] if prefix else e for e in experiments]


def _pct_cell(baseline_val, exp_val: float) -> str:
    """Format a comparison cell as 'value (±pct%)'.

    Args:
        baseline_val (float or None): Baseline metric value.
        exp_val (float or None): Experiment metric value.

    Returns:
        str: Formatted string, e.g. '0.3143 (−7%)'.

    Example:
        >>> _pct_cell(0.3389, 0.3143)
        '0.3143 (−7%)'
    """
    if exp_val is None:
        return 'N/A'
    if baseline_val is None or baseline_val == 0:
        return f'{exp_val:.4f}'
    pct = (exp_val - baseline_val) / baseline_val * 100
    sign = '+' if pct >= 0 else '−'
    return f'{exp_val:.4f} ({sign}{abs(pct):.0f}%)'


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def _render_table(headers: list, table_rows: list, title: str = '') -> str:
    """Render a Unicode box-drawing table from headers and row data.

    Args:
        headers (list[str]): Column header strings.
        table_rows (list[list[str]]): Each inner list is one data row.
        title (str): Optional title line above the table.

    Returns:
        str: Table with Unicode box-drawing borders.

    Example:
        >>> print(_render_table(['A', 'B'], [['x', 'y'], ['p', 'q']]))
    """
    col_widths = [
        max(len(headers[j]), max(len(r[j]) for r in table_rows))
        for j in range(len(headers))
    ]

    def hline(left, mid, right):
        return left + mid.join('─' * (w + 2) for w in col_widths) + right

    def render_row(cells):
        parts = [f' {c.ljust(col_widths[j])} ' for j, c in enumerate(cells)]
        return '│' + '│'.join(parts) + '│'

    lines = []
    if title:
        lines.append(title)
    lines.append(hline('┌', '┬', '┐'))
    lines.append(render_row(headers))
    lines.append(hline('├', '┼', '┤'))
    for ri, row in enumerate(table_rows):
        lines.append(render_row(row))
        if ri < len(table_rows) - 1:
            lines.append(hline('├', '┼', '┤'))
    lines.append(hline('└', '┴', '┘'))
    return '\n'.join(lines)


def build_lead_table(data_per_exp: dict, experiments: list, leads: list,
                     variables: list, title: str = '') -> str:
    """Build a comparison table showing the chosen metric at key lead times.

    Args:
        data_per_exp (dict): Maps experiment name to parsed metrics dict.
        experiments (list[str]): Experiment names; first is the baseline.
        leads (list[int]): Lead days to include as column groups.
        variables (list[str]): Variable names to include as rows.
        title (str): Optional title line above the table.

    Returns:
        str: Rendered Unicode table.

    Example:
        >>> t = build_lead_table(data, ['E00', 'E11'], [1, 5, 9], VARIABLES)
        >>> print(t)
    """
    baseline = experiments[0]
    short = _short_names(experiments)

    headers = ['Variable']
    for lead in leads:
        for i, exp in enumerate(experiments):
            headers.append(f'+{lead}d {short[i]}')

    table_rows = []
    for var in variables:
        cells = [var]
        base_vals = {
            ld: data_per_exp.get(baseline, {}).get(ld, {}).get(var)
            for ld in leads
        }
        for lead in leads:
            for i, exp in enumerate(experiments):
                val = data_per_exp.get(exp, {}).get(lead, {}).get(var)
                if i == 0:
                    cells.append(f'{val:.4f}' if val is not None else 'N/A')
                else:
                    cells.append(_pct_cell(base_vals[lead], val))
        table_rows.append(cells)

    return _render_table(headers, table_rows, title)


def build_summary_table(data_per_exp: dict, experiments: list,
                        variables: list, title: str = '') -> str:
    """Build a comparison table of metrics averaged across all lead times.

    Args:
        data_per_exp (dict): Maps experiment name to parsed metrics dict.
        experiments (list[str]): Experiment names; first is the baseline.
        variables (list[str]): Variable names to include as rows.
        title (str): Optional title line above the table.

    Returns:
        str: Rendered Unicode table.

    Example:
        >>> t = build_summary_table(data, ['E00', 'E11'], VARIABLES)
        >>> print(t)
    """
    baseline = experiments[0]
    short = _short_names(experiments)

    headers = ['Variable'] + list(short)

    table_rows = []
    for var in variables:
        base_val = data_per_exp.get(baseline, {}).get('summary', {}).get(var)
        cells = [var]
        for i, exp in enumerate(experiments):
            val = data_per_exp.get(exp, {}).get('summary', {}).get(var)
            if i == 0:
                cells.append(f'{val:.4f}' if val is not None else 'N/A')
            else:
                cells.append(_pct_cell(base_val, val))
        table_rows.append(cells)

    return _render_table(headers, table_rows, title)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Parse CLI arguments, load metrics files, and write comparison tables.

    Args:
        None: All parameters come from sys.argv via argparse.

    Returns:
        None: Writes .txt files to output_dir as side effects.

    Example:
        >>> # python src/analysis/compare_metrics.py \\
        >>> #     --experiments AFNO_BoB_Surf_E00 AFNO_BoB_Surf_E11 \\
        >>> #     --leads 1 5 9
    """
    parser = argparse.ArgumentParser(
        description='Compare forecast metrics across experiments'
    )
    parser.add_argument('--experiments', nargs='+', required=True,
                        help='Experiment names; first is the baseline')
    parser.add_argument('--results_dir', default='results/',
                        help='Root results directory (default: results/)')
    parser.add_argument('--leads', nargs='+', type=int, default=[1, 5, 9],
                        help='Lead days to include as columns (default: 1 5 9)')
    parser.add_argument('--metric', default='rmse',
                        choices=list(METRIC_LABEL),
                        help='Metric to tabulate (default: rmse)')
    parser.add_argument('--output_dir', default='results/tables/',
                        help='Output directory for .txt files (default: results/tables/)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_label = METRIC_LABEL[args.metric]

    for filename, tag in FILE_TAGS:
        data_per_exp = {}
        for exp in args.experiments:
            path = results_dir / exp / filename
            if path.exists():
                data_per_exp[exp] = parse_metrics_file(path, args.metric)
            else:
                print(f'  [skip] {path} not found')

        present = [e for e in args.experiments if e in data_per_exp]
        if len(present) < 2:
            print(f'\n[skip] fewer than 2 experiments have {filename}\n')
            continue

        sep = '=' * 72
        print(f'\n{sep}')
        print(f'{metric_label} comparison — {tag}')
        print(sep)

        lead_table = build_lead_table(
            data_per_exp, present, args.leads, VARIABLES,
            title=f'{metric_label} by lead time ({tag})',
        )
        summary_table = build_summary_table(
            data_per_exp, present, VARIABLES,
            title=f'{metric_label} averaged across all leads ({tag})',
        )

        output = f'{lead_table}\n\n{summary_table}\n'
        print(output)

        out_path = output_dir / f'compare_{args.metric}_{tag}.txt'
        out_path.write_text(output, encoding='utf-8')
        print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()

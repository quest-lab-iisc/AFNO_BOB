"""Compute seasonal RMSE and Pearson CC metrics and write a LaTeX results table.

Runs autoregressive 9-day forecasts over the full 2020-2025 test period, groups
IC dates by Bay-of-Bengal season, and reports RMSE and Pearson correlation
coefficient at lead days k=1,3,5,7,9 for all five ocean variables.

Season definitions (by IC date month):
  Winter      (DJF)  : Dec, Jan, Feb
  Pre-monsoon (MAM)  : Mar, Apr, May
  Monsoon     (JJAS) : Jun, Jul, Aug, Sep
  Post-monsoon (ON)  : Oct, Nov

Inputs:
    --config_file (str)     : Config YAML in config/ (default: afno_bob_config.yaml).
    --name (str)            : Experiment name override.
    --model_path (str)      : Path to .pth weights (default: from config).
    --extra_data_dir (str)  : Directory with ocean_2021_2025.nc / atm_2021_2025.nc.
    --extra_start_date (str): First evaluation date in extra data (dd-mm-yyyy).
    --extra_num_days (int)  : Days to evaluate from extra data (default 1826).
    --extra_ref_date (str)  : Index-0 date for extra NetCDF (dd-mm-yyyy).
    --extra_ocean_file (str): File prefix for extra ocean NetCDF.
    --extra_atm_file (str)  : File prefix for extra atm NetCDF.
    --output_dir (str)      : Directory for outputs (default: results/<name>).
    --device (str)          : PyTorch device (default: from config).

Outputs:
    <output_dir>/seasonal_metrics_table.tex  LaTeX tabular source.
    <output_dir>/seasonal_metrics.csv        Raw seasonal × lead × variable CSV.
    logs/seasonal_metrics_table.log          Verbatim console log.

Example:
    conda activate BoB_Surf_2
    python src/inference/run_seasonal_metrics_table.py \\
        --name AFNO_BoB_Surf_E11p1 \\
        --extra_data_dir data/2021_2025 \\
        --combined_output
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from configmypy import ConfigPipeline, YamlConfig
from inference.run_metrics import (
    load_model, date_to_day_index,
    run_autoregressive_forecast, postprocess_forecasts,
    load_ground_truth_batch, compute_forecast_metrics,
)
from training.utils.experiment_logger import TeeLogger, cleanup_logging


# ---------------------------------------------------------------------------
# Season assignment
# ---------------------------------------------------------------------------

SEASON_MAP: dict[int, str] = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Pre-monsoon', 4: 'Pre-monsoon', 5: 'Pre-monsoon',
    6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon',
    10: 'Post-monsoon', 11: 'Post-monsoon',
}

SEASON_ORDER = ['Winter', 'Pre-monsoon', 'Monsoon', 'Post-monsoon']

SEASON_LABELS = {
    'Winter':      r'Winter (DJF)',
    'Pre-monsoon': r'Pre-monsoon (MAM)',
    'Monsoon':     r'Monsoon (JJAS)',
    'Post-monsoon': r'Post-monsoon (ON)',
}

VAR_LABELS = {
    'thetao': 'SST',
    'so':     'SSS',
    'zos':    'SSH',
    'uo':     'Zonal current',
    'vo':     'Meridional current',
}

VAR_ORDER = ['thetao', 'so', 'zos', 'uo', 'vo']

TABLE_LEADS = [1, 3, 5, 7, 9]


# ---------------------------------------------------------------------------
# Seasonal evaluation loop
# ---------------------------------------------------------------------------

def run_seasonal_evaluation(config, model, device,
                             ocean_data: xr.Dataset, atm_data: xr.Dataset,
                             start_date: str, num_days: int,
                             reference_date: str,
                             seasonal_raw: dict,
                             desc: str = 'Evaluating') -> int:
    """Run the autoregressive evaluation loop and accumulate metrics by season.

    For each IC date the function determines the BoB season from the calendar
    month, runs a 9-day forecast, computes per-day RMSE and Pearson CC, and
    appends the result to ``seasonal_raw[season][lead][var]``.

    Args:
        config: Configuration object (configmypy).
        model (torch.nn.Module): Trained AFNO model in eval mode.
        device (torch.device): Compute device.
        ocean_data (xr.Dataset): Ocean NetCDF dataset.
        atm_data (xr.Dataset): Atmospheric NetCDF dataset.
        start_date (str): First IC date (dd-mm-yyyy).
        num_days (int): Evaluation window length in days.
        reference_date (str): Calendar date at index 0 in the datasets (dd-mm-yyyy).
        seasonal_raw (dict): Accumulator: season → lead → var → list[metric_dict].
            Pass the same dict across multiple calls to pool 2020 and 2021-2025.
        desc (str): tqdm label.

    Returns:
        int: Number of IC dates evaluated.

    Example:
        >>> acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        >>> n = run_seasonal_evaluation(cfg, mdl, dev, oc, at,
        ...     '01-01-2020', 366, '01-01-2020', acc)
    """
    max_lead      = config.evaluation.max_forecast_days
    start_idx     = date_to_day_index(start_date, reference_date)
    dataset_size  = len(ocean_data.time)
    num_eval_days = min(num_days - max_lead + 1,
                        dataset_size - start_idx - max_lead)

    ref_dt = datetime.strptime(reference_date, '%d-%m-%Y')
    n_evaluated = 0

    for eval_day in tqdm(range(num_eval_days), desc=desc, file=sys.stdout,
                          dynamic_ncols=True):
        current_idx = start_idx + eval_day
        ic_date     = ref_dt + timedelta(days=current_idx)
        season      = SEASON_MAP[ic_date.month]

        try:
            forecast_dict, mean_dict = run_autoregressive_forecast(
                config, model, current_idx, max_lead, device, ocean_data, atm_data)
        except ValueError as exc:
            tqdm.write(f'  Skipping {ic_date.date()}: {exc}')
            continue

        postprocessed = postprocess_forecasts(forecast_dict, mean_dict, config)
        ground_truth  = load_ground_truth_batch(
            config, current_idx, max_lead, ocean_data)
        metrics_dict  = compute_forecast_metrics(postprocessed, ground_truth, config)

        for lead in range(1, max_lead + 1):
            for var in config.data.out_variable:
                seasonal_raw[season][lead][var].append(metrics_dict[lead][var])

        n_evaluated += 1

    return n_evaluated


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_seasonal(seasonal_raw: dict, variables: list[str]) -> dict:
    """Average raw per-day metrics within each season × lead × variable cell.

    Args:
        seasonal_raw (dict): season → lead → var → list[{rmse, pearson, ...}].
        variables (list[str]): Variable names to aggregate.

    Returns:
        dict: season → lead → var → {'rmse': float, 'pearson': float, 'n': int}.

    Example:
        >>> agg = aggregate_seasonal(seasonal_raw, ['thetao', 'so'])
        >>> agg['Winter'][1]['thetao']['rmse']
        0.147
    """
    result: dict = {}
    for season, lead_dict in seasonal_raw.items():
        result[season] = {}
        for lead, var_dict in lead_dict.items():
            result[season][lead] = {}
            for var in variables:
                ms = var_dict.get(var, [])
                if ms:
                    result[season][lead][var] = {
                        'rmse':    float(np.mean([m['rmse']    for m in ms])),
                        'pearson': float(np.mean([m['pearson'] for m in ms])),
                        'n':       len(ms),
                    }
                else:
                    result[season][lead][var] = {'rmse': float('nan'),
                                                  'pearson': float('nan'), 'n': 0}
    return result


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_seasonal_csv(agg: dict, variables: list[str], csv_path: Path) -> None:
    """Write season × lead × variable metrics to a flat CSV file.

    Args:
        agg (dict): Output of aggregate_seasonal().
        variables (list[str]): Variable names.
        csv_path (Path): Destination CSV path.

    Returns:
        None: Writes CSV to csv_path.

    Example:
        >>> save_seasonal_csv(agg, ['thetao', 'so'], Path('results/seasonal.csv'))
    """
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['season', 'lead_day', 'variable', 'rmse', 'pearson', 'n'])
        for season in SEASON_ORDER:
            if season not in agg:
                continue
            for lead in sorted(agg[season]):
                for var in variables:
                    m = agg[season][lead].get(var, {})
                    writer.writerow([
                        season, lead, var,
                        f"{m.get('rmse', float('nan')):.6f}",
                        f"{m.get('pearson', float('nan')):.6f}",
                        m.get('n', 0),
                    ])
    print(f'Saved: {csv_path}')


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def write_latex_table(agg: dict, variables: list[str],
                      tex_path: Path, caption: str, label: str) -> None:
    """Write the seasonal evaluation results as a LaTeX table.

    Produces a publication-ready ``tabular`` environment matching the format
    with RMSE and CC (Pearson) at lead days k=1,3,5,7,9 for each season column.

    Args:
        agg (dict): Output of aggregate_seasonal().
        variables (list[str]): Variables in display order.
        tex_path (Path): Destination .tex file path.
        caption (str): LaTeX table caption.
        label (str): LaTeX label string.

    Returns:
        None: Writes .tex to tex_path.

    Example:
        >>> write_latex_table(agg, ['thetao','so','zos','uo','vo'],
        ...     Path('results/table.tex'), 'caption text', 'tab:seasonal')
    """
    def cell(val: float, fmt: str = '.3f') -> str:
        """Format a float for a table cell, or --- if NaN."""
        return f'{val:{fmt}}' if not np.isnan(val) else r'---'

    lines: list[str] = []
    lines.append(r'\begin{table}[ht]')
    lines.append(f'  \\caption{{{caption}}}')
    lines.append(f'  \\label{{{label}}}')
    lines.append(r'  \centering')
    lines.append(r'  \resizebox{\textwidth}{!}')
    lines.append(r'  {')
    lines.append(r'  \begin{tabular}{c|c|cc|cc|cc|cc}')
    lines.append(r'    \toprule')

    # Season header row — \textbf on season labels, \textbf{Variable} in col-2 header
    season_cols = ' & '.join(
        rf'\multicolumn{{2}}{{{"c|" if i < 3 else "c"}}}{{\textbf{{{SEASON_LABELS[s]}}}}}'
        for i, s in enumerate(SEASON_ORDER)
    )
    lines.append(f'    & \\textbf{{Variable}} & {season_cols} \\\\')
    lines.append(r'    \midrule')

    # Metric sub-header
    metric_header = ' & '.join(
        r'RMSE & $r$' for _ in SEASON_ORDER
    )
    lines.append(
        r'    \textbf{Lead day} & & ' + metric_header + r' \\'
    )
    lines.append(r'    \midrule')

    for i_lead, lead in enumerate(TABLE_LEADS):
        n_vars = len(variables)
        for i_var, var in enumerate(variables):
            label_var  = VAR_LABELS.get(var, var)
            first_cell = (rf'    \multirow{{{n_vars}}}{{*}}{{\centering $k={lead}$}}'
                          if i_var == 0 else '   ')

            cells = []
            for season in SEASON_ORDER:
                m = agg.get(season, {}).get(lead, {}).get(var, {})
                rmse_v = m.get('rmse',    float('nan'))
                cc_v   = m.get('pearson', float('nan'))
                cells.append(f'{cell(rmse_v)} & {cell(cc_v)}')

            lines.append(
                f'{first_cell} & {label_var} & ' + ' & '.join(cells) + r' \\'
            )

        if i_lead < len(TABLE_LEADS) - 1:
            lines.append(r'    \midrule')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}}')
    lines.append(r'\end{table}')

    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text('\n'.join(lines) + '\n')
    print(f'Saved: {tex_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments.

    Example:
        >>> args = parse_args()
    """
    p = argparse.ArgumentParser(
        description='Seasonal RMSE/CC metrics and LaTeX table for BoB_Surf')
    p.add_argument('--config_file',       default='afno_bob_config.yaml')
    p.add_argument('--name',              default=None)
    p.add_argument('--model_path',        default=None)
    p.add_argument('--extra_data_dir',    default=None)
    p.add_argument('--extra_start_date',  default='01-01-2021')
    p.add_argument('--extra_num_days',    type=int, default=1826)
    p.add_argument('--extra_ref_date',    default='01-01-2021')
    p.add_argument('--extra_ocean_file',  default='ocean_2021_2025')
    p.add_argument('--extra_atm_file',    default='atm_2021_2025')
    p.add_argument('--output_dir',        default=None)
    p.add_argument('--device',            default=None)
    p.add_argument('--from_csv',          default=None,
                   help='Skip inference; regenerate table from an existing seasonal_metrics.csv')
    p.add_argument(
        '--caption',
        default=(
            'Seasonal evaluation of AFNO RT forecast skill across the '
            '2020--2025 test period. Metrics are RMSE and Pearson correlation '
            '($r$) at lead days $k = 1, 3, 5, 7, 9$ for four Bay-of-Bengal '
            'seasons: Winter (DJF), Pre-monsoon (MAM), Monsoon (JJAS), and '
            'Post-monsoon (ON).'
        ),
    )
    p.add_argument('--label', default='tab:seasonal_evaluation_afno')
    return p.parse_args()


def main() -> None:
    """Entry point: load model, run seasonal evaluation, write table.

    Example:
        conda activate BoB_Surf_2
        python src/inference/run_seasonal_metrics_table.py \\
            --name AFNO_BoB_Surf_E11p1 \\
            --extra_data_dir data/2021_2025
    """
    args = parse_args()

    # --- Config ---
    pipe = ConfigPipeline([
        YamlConfig(f'./{args.config_file}', config_name='default',
                   config_folder='config/'),
        YamlConfig(config_folder='config/'),
    ])
    config = pipe.read_conf()
    if args.name:
        config.name = args.name
    if args.device:
        config.device = args.device

    out_dir = Path(args.output_dir or f'results/{config.name}')
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Logging ---
    Path('logs').mkdir(exist_ok=True)
    tee = TeeLogger(f'logs/seasonal_metrics_table_{config.name}.log')
    sys.stdout = tee

    try:
        # ── Fast path: regenerate table from existing CSV ──────────────────
        if args.from_csv:
            csv_path = Path(args.from_csv)
            if not csv_path.exists():
                raise FileNotFoundError(f'CSV not found: {csv_path}')
            seasonal_raw: dict = defaultdict(
                lambda: defaultdict(lambda: defaultdict(list)))
            with open(csv_path, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    season = row['season']
                    lead   = int(row['lead_day'])
                    var    = row['variable']
                    # store as single-element list so aggregate_seasonal works
                    seasonal_raw[season][lead][var].append({
                        'rmse':    float(row['rmse']),
                        'pearson': float(row['pearson']),
                    })
            variables = [v for v in VAR_ORDER if v in
                         {r['variable'] for r in csv.DictReader(open(csv_path))}]
            agg = aggregate_seasonal(seasonal_raw, variables)
            out_dir = Path(args.output_dir or csv_path.parent)
            out_dir.mkdir(parents=True, exist_ok=True)
            write_latex_table(
                agg, variables,
                out_dir / 'seasonal_metrics_table.tex',
                args.caption, args.label,
            )
            print(f'Table regenerated from {csv_path}')
            return

        device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        print(f'Device: {device}')

        model_path = (args.model_path
                      or str(Path(config.results.model_dir) / f'{config.name}.pth'))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model not found: {model_path}')
        model = load_model(config, model_path, device)
        print(f'Model loaded from {model_path}')

        data_dir   = config.data.data_dir
        ocean_file = Path(data_dir) / f'{config.data.file_prefix}.nc'
        atm_file   = Path(data_dir) / f'{config.data.file_prefix_atm}.nc'
        ocean_data = xr.open_dataset(ocean_file)
        atm_data   = xr.open_dataset(atm_file)

        # Accumulator: season → lead → var → [metric_dicts]
        seasonal_raw: dict = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list)))

        # --- Primary period (2020) ---
        print(f'\n=== Primary evaluation: {config.evaluation.start_date} ===')
        n1 = run_seasonal_evaluation(
            config, model, device, ocean_data, atm_data,
            config.evaluation.start_date,
            config.evaluation.num_days,
            config.evaluation.reference_date,
            seasonal_raw,
            desc='2020',
        )
        print(f'  Evaluated {n1} IC dates')
        ocean_data.close()
        atm_data.close()

        # --- Extended period (2021-2025) ---
        if args.extra_data_dir:
            extra_dir   = Path(args.extra_data_dir)
            extra_ocean = xr.open_dataset(
                extra_dir / f'{args.extra_ocean_file}.nc')
            extra_atm   = xr.open_dataset(
                extra_dir / f'{args.extra_atm_file}.nc')

            print(f'\n=== Extended evaluation: {args.extra_start_date} '
                  f'({args.extra_num_days} days) ===')
            n2 = run_seasonal_evaluation(
                config, model, device, extra_ocean, extra_atm,
                args.extra_start_date,
                args.extra_num_days,
                args.extra_ref_date,
                seasonal_raw,
                desc='2021-2025',
            )
            print(f'  Evaluated {n2} IC dates')
            extra_ocean.close()
            extra_atm.close()

        # --- Aggregate --- enforce canonical display order
        config_vars = set(config.data.out_variable)
        variables   = [v for v in VAR_ORDER if v in config_vars]
        agg = aggregate_seasonal(seasonal_raw, variables)

        # Print summary table to console
        print('\n=== Seasonal metrics summary (leads 1,3,5,7,9) ===')
        header = f"{'Season':<15} {'Lead':>4} {'Var':<12} {'RMSE':>8} {'CC':>8} {'n':>6}"
        print(header)
        print('-' * len(header))
        for season in SEASON_ORDER:
            for lead in TABLE_LEADS:
                for var in variables:
                    m = agg.get(season, {}).get(lead, {}).get(var, {})
                    print(f"{season:<15} {lead:>4}  {VAR_LABELS.get(var,var):<12}"
                          f"  {m.get('rmse', float('nan')):>8.4f}"
                          f"  {m.get('pearson', float('nan')):>8.4f}"
                          f"  {m.get('n', 0):>6}")

        # --- Save outputs ---
        save_seasonal_csv(agg, variables, out_dir / 'seasonal_metrics.csv')
        write_latex_table(
            agg, variables,
            out_dir / 'seasonal_metrics_table.tex',
            args.caption,
            args.label,
        )

        print(f'\nAll outputs written to {out_dir}/')

    except Exception as exc:
        import traceback
        print(f'\n[FATAL] {exc}')
        traceback.print_exc()
    finally:
        cleanup_logging(tee)


if __name__ == '__main__':
    main()

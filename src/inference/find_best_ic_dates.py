"""Find initial-condition dates with the lowest 9-day RMSE per season (2020-2025).

For every valid IC date in the test period, runs a 9-step autoregressive
rollout with AFNO RT and records per-variable RMSE at lead day +9.  RMSEs
are normalised by the seasonal reference (mean +9-day RMSE from an existing
seasonal_metrics.csv) so that all five variables contribute equally regardless
of physical units.  The combined score is the mean normalised RMSE across all
five variables.  Results are saved to a CSV and the best IC per season is
reported to stdout.

Season definitions (by IC date month):
    Winter      (DJF)  : Dec, Jan, Feb
    Pre-monsoon (MAM)  : Mar, Apr, May
    Monsoon     (JJAS) : Jun, Jul, Aug, Sep
    Post-monsoon (ON)  : Oct, Nov

Corruption guard: IC dates whose 9-step rollout encounters an input tensor
with abs-max > 1e4 are skipped automatically.

Inputs:
    --config_file (str)    : YAML config in config/ (default: afno_bob_surf_e14.yaml).
    --model_path (str)     : Path to .pth weights (default: results/models/{name}.pth).
    --ref_csv (str)        : Path to seasonal_metrics.csv used as normalisation
                             reference (default: results/AFNO_BoB_Surf_E14/seasonal_metrics.csv).
    --data_dir (str)       : Primary dataset directory (default: from config).
    --extra_data_dir (str) : 2021-2025 dataset directory (default:
                             data/2021_2025).
    --extra_num_days (int) : IC dates to scan in extended period (default: 1711).
    --output_dir (str)     : Directory to write best_ic_dates.csv (default:
                             results/AFNO_BoB_Surf_E14/).
    --top_n (int)          : Report the top-N best ICs per season (default: 5).
    --device (str)         : PyTorch device (default: from config).

Outputs:
    <output_dir>/best_ic_dates.csv  — full per-IC search results with scores.
    logs/find_best_ic_dates.log     — verbatim console log.

Example:
    conda activate BoB_Surf_2
    python src/inference/find_best_ic_dates.py \\
        --config_file afno_bob_surf_e14.yaml \\
        --model_path results/models/AFNO_BoB_Surf_E14.pth \\
        --extra_data_dir data/2021_2025 \\
        --device cuda:1
"""

import argparse
import csv
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
# Constants
# ---------------------------------------------------------------------------

SEASON_MAP: dict[int, str] = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Pre-monsoon', 4: 'Pre-monsoon', 5: 'Pre-monsoon',
    6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon',
    10: 'Post-monsoon', 11: 'Post-monsoon',
}

SEASON_ORDER = ['Winter', 'Pre-monsoon', 'Monsoon', 'Post-monsoon']

VAR_ORDER = ['thetao', 'so', 'zos', 'uo', 'vo']

VAR_LABELS = {
    'thetao': 'SST',
    'so':     'SSS',
    'zos':    'SSH',
    'uo':     'Zonal current',
    'vo':     'Meridional current',
}

LEAD_9 = 9   # lead day we optimise at

PRIMARY_REF   = '01-01-1993'
EXTENDED_REF  = '01-01-2021'


# ---------------------------------------------------------------------------
# Load reference normalisation from seasonal_metrics.csv
# ---------------------------------------------------------------------------

def load_reference_rmse(csv_path: Path) -> dict[str, dict[str, float]]:
    """Load seasonal mean RMSE at lead day 9 for each variable.

    Args:
        csv_path (Path): Path to seasonal_metrics.csv produced by
            run_seasonal_metrics_table.py.

    Returns:
        dict[str, dict[str, float]]: ref[season][var] = mean +9d RMSE.

    Example:
        >>> ref = load_reference_rmse(Path('results/.../seasonal_metrics.csv'))
        >>> ref['Winter']['thetao']
        0.3747
    """
    ref: dict[str, dict[str, float]] = defaultdict(dict)
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            if int(row['lead_day']) == LEAD_9:
                ref[row['season']][row['variable']] = float(row['rmse'])
    return ref


# ---------------------------------------------------------------------------
# Single-IC evaluation
# ---------------------------------------------------------------------------

def eval_ic(config, model, day_idx: int, ocean_data, atm_data, device
            ) -> dict[str, float] | None:
    """Run a 9-step rollout and return per-variable RMSE at lead +9, or None on corruption.

    Args:
        config: configmypy configuration object.
        model (torch.nn.Module): Loaded AFNO model in eval mode.
        day_idx (int): Integer index into ocean_data / atm_data for the IC.
        ocean_data (xr.Dataset): Ocean dataset.
        atm_data (xr.Dataset): Atmospheric dataset.
        device (torch.device): Compute device.

    Returns:
        dict[str, float] | None: Per-variable RMSE at lead +9 (keys = variable
            names), or None if a corruption guard fires.

    Example:
        >>> rmse9 = eval_ic(config, model, 9876, ocean_ds, atm_ds, device)
    """
    try:
        forecast_dict, mean_dict = run_autoregressive_forecast(
            config, model, day_idx, LEAD_9, device, ocean_data, atm_data)
    except ValueError:
        return None

    preds  = postprocess_forecasts(forecast_dict, mean_dict, config)
    truths = load_ground_truth_batch(config, day_idx, LEAD_9, ocean_data)

    # Compute RMSE at exactly lead +9 (both dicts are 1-indexed by lead_time)
    rmse9 = {}
    for var in config.data.out_variable:
        pred_field  = preds[LEAD_9][var]
        truth_field = truths[LEAD_9][var]
        valid = ~np.isnan(truth_field)
        if valid.sum() == 0:
            rmse9[var] = float('nan')
        else:
            rmse9[var] = float(np.sqrt(np.mean((pred_field[valid] - truth_field[valid]) ** 2)))

    return rmse9


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

def run_search(config, model, device, ref_rmse,
               primary_data, extended_data,
               extra_num_days, output_dir, top_n):
    """Iterate over all IC dates and rank by combined normalised RMSE at +9d.

    Args:
        config: configmypy configuration object.
        model (torch.nn.Module): AFNO RT model in eval mode.
        device (torch.device): Compute device.
        ref_rmse (dict): Seasonal reference RMSEs from load_reference_rmse.
        primary_data (tuple): (ocean_ds, atm_ds) for 2020.
        extended_data (tuple | None): (ocean_ds, atm_ds) for 2021-2025,
            or None if not provided.
        extra_num_days (int): Number of IC dates to scan in extended period.
        output_dir (Path): Where to write best_ic_dates.csv.
        top_n (int): Report this many best ICs per season.

    Returns:
        dict[str, list[dict]]: Season → sorted list of IC result dicts.

    Example:
        >>> ranked = run_search(config, model, device, ref, (o, a), (oe, ae),
        ...                     1711, Path('results/...'), 5)
    """
    ocean_2020, atm_2020 = primary_data

    # Build list of (date, day_idx, ocean_ds, atm_ds, ref_date) for every IC
    ic_list = []

    # 2020: days 0 … 356 (indices 9922+), stop 9 before end so +9 is valid
    start_idx_2020 = date_to_day_index('01-01-2020', PRIMARY_REF)
    end_idx_2020   = date_to_day_index('31-12-2020', PRIMARY_REF)
    for idx in range(start_idx_2020, end_idx_2020 - LEAD_9 + 1):
        d = datetime(1993, 1, 1) + timedelta(days=idx)
        ic_list.append((d, idx, ocean_2020, atm_2020))

    # 2021-2025 extended
    if extended_data is not None:
        ocean_ext, atm_ext = extended_data
        for rel_idx in range(extra_num_days):
            d = datetime(2021, 1, 1) + timedelta(days=rel_idx)
            ic_list.append((d, rel_idx, ocean_ext, atm_ext))

    tqdm.write(f'Total IC dates to evaluate: {len(ic_list)}')

    # --- run search ---
    rows = []
    by_season: dict[str, list] = defaultdict(list)

    for d, idx, o_ds, a_ds in tqdm(ic_list, desc='IC search', unit='day', file=sys.stdout):
        season = SEASON_MAP[d.month]
        rmse9  = eval_ic(config, model, idx, o_ds, a_ds, device)
        if rmse9 is None:
            tqdm.write(f'  Skipping {d.date()}: corrupted input')
            continue

        # Combined score: mean normalised RMSE across variables available in ref
        norm_scores = []
        for var in VAR_ORDER:
            if var not in rmse9:
                continue
            ref_val = ref_rmse.get(season, {}).get(var, None)
            if ref_val and ref_val > 0:
                norm_scores.append(rmse9[var] / ref_val)
        combined = float(np.mean(norm_scores)) if norm_scores else float('nan')

        row = {
            'date':     d.strftime('%Y-%m-%d'),
            'season':   season,
            'combined': combined,
        }
        for var in VAR_ORDER:
            row[f'rmse_{var}'] = rmse9.get(var, float('nan'))
        rows.append(row)
        by_season[season].append(row)

    # --- write full CSV ---
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'best_ic_dates.csv'
    fieldnames = ['date', 'season', 'combined'] + [f'rmse_{v}' for v in VAR_ORDER]
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: f'{v:.6f}' if isinstance(v, float) else v
                        for k, v in row.items()})
    tqdm.write(f'\nSaved full results: {csv_path}')

    # --- rank and report top-N per season ---
    ranked: dict[str, list] = {}
    print('\n' + '=' * 72)
    print(f'TOP-{top_n} BEST IC DATES BY SEASON  (lowest combined normalised RMSE at +9d)')
    print('=' * 72)
    for season in SEASON_ORDER:
        season_rows = sorted(by_season[season], key=lambda r: r['combined'])
        ranked[season] = season_rows[:top_n]
        print(f'\n{season}:')
        hdr = f"  {'Date':<12}  {'Combined':>9}  " + '  '.join(
            f'{VAR_LABELS[v]:>18}' for v in VAR_ORDER)
        print(hdr)
        print('  ' + '-' * (len(hdr) - 2))
        for r in season_rows[:top_n]:
            vals = '  '.join(
                f'{r[f"rmse_{v}"]:>18.4f}' for v in VAR_ORDER)
            print(f"  {r['date']:<12}  {r['combined']:>9.4f}  {vals}")

    print('\n' + '=' * 72)
    print('SINGLE BEST IC PER SEASON:')
    for season in SEASON_ORDER:
        if ranked.get(season):
            best = ranked[season][0]
            print(f'  {season:<15}: {best["date"]}  (combined score = {best["combined"]:.4f})')
    print('=' * 72)

    return ranked


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments.

    Example:
        >>> args = parse_args()
    """
    p = argparse.ArgumentParser(description='Find best IC dates per season for AFNO RT')
    p.add_argument('--config_file',    default='afno_bob_surf_e14.yaml')
    p.add_argument('--model_path',     default=None)
    p.add_argument('--ref_csv',        default='results/AFNO_BoB_Surf_E14/seasonal_metrics.csv')
    p.add_argument('--data_dir',       default=None,
                   help='Primary dataset dir (default: from config)')
    p.add_argument('--extra_data_dir', default='data/2021_2025')
    p.add_argument('--extra_num_days', type=int, default=1711)
    p.add_argument('--output_dir',     default='results/AFNO_BoB_Surf_E14')
    p.add_argument('--top_n',          type=int, default=5)
    p.add_argument('--device',         default=None)
    return p.parse_args()


def main():
    """Entry point: load everything, run IC search, print and save results.

    Example:
        >>> # python src/inference/find_best_ic_dates.py --device cuda:1
    """
    args = parse_args()

    pipe   = ConfigPipeline([YamlConfig(args.config_file, config_name='default',
                                        config_folder='config/')])
    config = pipe.read_conf()
    if args.model_path:
        config.results.model_dir = str(Path(args.model_path).parent)
        config.name              = Path(args.model_path).stem

    device_str = args.device or config.device
    device = torch.device(device_str if (torch.cuda.is_available()
                                         or 'cpu' in device_str) else 'cpu')

    log_path = Path('logs') / 'find_best_ic_dates.log'
    log_path.parent.mkdir(exist_ok=True)
    tee = TeeLogger(str(log_path))
    sys.stdout = tee
    sys.stderr = tee

    try:
        print(f'Device     : {device}')
        print(f'Config     : {args.config_file}')
        print(f'Ref CSV    : {args.ref_csv}')

        model_path = (args.model_path or
                      f'{config.results.model_dir}/{config.name}.pth')
        print(f'Model      : {model_path}')
        model = load_model(config, model_path, device)

        ref_rmse = load_reference_rmse(Path(args.ref_csv))
        print(f'Reference seasons loaded: {list(ref_rmse.keys())}')

        data_dir  = Path(args.data_dir or config.data.data_dir)
        ocean_2020 = xr.open_dataset(data_dir / f'{config.data.file_prefix}.nc')
        atm_2020   = xr.open_dataset(data_dir / f'{config.data.file_prefix_atm}.nc')
        print(f'Primary data: {data_dir}')

        extended_data = None
        if args.extra_data_dir:
            ext_dir    = Path(args.extra_data_dir)
            ocean_ext  = xr.open_dataset(ext_dir / 'ocean_2021_2025.nc')
            atm_ext    = xr.open_dataset(ext_dir / 'atm_2021_2025.nc')
            extended_data = (ocean_ext, atm_ext)
            print(f'Extended data: {ext_dir}  ({args.extra_num_days} IC days)')

        ranked = run_search(
            config, model, device, ref_rmse,
            (ocean_2020, atm_2020), extended_data,
            args.extra_num_days,
            Path(args.output_dir), args.top_n,
        )

    except Exception as exc:
        print(f'\nERROR: {exc}')
        raise
    finally:
        cleanup_logging(tee)


if __name__ == '__main__':
    main()

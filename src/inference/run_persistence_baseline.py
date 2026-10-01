"""Compute persistence baseline metrics over the 2020-2025 test period.

The persistence forecast simply repeats the initial ocean state (IC) as the
prediction for every lead day 1–9.  This provides a lower-bound benchmark:
any model that cannot beat persistence at short lead times offers no skill.

Inputs:
    --config_file (str): YAML config in config/ (default: afno_bob_surf_e11p1.yaml).
    --extra_data_dir (str): Directory with 2021-2025 ocean/atm NetCDF files.
    --extra_start_date (str): First evaluation date in extra data (dd-mm-yyyy).
    --extra_num_days (int): Days to evaluate from extra data (default 1826).
    --extra_ref_date (str): Index-0 date for extra NetCDF (dd-mm-yyyy).
    --extra_ocean_file (str): File prefix for extra ocean NetCDF.
    --extra_atm_file (str): File prefix for extra atm NetCDF.
    --combined_output (flag): Pool 2020 and 2021-2025 metrics.
    --output_dir (str): Output directory (default: results/Persistence_Baseline).

Outputs:
    <output_dir>/forecast_metrics.txt           2020 metrics.
    <output_dir>/forecast_metrics_extended.txt  2021-2025 metrics (if extra).
    <output_dir>/forecast_metrics_combined.txt  Pooled metrics (if combined).
    logs/persistence_baseline.log               Verbatim console log.

Example:
    conda activate BoB_Surf_2
    python src/inference/run_persistence_baseline.py \\
        --extra_data_dir data/2021_2025 \\
        --combined_output
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from configmypy import ConfigPipeline, YamlConfig
from data_pipeline.preprocessing.transformers.normalize_transform import PreprocessTransform
from inference.run_metrics import (
    load_normalization_stats, date_to_day_index,
    load_initial_ocean_state, load_ground_truth_batch,
    postprocess_ocean_variable, compute_metrics,
    average_raw_metrics, save_forecast_metrics,
)
from training.utils.experiment_logger import TeeLogger, cleanup_logging

N_NORTH_ROWS = 20


def run_persistence_evaluation(config, ocean_data, atm_data,
                                start_date: str, num_days: int,
                                reference_date: str,
                                desc: str = 'Persistence') -> tuple[dict, dict, int]:
    """Evaluate the persistence forecast over a date range.

    For each IC date, the ocean state at time t is used as the prediction for
    t+1, t+2, ..., t+9.  RMSE, MAE, R², and Pearson CC are computed against
    the true ocean state at each lead day.

    Args:
        config: Configuration namespace.
        ocean_data: xarray.Dataset with ocean variables.
        atm_data: xarray.Dataset (unused — included for API consistency).
        start_date (str): First IC date (dd-mm-yyyy).
        num_days (int): Evaluation window length in days.
        reference_date (str): Index-0 date in the dataset (dd-mm-yyyy).
        desc (str): tqdm progress bar label.

    Returns:
        tuple:
            averaged (dict): lead → var → {rmse, mae, r2, pearson}.
            raw (dict): lead → var → list[metric_dict].
            num_eval_days (int): Number of IC dates evaluated.

    Example:
        >>> avg, raw, n = run_persistence_evaluation(cfg, oc, at, '01-01-2020', 365,
        ...                                          '01-01-2020')
        >>> avg[1]['thetao']['rmse']
        0.09...
    """
    max_lead     = config.evaluation.max_forecast_days
    start_idx    = date_to_day_index(start_date, reference_date)
    dataset_size = len(ocean_data.time)
    num_eval     = min(num_days - max_lead + 1,
                       dataset_size - start_idx - max_lead)

    mean_dict, _ = load_normalization_stats(config)
    transform    = PreprocessTransform(config)

    raw = {lt: {var: [] for var in config.data.out_variable}
           for lt in range(1, max_lead + 1)}

    for eval_day in tqdm(range(num_eval), desc=desc, file=sys.stdout,
                          dynamic_ncols=True):
        idx = start_idx + eval_day

        # Load and postprocess the IC ocean state for all variables
        ocean_state = load_initial_ocean_state(
            config, idx, ocean_data, transform, mean_dict)

        ic_pp = {}
        for var in config.data.out_variable:
            arr_224 = ocean_state[var].squeeze().numpy()  # (224, 224) normalised
            pp = postprocess_ocean_variable(arr_224, mean_dict[var], var)
            pp[-N_NORTH_ROWS:, :] = np.nan
            ic_pp[var] = pp

        # Build persistence forecast: same IC repeated for all lead days
        persist_fc = {lt: ic_pp for lt in range(1, max_lead + 1)}

        # Load ground truth
        gt = load_ground_truth_batch(config, idx, max_lead, ocean_data)

        # Compute metrics per lead
        for lt in range(1, max_lead + 1):
            for var in config.data.out_variable:
                pred  = persist_fc[lt][var]
                truth = gt[lt][var]
                raw[lt][var].append(compute_metrics(pred, truth))

    averaged = average_raw_metrics(raw, config.data.out_variable, max_lead)
    return averaged, raw, num_eval


def main() -> None:
    """Entry point: run persistence evaluation and write metrics files.

    Example:
        python src/inference/run_persistence_baseline.py \\
            --extra_data_dir data/2021_2025 \\
            --combined_output
    """
    parser = argparse.ArgumentParser(description='Persistence baseline evaluation')
    parser.add_argument('--config_file',      default='afno_bob_surf_e11p1.yaml')
    parser.add_argument('--extra_data_dir',   default=None)
    parser.add_argument('--extra_start_date', default='01-01-2021')
    parser.add_argument('--extra_num_days',   type=int, default=1826)
    parser.add_argument('--extra_ref_date',   default='01-01-2021')
    parser.add_argument('--extra_ocean_file', default='ocean_2021_2025')
    parser.add_argument('--extra_atm_file',   default='atm_2021_2025')
    parser.add_argument('--combined_output',  action='store_true')
    parser.add_argument('--output_dir',       default='results/Persistence_Baseline')
    args = parser.parse_args()

    pipe = ConfigPipeline([
        YamlConfig(f'./{args.config_file}', config_name='default',
                   config_folder='config/'),
        YamlConfig(config_folder='config/'),
    ])
    config = pipe.read_conf()
    config.name = 'Persistence_Baseline'

    Path('logs').mkdir(exist_ok=True)
    tee = TeeLogger('logs/persistence_baseline.log')
    sys.stdout = tee

    try:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        data_dir   = config.data.data_dir
        ocean_data = xr.open_dataset(Path(data_dir) / f'{config.data.file_prefix}.nc')
        atm_data   = xr.open_dataset(Path(data_dir) / f'{config.data.file_prefix_atm}.nc')

        print('\n=== Persistence baseline: primary (2020) ===')
        avg, primary_raw, n1 = run_persistence_evaluation(
            config, ocean_data, atm_data,
            config.evaluation.start_date,
            config.evaluation.num_days,
            config.evaluation.reference_date,
            desc='2020',
        )
        save_forecast_metrics(avg, str(out_dir / 'forecast_metrics.txt'),
                              config, config.evaluation.start_date, n1)
        print(f'Saved: {out_dir}/forecast_metrics.txt')
        ocean_data.close(); atm_data.close()

        ext_raw = None; n2 = 0
        if args.extra_data_dir:
            extra_dir   = Path(args.extra_data_dir)
            extra_ocean = xr.open_dataset(extra_dir / f'{args.extra_ocean_file}.nc')
            extra_atm   = xr.open_dataset(extra_dir / f'{args.extra_atm_file}.nc')

            print(f'\n=== Persistence baseline: extended ({args.extra_start_date}) ===')
            ext_avg, ext_raw, n2 = run_persistence_evaluation(
                config, extra_ocean, extra_atm,
                args.extra_start_date, args.extra_num_days, args.extra_ref_date,
                desc='2021-2025',
            )
            save_forecast_metrics(ext_avg, str(out_dir / 'forecast_metrics_extended.txt'),
                                  config, args.extra_start_date, n2)
            print(f'Saved: {out_dir}/forecast_metrics_extended.txt')
            extra_ocean.close(); extra_atm.close()

        if args.combined_output and ext_raw is not None:
            max_lead = config.evaluation.max_forecast_days
            comb_raw = {
                lt: {var: primary_raw[lt][var] + ext_raw[lt][var]
                     for var in config.data.out_variable}
                for lt in range(1, max_lead + 1)
            }
            comb_avg = average_raw_metrics(
                comb_raw, config.data.out_variable, max_lead)
            label = f'{config.evaluation.start_date} to {args.extra_start_date}+{args.extra_num_days}d'
            save_forecast_metrics(comb_avg, str(out_dir / 'forecast_metrics_combined.txt'),
                                  config, label, n1 + n2)
            print(f'Saved: {out_dir}/forecast_metrics_combined.txt')

        print(f'\nAll outputs in {out_dir}/')

    except Exception as exc:
        import traceback
        print(f'\n[FATAL] {exc}'); traceback.print_exc()
    finally:
        cleanup_logging(tee)


if __name__ == '__main__':
    main()

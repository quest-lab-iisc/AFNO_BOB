"""
Multi-day forecast evaluation for coupled atmospheric-ocean model
=================================================================

Runs fully autoregressive inference for each day in the evaluation period
and accumulates metrics (RMSE, MAE, R², Pearson) for all 11 variables at
each forecast lead time.

Metrics are computed in original data resolution: predictions are
reverse-normalized and interpolated from the model's 224×224 working
space back to the native grid of each variable before comparison with
raw ground truth.

Extended evaluation (2021-2025 data) can be run by passing --extra_data_dir.
The extra data files must follow the naming convention
{extra_ocean_file}.nc / {extra_atm_file}.nc inside that directory.

Inputs:
    --name (str): Experiment name (overrides config).
    --model_path (str): Path to .pth checkpoint.
    --config_file (str): YAML config filename in config/.
    --extra_data_dir (str): Directory with supplementary NetCDF data.
    --extra_start_date (str): First evaluation date in new data (dd-mm-yyyy).
    --extra_num_days (int): Number of days to evaluate from extra data.
    --extra_ref_date (str): Date of index 0 in the extra NetCDF (dd-mm-yyyy).
    --extra_ocean_file (str): File prefix for extra ocean NetCDF.
    --extra_atm_file (str): File prefix for extra atm NetCDF.

Outputs:
    results/{name}/forecast_metrics.txt: Metrics for the primary evaluation period.
    results/{name}/forecast_metrics_extended.txt: Metrics for the extra period (if
        --extra_data_dir is given).

Example:
    python src/inference/run_metrics_atmocean.py --name AFNO_AtmOcean_E00

    # With 2021-2025 extension
    python src/inference/run_metrics_atmocean.py \\
        --name AFNO_AtmOcean_E00 \\
        --extra_data_dir data/2021_2025 \\
        --extra_num_days 1826
"""
import os
import sys
import argparse
import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from configmypy import ConfigPipeline, YamlConfig
from models.architectures.afno.afnonet import AFNONet
from inference.run_inference_atmocean import (
    load_model,
    load_normalization_stats,
    run_autoregressive_forecast,
    postprocess_atm_variable,
    postprocess_ocean_variable,
)
from inference.utils import compute_metrics, date_to_day_index


# ---------------------------------------------------------------------------
# Metrics I/O
# ---------------------------------------------------------------------------

def save_forecast_metrics(metrics_dict, save_path, config, start_date, num_eval_days):
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MULTI-DAY FORECAST EVALUATION METRICS (AtmOcean)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"  Model: {config.name}\n")
        f.write(f"  Start Date: {start_date}\n")
        f.write(f"  Evaluation Days: {num_eval_days}\n")
        f.write(f"  Max Forecast Lead: {config.evaluation.max_forecast_days} days\n\n")
        f.write("=" * 80 + "\n\n")

        all_vars = config.data.atm_variable + config.data.variable

        for lead_time in sorted(metrics_dict.keys()):
            f.write(f"LEAD TIME: +{lead_time} day(s)\n")
            f.write("-" * 80 + "\n")
            for var in all_vars:
                if var in metrics_dict[lead_time]:
                    m = metrics_dict[lead_time][var]
                    f.write(f"  {var.upper():8s}  "
                            f"RMSE={m['rmse']:.6f}  MAE={m['mae']:.6f}  "
                            f"R²={m['r2']:.6f}  r={m['pearson']:.6f}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("SUMMARY (Averaged Across All Forecast Leads)\n")
        f.write("=" * 80 + "\n\n")

        for var in all_vars:
            rmse_list    = [metrics_dict[lt][var]['rmse']    for lt in metrics_dict]
            mae_list     = [metrics_dict[lt][var]['mae']     for lt in metrics_dict]
            r2_list      = [metrics_dict[lt][var]['r2']      for lt in metrics_dict]
            pearson_list = [metrics_dict[lt][var]['pearson'] for lt in metrics_dict]

            f.write(f"{var.upper():8s}  "
                    f"RMSE={np.mean(rmse_list):.6f}±{np.std(rmse_list):.6f}  "
                    f"MAE={np.mean(mae_list):.6f}±{np.std(mae_list):.6f}  "
                    f"R²={np.mean(r2_list):.6f}  r={np.mean(pearson_list):.6f}\n")

        f.write("\n" + "=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Postprocessing (all 11 variables to original resolution)
# ---------------------------------------------------------------------------

def postprocess_all_forecasts(forecast_dict, mean_dict, variance_dict, config):
    """
    Reverse normalization and interpolate all predictions to original data resolution.

    Returns:
        dict: {lead_time: {var: np.ndarray at original resolution}}
    """
    atm_vars   = config.data.atm_variable
    ocean_vars = config.data.variable
    postprocessed = {}

    for lead_time, preds in forecast_dict.items():
        step = {}
        for var in atm_vars:
            step[var] = postprocess_atm_variable(preds[var], var, mean_dict, variance_dict)
        for var in ocean_vars:
            step[var] = postprocess_ocean_variable(preds[var], var, mean_dict)
        postprocessed[lead_time] = step

    return postprocessed


# ---------------------------------------------------------------------------
# Ground truth at original resolution
# ---------------------------------------------------------------------------

def load_ground_truth_original(config, day_index, ocean_data, atm_data):
    """
    Load raw ground truth at original resolution for all 11 variables.

    Returns:
        dict: {var: np.ndarray at original resolution}
    """
    gt = {}
    for var in config.data.atm_variable:
        gt[var] = atm_data[var][day_index:day_index+1].values.squeeze()
    for var in config.data.variable:
        gt[var] = ocean_data[var][day_index:day_index+1].values.squeeze()
    return gt


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def average_raw_metrics(raw_metrics, variables, max_forecast_days):
    """Average a raw per-day metrics dict into a single averaged dict.

    Args:
        raw_metrics (dict): {lead_time: {var: [list of metric dicts]}}.
        variables (list): Variable names.
        max_forecast_days (int): Maximum forecast lead time.

    Returns:
        dict: {lead_time: {var: {rmse, mae, r2, pearson}}}.

    Example:
        >>> avg = average_raw_metrics(raw, ['ssr', 'thetao'], 9)
    """
    averaged = {}
    for lt in range(1, max_forecast_days + 1):
        averaged[lt] = {}
        for var in variables:
            ms = raw_metrics[lt][var]
            averaged[lt][var] = {
                'rmse':    np.mean([m['rmse']    for m in ms]),
                'mae':     np.mean([m['mae']     for m in ms]),
                'r2':      np.mean([m['r2']      for m in ms]),
                'pearson': np.mean([m['pearson'] for m in ms]),
            }
    return averaged


def run_evaluation(config, model, device, ocean_data, atm_data,
                   start_date, num_days, reference_date, desc="Evaluating"):
    """Run evaluation loop on a given pair of datasets and return metrics.

    Args:
        config: Configuration object.
        model (torch.nn.Module): Trained model in eval mode.
        device (torch.device): Compute device.
        ocean_data (xr.Dataset): Ocean NetCDF dataset.
        atm_data (xr.Dataset): Atmospheric NetCDF dataset.
        start_date (str): Evaluation start date in dd-mm-yyyy format.
        num_days (int): Total days in the evaluation window (including lead buffer).
        reference_date (str): Date corresponding to index 0 in the datasets.
        desc (str): tqdm progress bar label.

    Returns:
        tuple: (averaged_metrics, raw_metrics, start_day_index, num_eval_days)
            averaged_metrics maps lead_time -> {var: {rmse, mae, r2, pearson}}.
            raw_metrics maps lead_time -> {var: [list of per-day metric dicts]}.

    Example:
        >>> averaged, raw, start_idx, n_days = run_evaluation(
        ...     config, model, device, ocean_data, atm_data,
        ...     "01-01-2021", 365, "01-01-2021")
    """
    max_forecast_days = config.evaluation.max_forecast_days
    start_day_index   = date_to_day_index(start_date, reference_date)
    num_eval_days     = num_days - max_forecast_days + 1

    # Cap to what the dataset can support — the formula assumes a one-day buffer
    # beyond num_days, but files with exactly num_days steps have no such buffer.
    dataset_size  = len(ocean_data.time)
    max_safe_days = dataset_size - start_day_index - max_forecast_days
    num_eval_days = min(num_eval_days, max_safe_days)

    print(f"Start date: {start_date}  (index {start_day_index})")
    print(f"Evaluation days: {num_eval_days},  max lead: {max_forecast_days}")

    all_vars    = config.data.atm_variable + config.data.variable
    raw_metrics = {lt: {var: [] for var in all_vars}
                   for lt in range(1, max_forecast_days + 1)}

    print("(Metrics computed at original data resolution)")
    for eval_day in tqdm(range(num_eval_days), desc=desc):
        current_idx = start_day_index + eval_day

        forecast_dict, mean_dict, variance_dict = run_autoregressive_forecast(
            config, model, current_idx, max_forecast_days, device, ocean_data, atm_data)

        postprocessed = postprocess_all_forecasts(
            forecast_dict, mean_dict, variance_dict, config)

        for lead_time in range(1, max_forecast_days + 1):
            gt = load_ground_truth_original(
                config, current_idx + lead_time, ocean_data, atm_data)
            for var in all_vars:
                raw_metrics[lead_time][var].append(
                    compute_metrics(postprocessed[lead_time][var], gt[var]))

    averaged = average_raw_metrics(raw_metrics, all_vars, max_forecast_days)
    return averaged, raw_metrics, start_day_index, num_eval_days


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Entry point for coupled atm-ocean multi-day forecast evaluation.

    Args:
        (CLI args — see module docstring)

    Returns:
        None: Writes metric files to results/{name}/.

    Example:
        >>> # python src/inference/run_metrics_atmocean.py --name AFNO_AtmOcean_E00
    """
    parser = argparse.ArgumentParser(
        description='Multi-day evaluation for coupled atm-ocean model')
    parser.add_argument('--name',       type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--config_file', type=str, default='afno_atmocean_config.yaml')
    # Extended evaluation (2021-2025 data)
    parser.add_argument('--extra_data_dir',  type=str, default=None,
                        help='Directory containing supplementary NetCDF files')
    parser.add_argument('--extra_start_date', type=str, default='01-01-2021',
                        help='First evaluation date in extra data (dd-mm-yyyy)')
    parser.add_argument('--extra_num_days', type=int, default=1826,
                        help='Number of days to evaluate from extra data (default 1826 = 5 years)')
    parser.add_argument('--extra_ref_date', type=str, default='01-01-2021',
                        help='Date at index 0 in the extra NetCDF (dd-mm-yyyy)')
    parser.add_argument('--extra_ocean_file', type=str, default='ocean_2021_2025',
                        help='File prefix for extra ocean NetCDF')
    parser.add_argument('--extra_atm_file', type=str, default='atm_2021_2025',
                        help='File prefix for extra atm NetCDF')
    parser.add_argument('--combined_output', action='store_true',
                        help='Also write forecast_metrics_combined.txt pooling both periods')
    args = parser.parse_args()

    print("=== Loading Configuration ===")
    pipe = ConfigPipeline([
        YamlConfig(f"./{args.config_file}", config_name='default', config_folder='config/'),
        YamlConfig(config_folder='config/')
    ])
    config = pipe.read_conf()

    if args.name:
        config.name = args.name

    start_date        = config.evaluation.start_date
    num_days          = config.evaluation.num_days
    reference_date    = config.evaluation.reference_date
    max_forecast_days = config.evaluation.max_forecast_days

    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model_path = args.model_path or f"{config.results.model_dir}/{config.name}.pth"
    if not os.path.exists(model_path):
        print(f"Error: model not found at {model_path}")
        sys.exit(1)
    model = load_model(config, model_path, device)

    data_dir   = config.data.data_dir
    ocean_data = xr.open_dataset(Path(data_dir) / f'{config.data.file_prefix}.nc')
    atm_data   = xr.open_dataset(Path(data_dir) / f'{config.data.file_prefix_atm}.nc')

    # --- Primary evaluation (configured test period) -------------------------
    print(f"\n=== Primary Evaluation ({start_date}, {num_days} days) ===")
    averaged, primary_raw, _, num_eval_days = run_evaluation(
        config, model, device, ocean_data, atm_data,
        start_date, num_days, reference_date, desc="Primary eval")

    out_dir = Path(config.results.save_dir) / config.name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "forecast_metrics.txt"
    save_forecast_metrics(averaged, str(out_file), config, start_date, num_eval_days)
    print(f"Metrics saved to: {out_file}")

    # --- Extended evaluation (extra data, if requested) ----------------------
    ext_raw = None
    ext_eval_days = 0
    if args.extra_data_dir:
        extra_dir = Path(args.extra_data_dir)
        extra_ocean_path = extra_dir / f'{args.extra_ocean_file}.nc'
        extra_atm_path   = extra_dir / f'{args.extra_atm_file}.nc'

        if not extra_ocean_path.exists():
            print(f"Error: extra ocean file not found: {extra_ocean_path}")
            sys.exit(1)
        if not extra_atm_path.exists():
            print(f"Error: extra atm file not found: {extra_atm_path}")
            sys.exit(1)

        print(f"\n=== Extended Evaluation ({args.extra_start_date}, "
              f"{args.extra_num_days} days) ===")
        print(f"Ocean: {extra_ocean_path}")
        print(f"Atm:   {extra_atm_path}")

        extra_ocean = xr.open_dataset(extra_ocean_path)
        extra_atm   = xr.open_dataset(extra_atm_path)

        ext_averaged, ext_raw, _, ext_eval_days = run_evaluation(
            config, model, device, extra_ocean, extra_atm,
            args.extra_start_date, args.extra_num_days, args.extra_ref_date,
            desc="Extended eval")

        ext_out = out_dir / "forecast_metrics_extended.txt"
        save_forecast_metrics(ext_averaged, str(ext_out), config,
                              args.extra_start_date, ext_eval_days)
        print(f"Extended metrics saved to: {ext_out}")

        extra_ocean.close()
        extra_atm.close()

    # --- Combined output (primary + extended pooled) -------------------------
    if args.combined_output and ext_raw is not None:
        all_vars = config.data.atm_variable + config.data.variable
        max_forecast_days = config.evaluation.max_forecast_days
        combined_raw = {
            lt: {var: primary_raw[lt][var] + ext_raw[lt][var]
                 for var in all_vars}
            for lt in range(1, max_forecast_days + 1)
        }
        combined_averaged = average_raw_metrics(combined_raw, all_vars, max_forecast_days)
        combined_eval_days = num_eval_days + ext_eval_days
        combined_label = f"{start_date} to {args.extra_start_date}+{args.extra_num_days}d"

        comb_out = out_dir / "forecast_metrics_combined.txt"
        save_forecast_metrics(combined_averaged, str(comb_out), config,
                              combined_label, combined_eval_days)
        print(f"Combined metrics saved to: {comb_out}")
    elif args.combined_output and ext_raw is None:
        print("Warning: --combined_output requires --extra_data_dir; skipping combined file.")

    ocean_data.close()
    atm_data.close()
    print("\n=== Evaluation Complete ===")


if __name__ == "__main__":
    main()

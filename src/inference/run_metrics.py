"""
Multi-day forecast evaluation script (ocean-only model)
========================================================

Runs autoregressive multi-day forecasts and evaluates them against ground truth.
Computes RMSE, MAE, R², Pearson for each ocean variable at each forecast lead.

Extended evaluation (2021-2025 data) can be run by passing --extra_data_dir.
Results are saved separately to forecast_metrics_extended.txt.

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
    results/{name}/forecast_metrics.txt: Primary evaluation metrics.
    results/{name}/forecast_metrics_extended.txt: Extended metrics (if
        --extra_data_dir is given).

Configuration:
    The script reads from config file (default: config/afno_bob_config.yaml):
    - evaluation.start_date: Start date (dd-mm-yyyy)
    - evaluation.num_days: Total days in the evaluation window
    - evaluation.max_forecast_days: Maximum forecast lead
    - evaluation.reference_date: Date at index 0 in the original NetCDF

    Effective evaluation days = num_days - max_forecast_days + 1

Example:
    python src/inference/run_metrics.py --name AFNO_BoB_Surf_E01

    python src/inference/run_metrics.py --name AFNO_BoB_Surf_E01 \\
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
from datetime import datetime, timedelta
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from configmypy import ConfigPipeline, YamlConfig
from models.architectures.afno.afnonet import AFNONet
from data_pipeline.preprocessing.transformers.normalize_transform import PreprocessTransform
from inference.utils import (
    postprocess_ocean_variable,
    compute_metrics,
    date_to_day_index
)


def load_model(config, model_path, device):
    """
    Load trained model from checkpoint

    Args:
        config: Configuration object
        model_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded model
    """
    model = AFNONet(config)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            # Full checkpoint with training state
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"AFNO Model loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')}): {model_path}")
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)
            print(f"AFNO Model loaded from: {model_path}")
    else:
        # Legacy format
        model.load_state_dict(checkpoint)
        print(f"AFNO Model loaded from: {model_path}")

    model.to(device)
    model.eval()
    return model


def load_normalization_stats(config):
    """
    Load mean and variance statistics for normalization

    Args:
        config: Configuration object

    Returns:
        Tuple of (mean_dict, variance_dict)
    """
    mean_dir = config.data.mean_dir
    mean_dict = {}
    variance_dict = {}

    # Ocean variables
    for var in ['thetao', 'so', 'uo', 'vo', 'zos']:
        mean_file = f"{mean_dir}/mean_{var}_1993_2018_all_months.npy"
        try:
            mean_dict[var] = np.load(mean_file)
        except FileNotFoundError:
            mean_dict[var] = None

    # Atmospheric variables
    for var in ['ssr', 'tp', 'u10', 'v10', 'msl', 'tcc']:
        mean_file = f"{mean_dir}/{var}_mean_1993_2018_all_months.npy"
        try:
            mean_dict[var] = np.load(mean_file)
        except FileNotFoundError:
            mean_dict[var] = None

    # Load variance for atmospheric variables
    variance_dict['ssr_std'] = np.sqrt(np.load(f'{mean_dir}/ssr_var_1993_2018_all_months.npy'))
    variance_dict['tp_std'] = np.sqrt(np.load(f'{mean_dir}/tp_var_1993_2018_all_months.npy'))
    variance_dict['msl_std'] = np.sqrt(np.load(f'{mean_dir}/msl_var_1993_2018_all_months.npy'))

    return mean_dict, variance_dict


def preprocess_ocean_variable(data, mean, variable, transform, config=None):
    """
    Preprocess ocean variable (normalize and interpolate to 224x224)

    Args:
        data: Raw ocean data
        mean: Mean value for normalization
        variable: Variable name
        transform: PreprocessTransform instance
        config: Configuration object (used for north_mask_rows; optional)

    Returns:
        Preprocessed tensor (1, H, W) at 224x224 resolution
    """
    # Apply transform (normalization + interpolation)
    processed = transform(data, mean, variable=variable)

    # Mask northern boundary (same as training)
    rows = getattr(getattr(config, 'data', None), 'north_mask_rows', 0) if config else 0
    if rows > 0:
        processed[:, :, -rows:, :] = 0.0

    return processed.squeeze(1)  # Remove batch dim: (1, H, W)


def preprocess_atm_variable(data, mean, variable, transform, variance=None):
    """
    Preprocess atmospheric variable (normalize and interpolate to 224x224)

    Args:
        data: Raw atmospheric data
        mean: Mean value for normalization
        variable: Variable name
        transform: PreprocessTransform instance
        variance: Variance for scaling (optional)

    Returns:
        Preprocessed tensor (1, H, W) at 224x224 resolution
    """
    # Apply transform
    if variable in ['ssr', 'tp', 'msl'] and variance is not None:
        processed = transform(data, mean, variable=variable, type='atm', variance=variance)
    else:
        processed = transform(data, mean, variable=variable, type='atm')

    return processed.squeeze(1)  # Remove batch dim: (1, H, W)


def load_initial_ocean_state(config, day_index, ocean_data, transform, mean_dict):
    """
    Load and preprocess initial ocean state at time t

    Args:
        config: Configuration object
        day_index: Day index in the dataset
        ocean_data: Loaded ocean dataset
        transform: Preprocessing transform
        mean_dict: Dictionary of mean values

    Returns:
        Dictionary of preprocessed ocean variables (each 1xHxW at 224x224)
    """
    ocean_state = {}
    for ocean_var in config.data.variable:
        ocean_input = ocean_data[ocean_var][day_index:day_index+1].values
        ocean_input = preprocess_ocean_variable(ocean_input, mean_dict[ocean_var],
                                               ocean_var, transform, config)
        ocean_state[ocean_var] = ocean_input

    return ocean_state


def load_atmospheric_forcing(config, day_index, atm_data, transform, mean_dict, variance_dict):
    """
    Load and preprocess atmospheric forcing at time t+1

    Args:
        config: Configuration object
        day_index: Day index for t+1
        atm_data: Loaded atmospheric dataset
        transform: Preprocessing transform
        mean_dict: Dictionary of mean values
        variance_dict: Dictionary of variance values

    Returns:
        Concatenated atmospheric tensor (6, H, W) at 224x224
    """
    atm_inputs = []
    for atm_var in config.data.atm_variable:
        atm_input = atm_data[atm_var][day_index:day_index+1].values

        if atm_var == 'ssr':
            atm_input = preprocess_atm_variable(atm_input, mean_dict[atm_var],
                                               atm_var, transform, variance_dict['ssr_std'])
        elif atm_var == 'tp':
            atm_input = preprocess_atm_variable(atm_input, mean_dict[atm_var],
                                               atm_var, transform, variance_dict['tp_std'])
        elif atm_var == 'msl':
            atm_input = preprocess_atm_variable(atm_input, mean_dict[atm_var],
                                               atm_var, transform, variance_dict['msl_std'])
        else:
            atm_input = preprocess_atm_variable(atm_input, mean_dict[atm_var],
                                               atm_var, transform)

        atm_inputs.append(atm_input)

    # Concatenate all atmospheric variables
    atm_tensor = torch.cat(atm_inputs, dim=0)  # (6, H, W)
    return atm_tensor


def run_autoregressive_forecast(config, model, start_day_index, forecast_days, device,
                                ocean_data, atm_data):
    """
    Run autoregressive forecast for multiple days

    Predictions stay in preprocessed space (224x224) during the loop to minimize
    accumulation of round-off errors from repeated interpolation.

    Args:
        config: Configuration object
        model: Trained model
        start_day_index: Starting day index
        forecast_days: Number of days to forecast
        device: Device for computation
        ocean_data: Loaded ocean dataset
        atm_data: Loaded atmospheric dataset

    Returns:
        Dictionary mapping lead_time -> {variable: preprocessed_prediction (224x224)}
    """
    # Load normalization statistics
    mean_dict, variance_dict = load_normalization_stats(config)

    # Create transform
    transform = PreprocessTransform(config)

    # Load initial ocean state at time t (preprocessed, 224x224)
    ocean_state = load_initial_ocean_state(config, start_day_index, ocean_data,
                                          transform, mean_dict)

    # Store predictions for each lead time (in preprocessed space)
    forecast_dict = {}

    # Autoregressive loop
    for lead_time in range(1, forecast_days + 1):
        # Load atmospheric forcing at t+lead_time
        atm_forcing = load_atmospheric_forcing(config, start_day_index + lead_time,
                                               atm_data, transform, mean_dict, variance_dict)

        # Concatenate ocean state + atmospheric forcing
        ocean_vars = [ocean_state[var] for var in config.data.variable]
        ocean_tensor = torch.cat(ocean_vars, dim=0)  # (5, H, W)
        input_tensor = torch.cat([atm_forcing, ocean_tensor], dim=0)  # (11, H, W)

        # Detect corrupted ATM data (e.g. outlier pixels in source NetCDF).
        # Raise so the eval loop can skip this starting day entirely.
        if input_tensor.abs().max() > 1e4:
            raise ValueError(
                f"Corrupted input at lead {lead_time} "
                f"(max |value|={input_tensor.abs().max():.1f}); skipping this start day.")

        # Run model
        with torch.no_grad():
            input_batch = input_tensor.unsqueeze(0).to(device)  # (1, 11, H, W)
            output = model(input_batch)
            prediction = output.squeeze(0).cpu()  # (5, H, W)

        # Split prediction into individual variables (keep in preprocessed space)
        num_channels_per_var = prediction.shape[0] // len(config.data.out_variable)
        predictions_step = {}

        for i, var in enumerate(config.data.out_variable):
            pred_data = prediction[i*num_channels_per_var:(i+1)*num_channels_per_var]
            pred_data = pred_data.squeeze()  # (224, 224)

            # Store preprocessed prediction
            predictions_step[var] = pred_data.numpy()

            # Update ocean state for next iteration (keep in preprocessed space)
            ocean_state[var] = pred_data.unsqueeze(0)  # (1, 224, 224)

        # Store predictions for this lead time
        forecast_dict[lead_time] = predictions_step

    # Return forecasts and normalization stats for postprocessing
    return forecast_dict, mean_dict


def postprocess_forecasts(forecast_dict, mean_dict, config):
    """
    Postprocess all forecasts (reverse normalization and interpolation)

    Args:
        forecast_dict: Dictionary mapping lead_time -> {variable: preprocessed_data}
        mean_dict: Dictionary of mean values
        config: Configuration object

    Returns:
        Dictionary mapping lead_time -> {variable: postprocessed_data}
    """
    postprocessed_dict = {}

    for lead_time, predictions in forecast_dict.items():
        postprocessed_step = {}

        for var in config.data.out_variable:
            pred_data = predictions[var]

            # Postprocess: reverse normalization and interpolate to original resolution
            pred_postprocessed = postprocess_ocean_variable(pred_data, mean_dict[var], var)

            # Handle northern boundary masking (set to NaN)
            rows = config.data.north_mask_rows
            if rows > 0:
                pred_postprocessed[-rows:, :] = np.nan

            postprocessed_step[var] = pred_postprocessed

        postprocessed_dict[lead_time] = postprocessed_step

    return postprocessed_dict


def load_ground_truth_batch(config, start_day_index, forecast_days, ocean_data):
    """
    Load ground truth for multiple forecast days

    Args:
        config: Configuration object
        start_day_index: Starting day index
        forecast_days: Number of forecast days
        ocean_data: Loaded ocean dataset

    Returns:
        Dictionary mapping lead_time -> {variable: ground_truth_data}
    """
    ground_truth_dict = {}

    for lead_time in range(1, forecast_days + 1):
        day_idx = start_day_index + lead_time
        ground_truth = {}

        for var in config.data.out_variable:
            gt_data = ocean_data[var][day_idx:day_idx+1].values.squeeze()
            ground_truth[var] = gt_data

        ground_truth_dict[lead_time] = ground_truth

    return ground_truth_dict


def compute_forecast_metrics(predictions_dict, ground_truth_dict, config):
    """
    Compute metrics for all forecast lead times

    Args:
        predictions_dict: Dictionary mapping lead_time -> {variable: prediction}
        ground_truth_dict: Dictionary mapping lead_time -> {variable: ground_truth}
        config: Configuration object

    Returns:
        Dictionary mapping lead_time -> {variable: metrics}
    """
    metrics_all = {}

    for lead_time in predictions_dict.keys():
        metrics_step = {}

        for var in config.data.out_variable:
            pred = predictions_dict[lead_time][var]
            truth = ground_truth_dict[lead_time][var]

            # Compute metrics
            metrics = compute_metrics(pred, truth)
            metrics_step[var] = metrics

        metrics_all[lead_time] = metrics_step

    return metrics_all


def save_forecast_metrics(metrics_dict, save_path, config, start_date, num_eval_days):
    """
    Save forecast metrics to file

    Args:
        metrics_dict: Dictionary with evaluation results
        save_path: Path to save the metrics file
        config: Configuration object
        start_date: Start date string
        num_eval_days: Number of evaluation days
    """
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MULTI-DAY FORECAST EVALUATION METRICS\n")
        f.write("=" * 80 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Model: {config.name}\n")
        f.write(f"  Start Date: {start_date}\n")
        f.write(f"  Evaluation Days: {num_eval_days}\n")
        f.write(f"  Max Forecast Lead: {config.evaluation.max_forecast_days} days\n")
        f.write("\n" + "=" * 80 + "\n\n")

        # Detailed metrics for each lead time
        for lead_time in sorted(metrics_dict.keys()):
            f.write(f"LEAD TIME: +{lead_time} day(s)\n")
            f.write("-" * 80 + "\n")

            for var in config.data.out_variable:
                if var in metrics_dict[lead_time]:
                    metrics = metrics_dict[lead_time][var]
                    f.write(f"  {var.upper()}:\n")
                    f.write(f"    RMSE:              {metrics['rmse']:.6f}\n")
                    f.write(f"    MAE:               {metrics['mae']:.6f}\n")
                    f.write(f"    R²:                {metrics['r2']:.6f}\n")
                    f.write(f"    Pearson Corr:      {metrics['pearson']:.6f}\n")

            f.write("\n")

        # Summary: Average across all lead times
        f.write("=" * 80 + "\n")
        f.write("SUMMARY (Averaged Across All Forecast Leads)\n")
        f.write("=" * 80 + "\n\n")

        for var in config.data.out_variable:
            # Collect metrics across all lead times
            rmse_list = [metrics_dict[lt][var]['rmse'] for lt in metrics_dict.keys()]
            mae_list = [metrics_dict[lt][var]['mae'] for lt in metrics_dict.keys()]
            r2_list = [metrics_dict[lt][var]['r2'] for lt in metrics_dict.keys()]
            pearson_list = [metrics_dict[lt][var]['pearson'] for lt in metrics_dict.keys()]

            f.write(f"{var.upper()}:\n")
            f.write(f"  RMSE:              {np.mean(rmse_list):.6f} ± {np.std(rmse_list):.6f}\n")
            f.write(f"  MAE:               {np.mean(mae_list):.6f} ± {np.std(mae_list):.6f}\n")
            f.write(f"  R²:                {np.mean(r2_list):.6f} ± {np.std(r2_list):.6f}\n")
            f.write(f"  Pearson Corr:      {np.mean(pearson_list):.6f} ± {np.std(pearson_list):.6f}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")


def average_raw_metrics(raw_metrics, variables, max_forecast_days):
    """Average a raw per-day metrics dict into a single averaged dict.

    Args:
        raw_metrics (dict): {lead_time: {var: [list of metric dicts]}}.
        variables (list): Variable names.
        max_forecast_days (int): Maximum forecast lead time.

    Returns:
        dict: {lead_time: {var: {rmse, mae, r2, pearson}}}.

    Example:
        >>> avg = average_raw_metrics(raw, ['thetao', 'so'], 9)
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
        reference_date (str): Date corresponding to index 0 in the datasets (dd-mm-yyyy).
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

    print(f"  Start date: {start_date}  (index {start_day_index})")
    print(f"  Effective evaluation days: {num_eval_days}")

    raw_metrics = {lt: {var: [] for var in config.data.out_variable}
                   for lt in range(1, max_forecast_days + 1)}

    skipped = 0
    for eval_day in tqdm(range(num_eval_days), desc=desc, file=sys.stdout,
                         dynamic_ncols=True):
        current_day_index = start_day_index + eval_day

        try:
            forecast_dict, mean_dict = run_autoregressive_forecast(
                config, model, current_day_index, max_forecast_days, device,
                ocean_data, atm_data)
        except ValueError as exc:
            tqdm.write(f"  Skipping day {eval_day}: {exc}")
            skipped += 1
            continue

        postprocessed_forecasts = postprocess_forecasts(forecast_dict, mean_dict, config)

        ground_truth_dict = load_ground_truth_batch(
            config, current_day_index, max_forecast_days, ocean_data)

        metrics_dict = compute_forecast_metrics(
            postprocessed_forecasts, ground_truth_dict, config)

        for lead_time in range(1, max_forecast_days + 1):
            for var in config.data.out_variable:
                raw_metrics[lead_time][var].append(metrics_dict[lead_time][var])

    if skipped:
        tqdm.write(f"  NOTE: {skipped} day(s) skipped due to corrupted input data.")

    averaged_metrics = average_raw_metrics(
        raw_metrics, config.data.out_variable, max_forecast_days)
    return averaged_metrics, raw_metrics, start_day_index, num_eval_days


def main():
    """Entry point for ocean-only multi-day forecast evaluation.

    Args:
        (CLI args — see module docstring)

    Returns:
        None: Writes metric files to results/{name}/.

    Example:
        >>> # python src/inference/run_metrics.py --name AFNO_BoB_Surf_E01
    """
    parser = argparse.ArgumentParser(description='Run multi-day forecast evaluation')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name (overrides config name)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--config_file', type=str, default='afno_bob_config.yaml',
                        help='Config file name (default: afno_bob_config.yaml)')
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
    print(f"Using config file: {args.config_file}")
    pipe = ConfigPipeline([
        YamlConfig(f"./{args.config_file}", config_name='default', config_folder='config/'),
        YamlConfig(config_folder='config/')
    ])
    config = pipe.read_conf()

    if args.name:
        config.name = args.name
        print(f"Overriding config name with: {args.name}")

    start_date        = config.evaluation.start_date
    num_days          = config.evaluation.num_days
    max_forecast_days = config.evaluation.max_forecast_days
    reference_date    = config.evaluation.reference_date

    print(f"\nEvaluation Configuration:")
    print(f"  Start Date: {start_date}, Total Days: {num_days}")
    print(f"  Max Forecast Lead: {max_forecast_days} days")

    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model_path = args.model_path or str(Path(config.results.model_dir) / f"{config.name}.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    model = load_model(config, model_path, device)

    data_dir   = config.data.data_dir
    ocean_file = Path(data_dir) / f'{config.data.file_prefix}.nc'
    atm_file   = Path(data_dir) / f'{config.data.file_prefix_atm}.nc'
    ocean_data = xr.open_dataset(ocean_file)
    atm_data   = xr.open_dataset(atm_file)
    print(f"\nOcean data: {ocean_file}")
    print(f"Atm data:   {atm_file}")

    # --- Primary evaluation --------------------------------------------------
    print(f"\n=== Primary Evaluation ({start_date}, {num_days} days) ===")
    averaged_metrics, primary_raw, _, num_eval_days = run_evaluation(
        config, model, device, ocean_data, atm_data,
        start_date, num_days, reference_date, desc="Primary eval")

    out_dir = Path(config.results.save_dir) / config.name
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / "forecast_metrics.txt"
    save_forecast_metrics(averaged_metrics, str(output_file), config,
                          start_date, num_eval_days)
    print(f"\nMetrics saved to: {output_file}")

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
        max_forecast_days = config.evaluation.max_forecast_days
        combined_raw = {
            lt: {var: primary_raw[lt][var] + ext_raw[lt][var]
                 for var in config.data.out_variable}
            for lt in range(1, max_forecast_days + 1)
        }
        combined_averaged = average_raw_metrics(
            combined_raw, config.data.out_variable, max_forecast_days)
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

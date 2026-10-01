"""
Multi-day forecast evaluation script for TFNO model
====================================================

This script performs autoregressive multi-day forecasts using a trained TFNO (Tensorized Fourier
Neural Operator) model and evaluates them against ground truth data. It computes comprehensive
metrics (RMSE, MAE, R², Pearson) for each ocean variable at each forecast lead time.

Usage:
    # Run with default configuration (uses name from config file)
    python src/inference/run_metrics_tfno.py

    # Override experiment name
    python src/inference/run_metrics_tfno.py --name TFNO_BoB_Surf_E01

    # Use custom model checkpoint
    python src/inference/run_metrics_tfno.py --model_path results/models/custom_tfno_model.pth

    # Use a different config file
    python src/inference/run_metrics_tfno.py --config_file my_tfno_config.yaml

    # Combine multiple options
    python src/inference/run_metrics_tfno.py --name TFNO_BoB_Surf_E01 --model_path results/models/TFNO_BoB_Surf_E01.pth --config_file tfno_config.yaml

Configuration:
    The script reads evaluation parameters from config file (default: config/tfno_bob_config.yaml):
    - evaluation.start_date: Start date for evaluation (format: dd-mm-yyyy)
    - evaluation.num_days: Total number of days in evaluation period
    - evaluation.max_forecast_days: Maximum forecast lead time (e.g., 9 for 1-9 day forecasts)
    - evaluation.reference_date: Reference date for day index calculation

    Note: Effective evaluation days = num_days - max_forecast_days + 1
    This ensures we don't run out of atmospheric data for longer forecast leads.

Output:
    Results are saved to: results/{name}/forecast_metrics.txt
    - Detailed metrics for each variable at each forecast lead
    - Summary statistics averaged across all forecast leads
    - Evaluation configuration and parameters

Autoregressive Forecasting:
    - Ocean variables: Predictions from step t become inputs for step t+1
    - Atmospheric variables: Always loaded fresh from dataset for each step
    - Initial ocean state: Ground truth at start_date (time t)
    - Predictions stay in preprocessed space (224x224) during loop
    - Postprocessing done once at the end to minimize round-off errors

TFNO Model:
    This script loads TFNO models trained using train_TFNO.py. The TFNO architecture
    uses Tucker factorization to reduce parameters while maintaining FNO performance.
"""
import os
import sys
import argparse
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from configmypy import ConfigPipeline, YamlConfig
from neuralop.models import TFNO
from data_pipeline.preprocessing.transformers.normalize_transform import PreprocessTransform
from inference.utils import (
    postprocess_ocean_variable,
    compute_metrics,
    date_to_day_index
)


class TFNOWrapper(nn.Module):
    """
    Wrapper around neuralop TFNO to match the interface expected by the inference pipeline.

    This wrapper is identical to the one used in train_TFNO.py to ensure compatibility.
    """
    def __init__(self, config):
        super().__init__()

        # Extract TFNO configuration
        n_modes = tuple(config.tfno.n_modes)
        hidden_channels = config.tfno.hidden_channels
        lifting_channel_ratio = config.tfno.lifting_channel_ratio
        projection_channel_ratio = config.tfno.projection_channel_ratio
        n_layers = config.tfno.n_layers
        factorization = config.tfno.factorization
        rank = config.tfno.rank

        # Determine non-linearity function
        if config.tfno.non_linearity == 'gelu':
            non_linearity = F.gelu
        elif config.tfno.non_linearity == 'relu':
            non_linearity = F.relu
        elif config.tfno.non_linearity == 'tanh':
            non_linearity = torch.tanh
        else:
            non_linearity = F.gelu  # default

        # Initialize TFNO model from neuralop
        self.tfno = TFNO(
            n_modes=n_modes,
            in_channels=config.data.in_chs,
            out_channels=config.data.out_chs,
            hidden_channels=hidden_channels,
            lifting_channel_ratio=lifting_channel_ratio,
            projection_channel_ratio=projection_channel_ratio,
            n_layers=n_layers,
            factorization=factorization,
            rank=rank,
            non_linearity=non_linearity,
            use_channel_mlp=config.tfno.use_channel_mlp,
            channel_mlp_expansion=config.tfno.channel_mlp_expansion,
            channel_mlp_dropout=config.tfno.channel_mlp_dropout,
            channel_mlp_skip=config.tfno.channel_mlp_skip,
            fno_skip=config.tfno.fno_skip,
            positional_embedding=config.tfno.positional_embedding,
            norm=config.tfno.norm,
        )

        self.in_channels = config.data.in_chs
        self.out_channels = config.data.out_chs
        self.rank = rank
        self.factorization = factorization

    def forward(self, x):
        """
        Forward pass through TFNO.

        Args:
            x: Input tensor of shape (batch, in_channels, height, width)

        Returns:
            Output tensor of shape (batch, out_channels, height, width)
        """
        return self.tfno(x)


def load_model(config, model_path, device):
    """
    Load trained TFNO model from checkpoint

    Args:
        config: Configuration object
        model_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded model
    """
    model = TFNOWrapper(config)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            # Full checkpoint with training state
            state_dict = checkpoint['model_state_dict']
            print(f"TFNO Model loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')}): {model_path}")
        else:
            # Direct state dict - might need to unwrap if it has wrapper keys
            state_dict = checkpoint

            # Check if keys are prefixed with 'tfno.' and need unwrapping
            if all(k.startswith('tfno.') for k in state_dict.keys() if not k.startswith('_')):
                # State dict is already in wrapper format, load directly
                print(f"TFNO Model loaded from wrapped state dict: {model_path}")
            else:
                print(f"TFNO Model loaded from: {model_path}")

        # Clean up state dict - remove PyTorch internal metadata
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('_')}

        # Try loading with error handling
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"Warning: Strict loading failed, trying with strict=False: {e}")
            model.load_state_dict(state_dict, strict=False)
    else:
        # Legacy format
        model.load_state_dict(checkpoint)
        print(f"TFNO Model loaded from: {model_path}")

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
    Run autoregressive forecast for multiple days using TFNO model

    Predictions stay in preprocessed space (224x224) during the loop to minimize
    accumulation of round-off errors from repeated interpolation.

    Args:
        config: Configuration object
        model: Trained TFNO model
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
            rows = getattr(getattr(config, 'data', None), 'north_mask_rows', 0) if config else 0
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
        f.write("TFNO MULTI-DAY FORECAST EVALUATION METRICS\n")
        f.write("=" * 80 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Model: {config.name}\n")
        f.write(f"  Model Type: TFNO (Tensorized Fourier Neural Operator)\n")
        f.write(f"  TFNO Modes: {config.tfno.n_modes}\n")
        f.write(f"  TFNO Layers: {config.tfno.n_layers}\n")
        f.write(f"  Hidden Channels: {config.tfno.hidden_channels}\n")
        f.write(f"  Factorization: {config.tfno.factorization}\n")
        f.write(f"  Rank: {config.tfno.rank} (approx. {config.tfno.rank*100:.0f}% of dense FNO)\n")
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


def _run_eval_loop(config, model, device, ocean_data, atm_data,
                   start_date, num_days, reference_date, max_forecast_days, desc):
    """Run evaluation loop; return averaged metrics, raw metrics, and eval day count.

    Args:
        config: Configuration namespace.
        model: Trained model in eval mode.
        device (torch.device): Compute device.
        ocean_data (xr.Dataset): Ocean dataset.
        atm_data (xr.Dataset): Atmospheric dataset.
        start_date (str): First IC date (dd-mm-yyyy).
        num_days (int): Evaluation window length.
        reference_date (str): Date at index 0 in datasets (dd-mm-yyyy).
        max_forecast_days (int): Number of forecast lead days.
        desc (str): tqdm progress bar label.

    Returns:
        tuple: (averaged_metrics, raw_metrics, num_eval_days)

    Example:
        >>> avg, raw, n = _run_eval_loop(cfg, mdl, dev, oc, at, '01-01-2020', 365,
        ...                              '01-01-2020', 9, 'Primary')
    """
    start_idx     = date_to_day_index(start_date, reference_date)
    dataset_size  = len(ocean_data.time)
    num_eval_days = min(num_days - max_forecast_days + 1,
                        dataset_size - start_idx - max_forecast_days)
    raw = {lt: {var: [] for var in config.data.out_variable}
           for lt in range(1, max_forecast_days + 1)}
    skipped = 0
    for eval_day in tqdm(range(num_eval_days), desc=desc, file=sys.stdout,
                         dynamic_ncols=True):
        idx = start_idx + eval_day
        try:
            fc, md = run_autoregressive_forecast(config, model, idx, max_forecast_days,
                                                 device, ocean_data, atm_data)
        except ValueError as exc:
            tqdm.write(f"  Skipping day {eval_day}: {exc}")
            skipped += 1
            continue
        pp = postprocess_forecasts(fc, md, config)
        gt = load_ground_truth_batch(config, idx, max_forecast_days, ocean_data)
        metrics = compute_forecast_metrics(pp, gt, config)
        for lt in range(1, max_forecast_days + 1):
            for var in config.data.out_variable:
                raw[lt][var].append(metrics[lt][var])
    if skipped:
        tqdm.write(f"  NOTE: {skipped} day(s) skipped due to corrupted input data.")
    averaged = {}
    for lt in range(1, max_forecast_days + 1):
        averaged[lt] = {}
        for var in config.data.out_variable:
            ml = raw[lt][var]
            averaged[lt][var] = {
                'rmse':    np.mean([m['rmse']    for m in ml]),
                'mae':     np.mean([m['mae']     for m in ml]),
                'r2':      np.mean([m['r2']      for m in ml]),
                'pearson': np.mean([m['pearson'] for m in ml]),
            }
    return averaged, raw, num_eval_days


def main():
    """Main evaluation function for TFNO model."""
    parser = argparse.ArgumentParser(description='Run multi-day forecast evaluation for TFNO model')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--config_file', type=str, default='tfno_bob_config.yaml')
    parser.add_argument('--extra_data_dir',   type=str, default=None)
    parser.add_argument('--extra_start_date', type=str, default='01-01-2021')
    parser.add_argument('--extra_num_days',   type=int, default=1826)
    parser.add_argument('--extra_ref_date',   type=str, default='01-01-2021')
    parser.add_argument('--extra_ocean_file', type=str, default='ocean_2021_2025')
    parser.add_argument('--extra_atm_file',   type=str, default='atm_2021_2025')
    parser.add_argument('--combined_output',  action='store_true')
    parser.add_argument('--device', type=str, default=None,
                        help='Override config device (e.g. cuda:0)')
    args = parser.parse_args()

    pipe = ConfigPipeline([
        YamlConfig(f"./{args.config_file}", config_name='default', config_folder='config/'),
        YamlConfig(config_folder='config/')
    ])
    config = pipe.read_conf()
    if args.name:
        config.name = args.name

    max_forecast_days = config.evaluation.max_forecast_days
    start_date        = config.evaluation.start_date
    num_days          = config.evaluation.num_days
    reference_date    = config.evaluation.reference_date

    dev_str = args.device if args.device else config.device
    device = torch.device(dev_str if torch.cuda.is_available() else 'cpu')
    model_path = args.model_path or f"{config.results.model_dir}/{config.name}.pth"
    if not os.path.exists(model_path):
        print(f"Error: TFNO model not found at {model_path}")
        sys.exit(1)
    model = load_model(config, model_path, device)

    data_dir   = config.data.data_dir
    ocean_data = xr.open_dataset(Path(data_dir) / f'{config.data.file_prefix}.nc')
    atm_data   = xr.open_dataset(Path(data_dir) / f'{config.data.file_prefix_atm}.nc')

    out_dir = Path(config.results.save_dir) / config.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Primary evaluation ({start_date}) ===")
    avg, primary_raw, num_eval_days = _run_eval_loop(
        config, model, device, ocean_data, atm_data,
        start_date, num_days, reference_date, max_forecast_days, desc='Primary')
    save_forecast_metrics(avg, str(out_dir / 'forecast_metrics.txt'),
                          config, start_date, num_eval_days)
    print(f"Saved: {out_dir}/forecast_metrics.txt")
    ocean_data.close(); atm_data.close()

    ext_raw = None; ext_eval_days = 0
    if args.extra_data_dir:
        extra_dir   = Path(args.extra_data_dir)
        extra_ocean = xr.open_dataset(extra_dir / f'{args.extra_ocean_file}.nc')
        extra_atm   = xr.open_dataset(extra_dir / f'{args.extra_atm_file}.nc')

        print(f"\n=== Extended evaluation ({args.extra_start_date}) ===")
        ext_avg, ext_raw, ext_eval_days = _run_eval_loop(
            config, model, device, extra_ocean, extra_atm,
            args.extra_start_date, args.extra_num_days, args.extra_ref_date,
            max_forecast_days, desc='Extended')
        save_forecast_metrics(ext_avg, str(out_dir / 'forecast_metrics_extended.txt'),
                              config, args.extra_start_date, ext_eval_days)
        print(f"Saved: {out_dir}/forecast_metrics_extended.txt")
        extra_ocean.close(); extra_atm.close()

    if args.combined_output and ext_raw is not None:
        comb_raw = {
            lt: {var: primary_raw[lt][var] + ext_raw[lt][var]
                 for var in config.data.out_variable}
            for lt in range(1, max_forecast_days + 1)
        }
        comb_avg = {}
        for lt in range(1, max_forecast_days + 1):
            comb_avg[lt] = {}
            for var in config.data.out_variable:
                ml = comb_raw[lt][var]
                comb_avg[lt][var] = {
                    'rmse':    np.mean([m['rmse']    for m in ml]),
                    'mae':     np.mean([m['mae']     for m in ml]),
                    'r2':      np.mean([m['r2']      for m in ml]),
                    'pearson': np.mean([m['pearson'] for m in ml]),
                }
        label = f"{start_date} to {args.extra_start_date}+{args.extra_num_days}d"
        save_forecast_metrics(comb_avg, str(out_dir / 'forecast_metrics_combined.txt'),
                              config, label, num_eval_days + ext_eval_days)
        print(f"Saved: {out_dir}/forecast_metrics_combined.txt")

    print("\n=== TFNO Evaluation Complete ===")


if __name__ == "__main__":
    main()

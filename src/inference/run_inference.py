"""
Inference script for autoregressive ocean state forecasting
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

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from models.architectures.afno.afnonet import AFNONet
from data_pipeline.preprocessing.transformers.normalize_transform import PreprocessTransform
from visualization.ocean_plotter import OceanPlotter
from inference.utils import (
    postprocess_ocean_variable,
    compute_metrics,
    save_metrics_to_file,
    date_to_day_index
)


def load_model(config, model_path, device):
    """
    Load trained model from checkpoint

    Args:
        config: Configuration object
        model_path: Path to model checkpoint (.pth file)
        device: Device to load model on

    Returns:
        Loaded model
    """
    model = AFNONet(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from: {model_path}")
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
    Preprocess ocean variable (normalize and interpolate)

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
    Preprocess atmospheric variable (normalize and interpolate)

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


def load_initial_ocean_state(config, day_index, transform, mean_dict):
    """
    Load and preprocess initial ocean state at time t

    Args:
        config: Configuration object
        day_index: Day index in the dataset
        transform: Preprocessing transform
        mean_dict: Dictionary of mean values

    Returns:
        Dictionary of preprocessed ocean variables (each 1xHxW at 224x224)
    """
    data_dir = config.data.data_dir
    ocean_file = Path(data_dir) / f'{config.data.file_prefix}.nc'
    ocean_data = xr.open_dataset(ocean_file)

    ocean_state = {}
    for ocean_var in config.data.variable:
        ocean_input = ocean_data[ocean_var][day_index:day_index+1].values
        ocean_input = preprocess_ocean_variable(ocean_input, mean_dict[ocean_var],
                                               ocean_var, transform, config)
        ocean_state[ocean_var] = ocean_input

    return ocean_state


def load_atmospheric_forcing(config, day_index, transform, mean_dict, variance_dict):
    """
    Load and preprocess atmospheric forcing at time t+1

    Args:
        config: Configuration object
        day_index: Day index for t+1
        transform: Preprocessing transform
        mean_dict: Dictionary of mean values
        variance_dict: Dictionary of variance values

    Returns:
        Concatenated atmospheric tensor (6, H, W) at 224x224
    """
    data_dir = config.data.data_dir
    atm_file = Path(data_dir) / f'{config.data.file_prefix_atm}.nc'
    atm_data = xr.open_dataset(atm_file)

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


def load_ground_truth(config, day_index, mean_dict):
    """
    Load ground truth ocean data at original resolution (not preprocessed)

    Args:
        config: Configuration object
        day_index: Day index in the dataset
        mean_dict: Dictionary of mean values (for getting original size)

    Returns:
        Dictionary of ground truth arrays for each output variable
    """
    data_dir = config.data.data_dir
    ocean_file = Path(data_dir) / f'{config.data.file_prefix}.nc'
    ocean_data = xr.open_dataset(ocean_file)

    ground_truth = {}
    for var in config.data.out_variable:
        # Load data at original resolution
        gt_data = ocean_data[var][day_index:day_index+1].values.squeeze()
        ground_truth[var] = gt_data

    return ground_truth


def run_autoregressive_forecast(config, model, day_index, num_days, device):
    """
    Run autoregressive forecasting for specified number of days

    Args:
        config: Configuration object
        model: Trained model
        day_index: Starting day index
        num_days: Number of forecast days
        device: Device for computation

    Returns:
        Dictionary mapping lead_time -> {variable: preprocessed_prediction (224x224)}
    """
    # Load normalization statistics
    mean_dict, variance_dict = load_normalization_stats(config)

    # Create transform
    transform = PreprocessTransform(config)

    # Load initial ocean state at time t (preprocessed, 224x224)
    print(f"\nLoading initial ocean state for day index: {day_index}")
    ocean_state = load_initial_ocean_state(config, day_index, transform, mean_dict)

    # Store predictions for each lead time (in preprocessed space)
    forecast_dict = {}

    # Autoregressive loop for lead_time in range(1, num_days+1)
    for lead_time in range(1, num_days + 1):
        print(f"\nForecasting lead time {lead_time}/{num_days}...")

        # Load atmospheric forcing at t+lead_time
        atm_forcing = load_atmospheric_forcing(config, day_index + lead_time,
                                               transform, mean_dict, variance_dict)

        # Concatenate ocean state + atmospheric forcing
        ocean_vars = [ocean_state[var] for var in config.data.variable]
        ocean_tensor = torch.cat(ocean_vars, dim=0)  # (5, H, W)
        input_tensor = torch.cat([atm_forcing, ocean_tensor], dim=0)  # (11, H, W)

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


def main():
    """Main inference function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run autoregressive ocean state forecasting')
    parser.add_argument('--input_date', type=str, default=None,
                       help='Input date in format dd-mm-yyyy (default: from config)')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name (overrides config name)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model checkpoint (default: results/models/{name}.pth)')
    args = parser.parse_args()

    # Load configuration
    print("=== Loading Configuration ===")
    pipe = ConfigPipeline([
        YamlConfig("./afno_bob_config.yaml", config_name='default', config_folder='config/'),
        YamlConfig(config_folder='config/')
    ])
    config = pipe.read_conf()

    # Override name if provided via command line
    if args.name:
        config.name = args.name
        print(f"Overriding config name with: {args.name}")

    # Get input date
    input_date = args.input_date if args.input_date else config.plot.input_date
    print(f"Input date: {input_date}")

    # Get num_days from config
    num_days = config.plot.num_days
    print(f"Number of forecast days: {num_days}")

    # Convert date to day index
    reference_date = "01-01-1993"
    day_index = date_to_day_index(input_date, reference_date)
    print(f"Day index: {day_index}")

    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = f"{config.results.model_dir}/{config.name}.pth"

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    model = load_model(config, model_path, device)

    # Run autoregressive forecast (returns preprocessed predictions)
    print(f"\n=== Running Autoregressive Forecast ({num_days} days) ===")
    forecast_dict, mean_dict = run_autoregressive_forecast(config, model, day_index,
                                                           num_days, device)

    # Postprocess all forecasts at once
    print("\n=== Postprocessing Forecasts ===")
    postprocessed_forecasts = postprocess_forecasts(forecast_dict, mean_dict, config)

    # Initialize plotter
    plotter = OceanPlotter(config)

    # Get color axis limits from config
    vmin_dict = {}
    vmax_dict = {}

    use_default = True
    if hasattr(config, 'visualization'):
        use_default = getattr(config.visualization, 'use_default', True)

        if not use_default:
            vmin_dict = config.visualization.vmin
            vmax_dict = config.visualization.vmax
            print("Using custom color axis limits from config")

    # Process each forecast lead time
    for lead_time in range(1, num_days + 1):
        print(f"\n=== Processing Lead Time {lead_time}/{num_days} ===")

        # Calculate prediction date
        input_date_obj = datetime.strptime(input_date, "%d-%m-%Y")
        prediction_date_obj = input_date_obj + timedelta(days=lead_time)
        prediction_date = prediction_date_obj.strftime("%d-%m-%Y")
        print(f"Prediction date (t+{lead_time}): {prediction_date}")

        # Load ground truth for this lead time
        print("Loading ground truth...")
        ground_truth = load_ground_truth(config, day_index + lead_time, mean_dict)

        # Create output directory
        output_dir = Path(config.results.save_dir) / config.name / input_date / f"lead_time_{lead_time}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving results to: {output_dir}")

        # Get predictions for this lead time
        predictions = postprocessed_forecasts[lead_time]

        # Compute metrics and plot for each variable
        metrics_dict = {}

        for var in config.data.out_variable:
            print(f"\nProcessing {var}...")

            pred = predictions[var]
            truth = ground_truth[var]

            # Compute metrics
            metrics = compute_metrics(pred, truth)
            metrics_dict[var] = metrics

            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  R²: {metrics['r2']:.6f}")
            print(f"  Pearson: {metrics['pearson']:.6f}")

            # Get color limits for this variable
            vmin = vmin_dict.get(var, None) if isinstance(vmin_dict, dict) else None
            vmax = vmax_dict.get(var, None) if isinstance(vmax_dict, dict) else None

            # Plot comparison
            save_path = output_dir / f"{var}_comparison.png"
            plotter.plot_comparison(pred, truth, var, prediction_date,
                                   str(save_path), vmin=vmin, vmax=vmax)
            print(f"  Saved: {save_path}")

        # Save metrics to file
        metrics_file = output_dir / "metrics.txt"
        save_metrics_to_file(metrics_dict, str(metrics_file))
        print(f"\nMetrics saved to: {metrics_file}")

    print("\n=== Autoregressive Forecasting Complete ===")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Multi-panel Ocean Forecast Visualization Script

This script generates multipanel plots for oceanic variables (SST, SSS, SSC, SSH)
at different forecast lead times. It uses autoregressive forecasting to generate
predictions and compares them with ground truth.

Supports multiple model architectures:
    - AFNO (Adaptive Fourier Neural Operator)
    - FNO (Fourier Neural Operator)
    - TFNO (Tensorized Fourier Neural Operator)
    - UNO (U-shaped Neural Operator)

Model type is automatically detected from the model name prefix (e.g., AFNO_*, FNO_*, etc.)

Usage:
    python make_plots.py --plot_config config/plot_config.yaml
    python make_plots.py --plot_config config/plot_config.yaml --model_config_only

Configuration:
    Plot config file must contain:
    - model_name: Name of the model to use (e.g., 'AFNO_BoB_Surf_E01', 'FNO_BoB_Surf_E02')
    - input_date: Date for generating forecasts (format: dd-mm-yyyy)
    - reference_date: Reference date for day index calculation
    - num_days: Number of forecast days
    - plot_limits: Optional colorbar limits for each variable
    - colormap: Optional colormaps for each variable
"""

import os
import sys
import yaml
import argparse
import subprocess
import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from matplotlib import gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from configmypy import ConfigPipeline, YamlConfig
from src.models.architectures.afno.afnonet import AFNONet
from neuralop.models import FNO, TFNO, UNO
from src.data_pipeline.preprocessing.transformers.normalize_transform import PreprocessTransform
from src.data_pipeline.loaders.data_loader import create_mask
from src.inference.utils import (
    postprocess_ocean_variable,
    date_to_day_index
)
import torch.nn as nn
import torch.nn.functional as F


def apply_mask_to_data(data, mask):
    """
    Apply land-sea mask to data, setting land values to NaN

    Args:
        data: Data array (H, W)
        mask: Binary mask (1, C, H, W) or (C, H, W) - 1 for ocean, 0 for land

    Returns:
        Masked data with land values set to NaN
    """
    # Handle different mask shapes
    if mask.ndim == 4:
        # (1, C, H, W) - take first batch and first channel
        mask_2d = mask[0, 0, :, :]
    elif mask.ndim == 3:
        # (C, H, W) - take first channel
        mask_2d = mask[0, :, :]
    elif mask.ndim == 2:
        # (H, W) - already 2D
        mask_2d = mask
    else:
        print(f"Warning: Unexpected mask shape {mask.shape}, not applying mask")
        return data

    # Ensure data is 2D
    if data.ndim > 2:
        print(f"Warning: Data should be 2D but has shape {data.shape}")
        return data

    # Apply mask: set land (mask=0) to NaN
    masked_data = np.where(mask_2d == 0, np.nan, data)

    return masked_data


class FNOWrapper(nn.Module):
    """Wrapper around neuralop FNO to match the interface expected by the inference pipeline."""
    def __init__(self, config):
        super().__init__()

        n_modes = tuple(config.fno.n_modes)
        hidden_channels = config.fno.hidden_channels
        lifting_channel_ratio = config.fno.lifting_channel_ratio
        projection_channel_ratio = config.fno.projection_channel_ratio
        n_layers = config.fno.n_layers

        non_linearity = F.gelu  # default
        if hasattr(config.fno, 'non_linearity'):
            if config.fno.non_linearity == 'gelu':
                non_linearity = F.gelu
            elif config.fno.non_linearity == 'relu':
                non_linearity = F.relu
            elif config.fno.non_linearity == 'tanh':
                non_linearity = torch.tanh

        self.fno = FNO(
            n_modes=n_modes,
            in_channels=config.data.in_chs,
            out_channels=config.data.out_chs,
            hidden_channels=hidden_channels,
            lifting_channel_ratio=lifting_channel_ratio,
            projection_channel_ratio=projection_channel_ratio,
            n_layers=n_layers,
            non_linearity=non_linearity,
            use_channel_mlp=config.fno.use_channel_mlp,
            channel_mlp_expansion=config.fno.channel_mlp_expansion,
            channel_mlp_dropout=config.fno.channel_mlp_dropout,
            channel_mlp_skip=config.fno.channel_mlp_skip,
        )

    def forward(self, x):
        return self.fno(x)


class TFNOWrapper(nn.Module):
    """Wrapper around neuralop TFNO to match the interface expected by the inference pipeline."""
    def __init__(self, config):
        super().__init__()

        n_modes = tuple(config.tfno.n_modes)
        hidden_channels = config.tfno.hidden_channels
        lifting_channel_ratio = config.tfno.lifting_channel_ratio
        projection_channel_ratio = config.tfno.projection_channel_ratio
        n_layers = config.tfno.n_layers
        factorization = config.tfno.factorization
        rank = config.tfno.rank

        non_linearity = F.gelu  # default
        if hasattr(config.tfno, 'non_linearity'):
            if config.tfno.non_linearity == 'gelu':
                non_linearity = F.gelu
            elif config.tfno.non_linearity == 'relu':
                non_linearity = F.relu
            elif config.tfno.non_linearity == 'tanh':
                non_linearity = torch.tanh

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
        )

    def forward(self, x):
        return self.tfno(x)


class UNOWrapper(nn.Module):
    """Wrapper around neuralop UNO to match the interface expected by the inference pipeline."""
    def __init__(self, config):
        super().__init__()

        n_modes = tuple(config.uno.n_modes)
        hidden_channels = config.uno.hidden_channels
        lifting_channel_ratio = config.uno.lifting_channel_ratio
        projection_channel_ratio = config.uno.projection_channel_ratio
        n_layers = config.uno.n_layers
        uno_out_channels = config.uno.uno_out_channels
        uno_n_modes = tuple(config.uno.uno_n_modes)
        uno_scalings = tuple(config.uno.uno_scalings)
        horizontal_skips_map = config.uno.horizontal_skips_map if hasattr(config.uno, 'horizontal_skips_map') else None

        non_linearity = F.gelu  # default
        if hasattr(config.uno, 'non_linearity'):
            if config.uno.non_linearity == 'gelu':
                non_linearity = F.gelu
            elif config.uno.non_linearity == 'relu':
                non_linearity = F.relu
            elif config.uno.non_linearity == 'tanh':
                non_linearity = torch.tanh

        self.uno = UNO(
            n_modes=n_modes,
            in_channels=config.data.in_chs,
            out_channels=config.data.out_chs,
            hidden_channels=hidden_channels,
            lifting_channel_ratio=lifting_channel_ratio,
            projection_channel_ratio=projection_channel_ratio,
            n_layers=n_layers,
            non_linearity=non_linearity,
            uno_out_channels=uno_out_channels,
            uno_n_modes=uno_n_modes,
            uno_scalings=uno_scalings,
            horizontal_skips_map=horizontal_skips_map,
        )

    def forward(self, x):
        return self.uno(x)


def load_plot_config(config_path):
    """
    Load plot configuration from YAML file

    Args:
        config_path: Path to plot config file

    Returns:
        Dictionary with plot configuration
    """
    with open(config_path, 'r') as f:
        plot_config = yaml.safe_load(f)

    # Validate required fields
    required_fields = ['model_name', 'input_date', 'reference_date', 'num_days']
    for field in required_fields:
        if field not in plot_config:
            raise ValueError(f"Plot config must contain '{field}' field")

    return plot_config


def generate_model_config(model_name, project_root):
    """
    Locate or generate model config YAML file.

    Priority order:
    1. Use manually created config in config/ (if exists)
    2. Use previously generated config in config/generated/ (if exists)
    3. Generate new config from runtable.csv (only if neither exists)

    Args:
        model_name: Name of the model
        project_root: Project root directory

    Returns:
        Path to config file (existing or newly generated)
    """
    print(f"=== Locating Model Config for {model_name} ===")

    # Priority 1: Check for manually created config in config/
    config_file = project_root / 'config' / f'{model_name.lower()}.yaml'
    if config_file.exists():
        print(f"Using existing config: {config_file}")
        return config_file

    # Priority 2: Check for previously generated config in config/generated/
    generated_config = project_root / 'config' / 'generated' / f'{model_name.lower()}.yaml'
    if generated_config.exists():
        print(f"Using generated config: {generated_config}")
        return generated_config

    # Priority 3: Generate new config from runtable.csv
    print(f"Config not found, generating from runtable.csv...")

    # Detect model type and select appropriate script
    model_type = detect_model_type(model_name)

    # Map model type to script name
    script_map = {
        'AFNO': 'csv_to_yaml_afno.py',
        'FNO': 'csv_to_yaml_fno.py',
        'TFNO': 'csv_to_yaml_tfno.py',
        'UNO': 'csv_to_yaml_uno.py'
    }

    script_name = script_map.get(model_type)
    if not script_name:
        raise ValueError(f"No YAML generation script for model type: {model_type}")

    csv_to_yaml_script = project_root / script_name
    if not csv_to_yaml_script.exists():
        raise FileNotFoundError(f"{script_name} not found at {csv_to_yaml_script}")

    print(f"Using {script_name} for {model_type} model")

    cmd = ['python', str(csv_to_yaml_script), '--name', model_name]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))

    if result.returncode != 0:
        print("Error generating config file:")
        print(result.stderr)
        sys.exit(1)

    print(result.stdout)

    # Check if config was generated
    if generated_config.exists():
        return generated_config
    elif config_file.exists():
        return config_file
    else:
        raise FileNotFoundError(f"Config file could not be found or generated for {model_name}")


def detect_model_type(model_name):
    """
    Detect model type from model name prefix.

    Args:
        model_name: Name of the model (e.g., 'AFNO_BoB_Surf_E01', 'FNO_BoB_Surf_E01')

    Returns:
        Model type string: 'AFNO', 'FNO', 'TFNO', or 'UNO'
    """
    name_upper = model_name.upper()
    if name_upper.startswith('AFNO'):
        return 'AFNO'
    elif name_upper.startswith('TFNO'):
        return 'TFNO'
    elif name_upper.startswith('UNO'):
        return 'UNO'
    elif name_upper.startswith('FNO'):
        return 'FNO'
    else:
        raise ValueError(f"Cannot detect model type from name: {model_name}. "
                        f"Expected name to start with AFNO, FNO, TFNO, or UNO")


def load_model(config, model_path, device, model_name):
    """
    Load trained model from checkpoint (supports AFNO, FNO, TFNO, UNO)

    Args:
        config: Configuration object
        model_path: Path to model checkpoint
        device: Device to load model on
        model_name: Name of the model (used to detect model type)

    Returns:
        Loaded model
    """
    # Detect model type from name
    model_type = detect_model_type(model_name)
    print(f"Detected model type: {model_type}")

    # Instantiate the appropriate model
    if model_type == 'AFNO':
        model = AFNONet(config)
    elif model_type == 'FNO':
        model = FNOWrapper(config)
    elif model_type == 'TFNO':
        model = TFNOWrapper(config)
    elif model_type == 'UNO':
        model = UNOWrapper(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            # Full checkpoint with training state
            state_dict = checkpoint['model_state_dict']
            print(f"Model loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')}): {model_path}")
        else:
            # Direct state dict
            state_dict = checkpoint
            print(f"Model loaded from: {model_path}")

        # Clean up state dict - remove PyTorch internal metadata (keys starting with '_')
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('_')}

        # Try loading with error handling
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"Warning: Strict loading failed, trying with strict=False: {e}")
            model.load_state_dict(state_dict, strict=False)
    else:
        # Legacy format - direct state dict
        state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('_')}
        model.load_state_dict(state_dict)
        print(f"Model loaded from: {model_path}")

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


def preprocess_ocean_variable(data, mean, variable, transform):
    """Preprocess ocean variable (normalize and interpolate to 224x224)"""
    processed = transform(data, mean, variable=variable)

    # Mask northern boundary (same as training)
    rows = config.data.north_mask_rows
    if rows > 0:
        processed[:, :, -rows:, :] = 0.0

    return processed.squeeze(1)  # Remove batch dim: (1, H, W)


def preprocess_atm_variable(data, mean, variable, transform, variance=None):
    """Preprocess atmospheric variable (normalize and interpolate to 224x224)"""
    if variable in ['ssr', 'tp', 'msl'] and variance is not None:
        processed = transform(data, mean, variable=variable, type='atm', variance=variance)
    else:
        processed = transform(data, mean, variable=variable, type='atm')

    return processed.squeeze(1)  # Remove batch dim: (1, H, W)


def load_initial_ocean_state(config, day_index, ocean_data, transform, mean_dict):
    """Load and preprocess initial ocean state at time t"""
    ocean_state = {}
    for ocean_var in config.data.variable:
        ocean_input = ocean_data[ocean_var][day_index:day_index+1].values
        ocean_input = preprocess_ocean_variable(ocean_input, mean_dict[ocean_var],
                                               ocean_var, transform)
        ocean_state[ocean_var] = ocean_input

    return ocean_state


def load_atmospheric_forcing(config, day_index, atm_data, transform, mean_dict, variance_dict):
    """Load and preprocess atmospheric forcing at time t+1"""
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
    """Postprocess all forecasts (reverse normalization and interpolation)"""
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
    """Load ground truth for multiple forecast days"""
    ground_truth_dict = {}

    for lead_time in range(1, forecast_days + 1):
        day_idx = start_day_index + lead_time
        ground_truth = {}

        for var in config.data.out_variable:
            gt_data = ocean_data[var][day_idx:day_idx+1].values.squeeze()
            ground_truth[var] = gt_data

        ground_truth_dict[lead_time] = ground_truth

    return ground_truth_dict


def get_colorbar_limits(data_list, plot_config, variable):
    """
    Get colorbar limits from config or compute from data

    Args:
        data_list: List of data arrays to compute limits from
        plot_config: Plot configuration dictionary
        variable: Variable name

    Returns:
        Tuple of (vmin, vmax)
    """
    # Check colorbar_mode in config (default: 'percentile')
    colorbar_mode = plot_config.get('colorbar_mode', 'percentile')

    if colorbar_mode == 'fixed':
        # Use fixed limits from config
        if 'plot_limits' in plot_config and variable in plot_config['plot_limits']:
            limits = plot_config['plot_limits'][variable]
            return limits['vmin'], limits['vmax']
        else:
            print(f"Warning: colorbar_mode='fixed' but no limits specified for {variable}, falling back to percentile")
            colorbar_mode = 'percentile'

    # Compute from data using percentiles
    if colorbar_mode == 'percentile':
        all_data = np.concatenate([d.flatten() for d in data_list])
        all_data = all_data[~np.isnan(all_data)]

        percentile = plot_config.get('percentile', [10, 90])
        vmin = np.percentile(all_data, percentile[0])
        vmax = np.percentile(all_data, percentile[1])

        return vmin, vmax

    # Fallback
    return 0, 1


def get_colormap(plot_config, variable, diff=False):
    """
    Get colormap from config or use default

    Args:
        plot_config: Plot configuration dictionary
        variable: Variable name
        diff: If True, return diverging colormap for differences

    Returns:
        Colormap name
    """
    if diff:
        return plot_config.get('diff_colormap', 'RdBu_r')

    # Default colormaps
    default_cmaps = {
        'thetao': 'RdYlBu_r',  # SST
        'so': 'viridis',        # SSS
        'uo': 'RdBu_r',         # Zonal current
        'vo': 'RdBu_r',         # Meridional current
        'zos': 'coolwarm'       # SSH
    }

    # Check config
    if 'colormap' in plot_config and variable in plot_config['colormap']:
        return plot_config['colormap'][variable]

    return default_cmaps.get(variable, 'viridis')


def create_multipanel_plot(predictions_dict, ground_truth_dict, plot_config, config,
                          input_date, save_dir, ocean_data, mask):
    """
    Create multipanel plots for oceanic variables

    Creates 4 rows (SST, SSS, SSC, SSH) x 3 columns (Forecast, Truth, Relative Diff)
    Shows forecast leads: 1, 3, 5, 7, 9 days (if available)

    Args:
        predictions_dict: Dictionary mapping lead_time -> {variable: prediction}
        ground_truth_dict: Dictionary mapping lead_time -> {variable: ground_truth}
        plot_config: Plot configuration
        config: Model configuration
        input_date: Input date string
        save_dir: Directory to save plots
        ocean_data: xarray Dataset with coordinate information
        mask: Land-sea mask to apply to data
    """
    # Lead times to plot
    lead_times = [1, 3, 5, 7, 9]
    available_leads = [lt for lt in lead_times if lt in predictions_dict]

    if not available_leads:
        print("Warning: No forecast leads available for plotting")
        return

    # Variables to plot with their names
    variables = [
        ('thetao', 'SST'),
        ('so', 'SSS'),
        (('uo', 'vo'), 'SSC'),  # Vector field
        ('zos', 'SSH')
    ]

    # Lon/lat ranges (Bay of Bengal)
    lon_range = plot_config.get('lon_range', [80, 100])
    lat_range = plot_config.get('lat_range', [5, 25])

    # Load actual coordinates from ocean data
    if 'lon' in ocean_data.coords:
        lons = ocean_data.coords['lon'].values
        lats = ocean_data.coords['lat'].values
    elif 'longitude' in ocean_data.coords:
        lons = ocean_data.coords['longitude'].values
        lats = ocean_data.coords['latitude'].values
    else:
        # Fallback: use linspace if coordinates not found
        print("Warning: Could not find lon/lat coordinates in ocean data, using linspace")
        # Get shape from first available data
        first_var = list(predictions_dict[available_leads[0]].keys())[0]
        data_shape = predictions_dict[available_leads[0]][first_var].shape
        lons = np.linspace(lon_range[0], lon_range[1], data_shape[1])
        lats = np.linspace(lat_range[0], lat_range[1], data_shape[0])

    # Debug: Print coordinate info
    print(f"Coordinate info - Lons shape: {lons.shape}, Lats shape: {lats.shape}")

    # Handle 1D vs 2D coordinates
    if lons.ndim == 1 and lats.ndim == 1:
        # 1D coordinates - create meshgrid
        lon_grid, lat_grid = np.meshgrid(lons, lats)
    elif lons.ndim == 2 and lats.ndim == 2:
        # Already 2D grids
        lon_grid, lat_grid = lons, lats
    else:
        print(f"Warning: Unexpected coordinate dimensions. Lons: {lons.ndim}D, Lats: {lats.ndim}D")
        # Fallback to creating meshgrid from 1D arrays
        if lons.ndim == 2:
            lons = lons[0, :]  # Take first row
        if lats.ndim == 2:
            lats = lats[:, 0]  # Take first column
        lon_grid, lat_grid = np.meshgrid(lons, lats)

    print(f"Grid shape - Lon_grid: {lon_grid.shape}, Lat_grid: {lat_grid.shape}")

    # Verify data shape matches grid
    first_var = list(predictions_dict[available_leads[0]].keys())[0]
    first_data_shape = predictions_dict[available_leads[0]][first_var].shape
    print(f"First data shape: {first_data_shape}")

    if first_data_shape != lon_grid.shape:
        print(f"WARNING: Data shape {first_data_shape} does not match grid shape {lon_grid.shape}")
        print("This may cause geography rendering issues!")

    # For each variable, create a multipanel plot
    for var_info, var_name in variables:
        is_vector = isinstance(var_info, tuple) and len(var_info) == 2

        if is_vector:
            var_u, var_v = var_info
            var_key = 'uo'  # Use 'uo' for colorbar limits (magnitude of currents)
        else:
            var_key = var_info

        print(f"Creating {var_name} plot...")

        # Create figure: 7 inches wide x 9 inches tall
        n_rows = len(available_leads)
        fig = plt.figure(figsize=(7, 9))

        # Create gridspec with extra row at bottom for colorbars
        gs = gridspec.GridSpec(n_rows + 1, 3, figure=fig,
                               height_ratios=[1] * n_rows + [0.05],
                               hspace=0.25, wspace=0.25)

        projection = ccrs.PlateCarree()

        # Collect all data for consistent color limits (with mask applied)
        all_forecast_data = []
        all_truth_data = []

        for lead in available_leads:
            if is_vector:
                # For vector field, use magnitude
                u_pred = apply_mask_to_data(predictions_dict[lead][var_u], mask)
                v_pred = apply_mask_to_data(predictions_dict[lead][var_v], mask)
                mag_pred = np.sqrt(u_pred**2 + v_pred**2)

                u_truth = apply_mask_to_data(ground_truth_dict[lead][var_u], mask)
                v_truth = apply_mask_to_data(ground_truth_dict[lead][var_v], mask)
                mag_truth = np.sqrt(u_truth**2 + v_truth**2)

                all_forecast_data.append(mag_pred)
                all_truth_data.append(mag_truth)
            else:
                all_forecast_data.append(apply_mask_to_data(predictions_dict[lead][var_key], mask))
                all_truth_data.append(apply_mask_to_data(ground_truth_dict[lead][var_key], mask))

        # Get global colorbar limits
        vmin, vmax = get_colorbar_limits(
            all_forecast_data + all_truth_data,
            plot_config, var_key
        )

        # Get colormap
        cmap = get_colormap(plot_config, var_key)
        # Always use RdBu_r for relative difference (blue to red with white at 0)
        # Using _r (reversed) means blue=negative, white=0, red=positive
        diff_cmap = 'RdBu_r'

        # Compute global symmetric limits for relative difference
        # across all lead times to ensure consistent colorbar
        all_rel_diff_data = []
        eps = 1
        for i, (forecast, truth) in enumerate(zip(all_forecast_data, all_truth_data)):
            rel_diff = (forecast - truth) / (np.abs(truth) + eps)
            all_rel_diff_data.append(rel_diff)

        # Get percentiles and make symmetric
        rel_diff_percentile = plot_config.get('rel_diff_percentile', [0, 100])
        all_rel_diff_flat = np.concatenate([rd.flatten() for rd in all_rel_diff_data])
        all_rel_diff_flat = all_rel_diff_flat[~np.isnan(all_rel_diff_flat)]

        rel_vmin_global = np.percentile(all_rel_diff_flat, rel_diff_percentile[0])
        rel_vmax_global = np.percentile(all_rel_diff_flat, rel_diff_percentile[1])

        # Make symmetric around 0
        rel_vlim_global = max(abs(rel_vmin_global), abs(rel_vmax_global))
        rel_vmin_global = -rel_vlim_global
        rel_vmax_global = rel_vlim_global

        print(f"  {var_name} relative difference limits: [{rel_vmin_global:.3f}, {rel_vmax_global:.3f}]")

        # Create explicit contour levels for forecast and ground truth
        forecast_levels = np.linspace(vmin, vmax, 21)

        # Store axes for colorbar creation
        all_axes_col1 = []
        all_axes_col2 = []
        all_axes_col3 = []
        last_im1 = None
        last_im2 = None
        last_im3 = None

        # Plot each lead time
        for row_idx, lead in enumerate(available_leads):
            if is_vector:
                # Vector field with magnitude as background
                u_pred = predictions_dict[lead][var_u]
                v_pred = predictions_dict[lead][var_v]

                # Apply mask to vector components
                u_pred = apply_mask_to_data(u_pred, mask)
                v_pred = apply_mask_to_data(v_pred, mask)

                u_truth = ground_truth_dict[lead][var_u]
                v_truth = ground_truth_dict[lead][var_v]

                # Apply mask to truth vector components
                u_truth = apply_mask_to_data(u_truth, mask)
                v_truth = apply_mask_to_data(v_truth, mask)

                # Compute magnitude (NaN will propagate)
                mag_pred = np.sqrt(u_pred**2 + v_pred**2)
                mag_truth = np.sqrt(u_truth**2 + v_truth**2)

                forecast_data = mag_pred
                truth_data = mag_truth
            else:
                # Apply mask to scalar variables
                forecast_data = apply_mask_to_data(predictions_dict[lead][var_key], mask)
                truth_data = apply_mask_to_data(ground_truth_dict[lead][var_key], mask)

            # Compute relative difference
            # Use global limits computed earlier for consistency across all rows
            rel_diff = (forecast_data - truth_data) / (np.abs(truth_data) + eps)

            # Column 1: Forecast
            ax1 = fig.add_subplot(gs[row_idx, 0], projection=projection)
            setup_map(ax1, lon_range, lat_range)

            im1 = ax1.contourf(lon_grid, lat_grid, forecast_data,
                              levels=forecast_levels, cmap=cmap, vmin=vmin, vmax=vmax,
                              transform=projection, extend='both')

            if is_vector:
                # Overlay vector field (subsample for clarity)
                skip = 10
                ax1.quiver(lon_grid[::skip, ::skip], lat_grid[::skip, ::skip],
                          u_pred[::skip, ::skip], v_pred[::skip, ::skip],
                          transform=projection, scale=5, width=0.002, alpha=0.7)

            if row_idx == 0:
                ax1.set_title('Forecast', fontsize=10, fontweight='bold')
            ax1.text(0.02, 0.98, f'Day +{lead}', transform=ax1.transAxes,
                    fontsize=8, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Column 2: Ground Truth
            ax2 = fig.add_subplot(gs[row_idx, 1], projection=projection)
            setup_map(ax2, lon_range, lat_range)

            im2 = ax2.contourf(lon_grid, lat_grid, truth_data,
                              levels=forecast_levels, cmap=cmap, vmin=vmin, vmax=vmax,
                              transform=projection, extend='both')

            if is_vector:
                # Overlay vector field
                ax2.quiver(lon_grid[::skip, ::skip], lat_grid[::skip, ::skip],
                          u_truth[::skip, ::skip], v_truth[::skip, ::skip],
                          transform=projection, scale=5, width=0.002, alpha=0.7)

            if row_idx == 0:
                ax2.set_title('Ground Truth', fontsize=10, fontweight='bold')

            # Column 3: Relative Difference
            ax3 = fig.add_subplot(gs[row_idx, 2], projection=projection)
            setup_map(ax3, lon_range, lat_range)

            # Create explicit levels for better colorbar control
            rel_diff_levels = np.linspace(rel_vmin_global, rel_vmax_global, 21)
            im3 = ax3.contourf(lon_grid, lat_grid, rel_diff,
                              levels=rel_diff_levels, cmap=diff_cmap,
                              vmin=rel_vmin_global, vmax=rel_vmax_global,
                              transform=projection, extend='both')

            if row_idx == 0:
                ax3.set_title('Relative Difference\n(Pred - Truth) / (Truth + eps)',
                            fontsize=10, fontweight='bold')

            # Store axes and image mappables
            all_axes_col1.append(ax1)
            all_axes_col2.append(ax2)
            all_axes_col3.append(ax3)
            last_im1 = im1
            last_im2 = im2
            last_im3 = im3

        # Add colorbars at the bottom
        # Colorbar for forecast and truth (spanning columns 1 and 2)
        cax1 = fig.add_subplot(gs[n_rows, 0:2])
        cbar1 = plt.colorbar(last_im2, cax=cax1, orientation='horizontal',
                            extend='both')
        cbar1.set_label('Magnitude (m/s)' if is_vector else get_units(var_key),
                       fontsize=9, labelpad=8)
        cbar1.ax.tick_params(labelsize=8, pad=4, length=4, width=1)

        # Set explicit colorbar limits to match the global data limits
        cbar1.mappable.set_clim(vmin=vmin, vmax=vmax)

        # Set explicit ticks for forecast/truth colorbar
        n_ticks_forecast = 7
        forecast_ticks = np.linspace(vmin, vmax, n_ticks_forecast)
        cbar1.set_ticks(forecast_ticks)
        forecast_tick_labels = [f'{tick:.2f}' for tick in forecast_ticks]
        cbar1.set_ticklabels(forecast_tick_labels)

        # Colorbar for relative difference (column 3)
        # This colorbar is symmetric around 0 with RdBu_r (blue=-, white=0, red=+)
        cax3 = fig.add_subplot(gs[n_rows, 2])
        cbar3 = plt.colorbar(last_im3, cax=cax3, orientation='horizontal',
                            extend='both')
        cbar3.set_label('Rel Diff (×10⁻²)', fontsize=9, labelpad=8)
        cbar3.ax.tick_params(labelsize=8, pad=4, length=4, width=1)

        # Set explicit colorbar limits to match the data limits
        cbar3.mappable.set_clim(vmin=rel_vmin_global, vmax=rel_vmax_global)

        # Format ticks to show symmetric values including 0
        # Create exactly 5 symmetric ticks from vmin to vmax
        n_ticks = 5  # Use 7 ticks for better granularity
        symmetric_ticks = np.linspace(rel_vmin_global, rel_vmax_global, n_ticks)
        cbar3.set_ticks(symmetric_ticks)

        # Format tick labels as percentages (multiply by 100)
        tick_labels = [f'{tick*100:.1f}' for tick in symmetric_ticks]
        cbar3.set_ticklabels(tick_labels)

        # Debug: print colorbar info
        print(f"  cbar3 limits: [{cbar3.vmin:.3f}, {cbar3.vmax:.3f}]")
        print(f"  cbar3 ticks: {symmetric_ticks}")

        # Main title
        fig.suptitle(f'{var_name} - Input Date: {input_date}',
                    fontsize=11, fontweight='bold', y=0.98)

        # print the colorbar limits used by reading from the colorbar cbar3 using get_clim()
        clim = cbar3.mappable.get_clim()
        print(f"  {var_name} relative difference colorbar limits used: [{clim[0]:.3f}, {clim[1]:.3f}]")
        

        # Save figure
        save_path = save_dir / f'{var_name.lower()}_{input_date.replace("-", "")}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")


def setup_map(ax, lon_range, lat_range):
    """Setup map features for a subplot"""
    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
                 crs=ccrs.PlateCarree())

    # Add features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle='--', alpha=0.5)

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray',
                     alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}


def get_units(variable):
    """Get units for variable"""
    units = {
        'thetao': '°C',
        'so': 'psu',
        'uo': 'm/s',
        'vo': 'm/s',
        'zos': 'm'
    }
    return units.get(variable, '')


def main():
    """Main plotting function"""
    parser = argparse.ArgumentParser(description='Create multipanel ocean forecast plots')
    parser.add_argument('--plot_config', type=str, required=True,
                       help='Path to plot configuration file')
    parser.add_argument('--model_config_only', action='store_true',
                       help='Only generate model config and exit')
    args = parser.parse_args()

    # Get project root directory (3 levels up from this file)
    project_root = Path(__file__).parent.parent.parent

    # Load plot configuration
    print("=== Loading Plot Configuration ===")
    plot_config = load_plot_config(args.plot_config)
    print(f"Model: {plot_config['model_name']}")
    print(f"Input Date: {plot_config['input_date']}")
    print(f"Reference Date: {plot_config['reference_date']}")
    print(f"Forecast Days: {plot_config['num_days']}")

    # Generate model config
    model_name = plot_config['model_name']
    model_config_file = generate_model_config(model_name, project_root)
    print(f"Model config generated: {model_config_file}")

    if args.model_config_only:
        print("Model config generation complete. Exiting.")
        return

    # Load model configuration
    print("\n=== Loading Model Configuration ===")
    pipe = ConfigPipeline([
        YamlConfig(f"./{model_config_file.name}", config_name='default',
                  config_folder=str(model_config_file.parent) + '/'),
        YamlConfig(config_folder='config/')
    ])
    config = pipe.read_conf()

    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Debug: Check config.results structure
    print(f"Debug - config.results type: {type(config.results)}")
    print(f"Debug - config.results.model_dir: {config.results.model_dir}")
    print(f"Debug - config.results.model_dir type: {type(config.results.model_dir)}")

    # Load model
    # Handle both string and boolean cases for model_dir
    if isinstance(config.results.model_dir, bool) or config.results.model_dir is None:
        # Fallback to default model directory
        model_dir = Path('results/models')
        print(f"Warning: config.results.model_dir is {config.results.model_dir}, using default: {model_dir}")
    else:
        model_dir = Path(config.results.model_dir)

    model_path = model_dir / f'{model_name}.pth'
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    model = load_model(config, str(model_path), device, model_name)

    # Load datasets
    print("\n=== Loading Datasets ===")
    data_dir = config.data.data_dir
    ocean_file = Path(data_dir) / f'{config.data.file_prefix}.nc'
    atm_file = Path(data_dir) / f'{config.data.file_prefix_atm}.nc'

    ocean_data = xr.open_dataset(ocean_file)
    atm_data = xr.open_dataset(atm_file)
    print(f"Ocean data: {ocean_file}")
    print(f"Atmospheric data: {atm_file}")

    # Create land-sea mask from ocean data
    print("\n=== Creating Land-Sea Mask ===")
    # Use first variable to create mask
    first_var = config.data.variable[0]
    data_for_mask = ocean_data[first_var].values
    dataset_length, mask = create_mask(data_for_mask)
    print(f"Mask shape: {mask.shape}")
    print(f"Ocean points: {np.sum(mask)}, Land points: {np.sum(mask == 0)}")

    # Convert input date to day index
    input_date = plot_config['input_date']
    reference_date = plot_config['reference_date']
    start_day_index = date_to_day_index(input_date, reference_date)
    print(f"\nInput day index: {start_day_index}")

    # Run autoregressive forecast
    num_days = plot_config['num_days']
    print(f"\n=== Running {num_days}-Day Forecast ===")
    forecast_dict, mean_dict = run_autoregressive_forecast(
        config, model, start_day_index, num_days, device,
        ocean_data, atm_data
    )

    # Postprocess forecasts
    print("Postprocessing forecasts...")
    postprocessed_forecasts = postprocess_forecasts(forecast_dict, mean_dict, config)

    # Load ground truth
    print("Loading ground truth...")
    ground_truth_dict = load_ground_truth_batch(config, start_day_index, num_days, ocean_data)

    # Create output directory
    # Handle both string and boolean cases for save_dir
    if isinstance(config.results.save_dir, bool) or config.results.save_dir is None:
        # Fallback to default save directory
        results_save_dir = Path('results')
        print(f"Warning: config.results.save_dir is {config.results.save_dir}, using default: {results_save_dir}")
    else:
        results_save_dir = Path(config.results.save_dir)

    save_dir = results_save_dir / model_name / 'plots'
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Creating Plots ===")
    print(f"Output directory: {save_dir}")

    # Create multipanel plots
    create_multipanel_plot(postprocessed_forecasts, ground_truth_dict, plot_config,
                          config, input_date, save_dir, ocean_data, mask)

    print("\n=== Plotting Complete ===")


if __name__ == "__main__":
    main()

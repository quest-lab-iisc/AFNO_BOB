"""
Inference script for autoregressive coupled atmospheric-ocean forecasting
=========================================================================

Key difference from run_inference.py:
  - Model predicts ALL 11 channels (6 atm + 5 ocean) at each step
  - No atmospheric data is loaded from disk during the autoregressive loop;
    the model's own predictions are used directly as the next input
  - Initial state loads BOTH atm(t) and ocean(t) from disk

Usage:
    python src/inference/run_inference_atmocean.py
    python src/inference/run_inference_atmocean.py --input_date 09-07-2020 --num_days 9
    python src/inference/run_inference_atmocean.py --model_path results/models/AFNO_AtmOcean_E00.pth
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

sys.path.append(str(Path(__file__).parent.parent))

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from models.architectures.afno.afnonet import AFNONet
from data_pipeline.preprocessing.transformers.normalize_transform import PreprocessTransform
from visualization.ocean_plotter import OceanPlotter
from inference.utils import compute_metrics, save_metrics_to_file, date_to_day_index


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(config, model_path, device):
    model = AFNONet(config)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from checkpoint (epoch {checkpoint.get('epoch', '?')}): {model_path}")
    else:
        model.load_state_dict(checkpoint)
        print(f"Model loaded from: {model_path}")
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Normalization statistics
# ---------------------------------------------------------------------------

def load_normalization_stats(config):
    mean_dir = config.data.mean_dir
    mean_dict = {}
    variance_dict = {}

    for var in ['thetao', 'so', 'uo', 'vo', 'zos']:
        f = f"{mean_dir}/mean_{var}_1993_2018_all_months.npy"
        mean_dict[var] = np.load(f) if Path(f).exists() else None

    for var in ['ssr', 'tp', 'u10', 'v10', 'msl', 'tcc']:
        f = f"{mean_dir}/{var}_mean_1993_2018_all_months.npy"
        mean_dict[var] = np.load(f) if Path(f).exists() else None

    variance_dict['ssr_std'] = np.sqrt(np.load(f'{mean_dir}/ssr_var_1993_2018_all_months.npy'))
    variance_dict['tp_std']  = np.sqrt(np.load(f'{mean_dir}/tp_var_1993_2018_all_months.npy'))
    variance_dict['msl_std'] = np.sqrt(np.load(f'{mean_dir}/msl_var_1993_2018_all_months.npy'))

    return mean_dict, variance_dict


# ---------------------------------------------------------------------------
# Preprocessing helpers (same logic as training dataset)
# ---------------------------------------------------------------------------

def preprocess_atm(data, var, mean_dict, variance_dict, transform, img_size=(224, 224)):
    """Normalize and interpolate one atmospheric variable; returns (1, H, W) tensor."""
    mean = mean_dict[var]
    if var == 'ssr' and mean is not None:
        proc = transform(data, mean, variable=var, type='atm', variance=variance_dict['ssr_std'])
    elif var == 'tp' and mean is not None:
        proc = transform(data, mean, variable=var, type='atm', variance=variance_dict['tp_std'])
    elif var == 'msl' and mean is not None:
        proc = transform(data, mean, variable=var, type='atm', variance=variance_dict['msl_std'])
    else:
        proc = transform(data, mean, variable=var, type='atm')
    return proc.squeeze(1)  # (1, H, W)


def preprocess_ocean(data, var, mean_dict, transform, img_size=(224, 224)):
    """Normalize and interpolate one ocean variable; returns (1, H, W) tensor."""
    proc = transform(data, mean_dict[var], variable=var)
    return proc.squeeze(1)  # (1, H, W)


# ---------------------------------------------------------------------------
# Postprocessing helpers
# ---------------------------------------------------------------------------

def postprocess_atm_variable(data, var, mean_dict, variance_dict):
    """
    Reverse normalization for atmospheric variable and interpolate to original resolution.

    Returns:
        numpy array at original atmospheric grid resolution
    """
    mean = mean_dict[var]
    data_2d = data.squeeze()                                    # (224, 224)

    # Determine target size: prefer var's own mean, fall back to any available
    # atm mean (all ERA5 atm vars share the same grid)
    if mean is not None:
        target_size = mean.squeeze().shape
    else:
        ref_mean = next(
            (mean_dict[v] for v in ['ssr', 'tp', 'tcc', 'u10', 'v10', 'msl']
             if mean_dict.get(v) is not None),
            None
        )
        target_size = ref_mean.squeeze().shape if ref_mean is not None else (224, 224)

    # Interpolate to original resolution
    t = torch.tensor(data_2d).unsqueeze(0).unsqueeze(0).float()  # (1,1,224,224)
    t = F.interpolate(t, size=target_size, mode='bilinear', align_corners=False)
    data_orig = t.squeeze(0).squeeze(0).numpy()                  # (H_orig, W_orig)

    # Reverse normalization
    if mean is not None:
        mean_2d = mean.squeeze()
        if var == 'ssr':
            data_orig = data_orig * variance_dict['ssr_std'].squeeze() + mean_2d
        elif var == 'tp':
            data_orig = data_orig * variance_dict['tp_std'].squeeze() + mean_2d
        elif var == 'msl':
            data_orig = data_orig * variance_dict['msl_std'].squeeze() + mean_2d
        else:
            data_orig = data_orig + mean_2d

    return data_orig


def postprocess_ocean_variable(data, var, mean_dict):
    """
    Reverse normalization for ocean variable and interpolate to original resolution.

    Returns:
        numpy array at original ocean grid resolution
    """
    from inference.utils import postprocess_ocean_variable as _post
    return _post(data, mean_dict[var], var)


# ---------------------------------------------------------------------------
# Initial state loading
# ---------------------------------------------------------------------------

def load_initial_state(config, day_index, ocean_data, atm_data,
                       mean_dict, variance_dict, transform):
    """
    Load and preprocess the initial 11-channel state at time t.

    Returns:
        input_tensor: (11, 224, 224) tensor = [atm(t) | ocean(t)]
    """
    channels = []

    # 6 atmospheric channels at t
    for var in config.data.atm_variable:
        raw = atm_data[var][day_index:day_index+1].values
        channels.append(preprocess_atm(raw, var, mean_dict, variance_dict, transform))

    # 5 ocean channels at t
    for var in config.data.variable:
        raw = ocean_data[var][day_index:day_index+1].values
        channels.append(preprocess_ocean(raw, var, mean_dict, transform))

    return torch.cat(channels, dim=0)  # (11, 224, 224)


# ---------------------------------------------------------------------------
# Autoregressive forecast
# ---------------------------------------------------------------------------

def load_land_sea_mask(config):
    """
    Load and return the land-sea mask interpolated to 224x224.

    Returns:
        mask: (1, 224, 224) float tensor — 1 for ocean, 0 for land
    """
    ocean_file = Path(config.data.data_dir) / f'{config.data.file_prefix}.nc'
    mask_data = xr.open_dataset(ocean_file)['thetao'].values  # (T, [1,] H, W)
    first = mask_data[0]                                       # ([1,] H, W)
    if first.ndim == 2:
        first = first[np.newaxis]                              # (1, H, W)
    mask = (~np.isnan(first)).astype(np.float32)               # (1, H, W)
    mask_t = torch.tensor(mask).unsqueeze(0)                   # (1, 1, H, W)
    mask_t = F.interpolate(mask_t, size=(224, 224), mode='bilinear', align_corners=False)
    return mask_t.squeeze(0)                                   # (1, 224, 224)


def run_autoregressive_forecast(config, model, day_index, num_days, device,
                                 ocean_data, atm_data):
    """
    Run fully autoregressive forecast.

    At each step the model's full 11-channel output is fed back as input;
    no atmospheric data is reloaded from disk.
    Ocean channel values under the land mask are zeroed before each feedback step.

    Returns:
        forecast_dict: {lead_time: {var: np.ndarray (224,224)}}
        mean_dict, variance_dict
    """
    mean_dict, variance_dict = load_normalization_stats(config)
    transform = PreprocessTransform(config)

    # Load land-sea mask for ocean channel zeroing
    ocean_mask = load_land_sea_mask(config)  # (1, 224, 224)
    n_atm = len(config.data.atm_variable)    # 6

    print(f"Loading initial state at day index {day_index}")
    x_t = load_initial_state(config, day_index,
                              ocean_data, atm_data,
                              mean_dict, variance_dict, transform)  # (11, 224, 224)

    # Variable order: 6 atm then 5 ocean
    all_vars = config.data.atm_variable + config.data.variable   # length 11

    forecast_dict = {}

    for lead_time in range(1, num_days + 1):
        print(f"  Forecasting lead time {lead_time}/{num_days} ...")

        with torch.no_grad():
            input_batch = x_t.unsqueeze(0).to(device)    # (1, 11, 224, 224)
            output = model(input_batch)
            x_t1 = output.squeeze(0).cpu()               # (11, 224, 224)

        # Fill ocean land points with per-channel nanmean of ocean pixels,
        # matching PreprocessTransform behaviour during training
        ocean_pixels = ocean_mask.squeeze(0) > 0.5        # (224, 224) bool
        for c in range(x_t1.shape[0] - n_atm):
            ch = x_t1[n_atm + c]                         # (224, 224)
            ch_mean = ch[ocean_pixels].mean()
            x_t1[n_atm + c] = torch.where(ocean_pixels, ch, ch_mean)

        # Store predictions per variable (in normalized/preprocessed space)
        step_preds = {}
        for i, var in enumerate(all_vars):
            step_preds[var] = x_t1[i].numpy()            # (224, 224)

        forecast_dict[lead_time] = step_preds

        # Prediction becomes next input (fully autoregressive)
        x_t = x_t1

    return forecast_dict, mean_dict, variance_dict


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------

def postprocess_forecasts(forecast_dict, mean_dict, variance_dict, config):
    """Reverse normalization for all forecast steps and all variables."""
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
# Ground truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(config, day_index, ocean_data, atm_data):
    """Load raw (original resolution) ground truth for all 11 variables at day_index."""
    gt = {}

    for var in config.data.atm_variable:
        gt[var] = atm_data[var][day_index:day_index+1].values.squeeze()

    for var in config.data.variable:
        gt[var] = ocean_data[var][day_index:day_index+1].values.squeeze()

    return gt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Autoregressive coupled atm-ocean forecast')
    parser.add_argument('--input_date', type=str, default=None)
    parser.add_argument('--num_days',   type=int, default=None)
    parser.add_argument('--name',       type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    args = parser.parse_args()

    # Load configuration
    print("=== Loading Configuration ===")
    pipe = ConfigPipeline([
        YamlConfig("./afno_atmocean_config.yaml", config_name='default', config_folder='config/'),
        YamlConfig(config_folder='config/')
    ])
    config = pipe.read_conf()

    if args.name:
        config.name = args.name

    input_date = args.input_date if args.input_date else config.plot.input_date
    num_days   = args.num_days   if args.num_days   else config.plot.num_days

    reference_date = "01-01-1993"
    day_index = date_to_day_index(input_date, reference_date)
    print(f"Input date: {input_date}  (day index: {day_index})")
    print(f"Forecast days: {num_days}")

    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model_path = args.model_path or f"{config.results.model_dir}/{config.name}.pth"
    if not os.path.exists(model_path):
        print(f"Error: model not found at {model_path}")
        sys.exit(1)
    model = load_model(config, model_path, device)

    # Open datasets
    data_dir  = config.data.data_dir
    ocean_data = xr.open_dataset(Path(data_dir) / f'{config.data.file_prefix}.nc')
    atm_data   = xr.open_dataset(Path(data_dir) / f'{config.data.file_prefix_atm}.nc')

    # Run forecast
    print(f"\n=== Running Autoregressive Forecast ({num_days} days) ===")
    forecast_dict, mean_dict, variance_dict = run_autoregressive_forecast(
        config, model, day_index, num_days, device, ocean_data, atm_data)

    # Postprocess
    print("\n=== Postprocessing Forecasts ===")
    postprocessed = postprocess_forecasts(forecast_dict, mean_dict, variance_dict, config)

    # Plotter
    plotter = OceanPlotter(config)

    use_default = True
    vmin_dict, vmax_dict = {}, {}
    if hasattr(config, 'visualization'):
        use_default = getattr(config.visualization, 'use_default', True)
        if not use_default:
            vmin_dict = config.visualization.vmin
            vmax_dict = config.visualization.vmax

    all_vars = config.data.atm_variable + config.data.variable

    for lead_time in range(1, num_days + 1):
        print(f"\n=== Lead Time +{lead_time} ===")

        pred_date = (datetime.strptime(input_date, "%d-%m-%Y")
                     + timedelta(days=lead_time)).strftime("%d-%m-%Y")
        gt = load_ground_truth(config, day_index + lead_time, ocean_data, atm_data)

        output_dir = (Path(config.results.save_dir) / config.name
                      / input_date / f"lead_time_{lead_time}")
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_dict = {}
        for var in all_vars:
            pred  = postprocessed[lead_time][var]
            truth = gt[var]

            metrics = compute_metrics(pred, truth)
            metrics_dict[var] = metrics
            print(f"  {var}: RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}"
                  f"  R²={metrics['r2']:.4f}  r={metrics['pearson']:.4f}")

            vmin = vmin_dict.get(var, None) if isinstance(vmin_dict, dict) else None
            vmax = vmax_dict.get(var, None) if isinstance(vmax_dict, dict) else None

            save_path = output_dir / f"{var}_comparison.png"
            plotter.plot_comparison(pred, truth, var, pred_date,
                                    str(save_path), vmin=vmin, vmax=vmax)

        save_metrics_to_file(metrics_dict, str(output_dir / "metrics.txt"))
        print(f"  Results saved to: {output_dir}")

    print("\n=== Autoregressive Forecast Complete ===")


if __name__ == "__main__":
    main()

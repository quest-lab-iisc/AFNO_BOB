"""Evaluate AFNO ocean model SST predictions against OSTIA L4 analysis.

Runs the same autoregressive 9-day forecasts as run_metrics.py over the 2020
test period, but computes metrics for sea-surface temperature (thetao) only,
validated against the OSTIA L4 SST analysis (an independent observational
product not used in training).

OSTIA provides an independent benchmark free of GLORYS reanalysis biases.
Predictions from the 224×224 model grid are first interpolated to the GLORYS
229×265 grid (denormalised), then the OSTIA field is bilinearly interpolated
to the same GLORYS grid for comparison.

Inputs:
    --name (str): Experiment name (overrides config).
    --model_path (str): Path to .pth model checkpoint.
    --config_file (str): YAML config filename in config/ (default: afno_bob_config.yaml).
    --ostia_file (str): Path to OSTIA netCDF (default: data/1993_2020/ostia_2020.nc).
    --output_dir (str): Output directory (default: results/{name}/).
    --device (str): Torch device (default: from config).

Outputs:
    {output_dir}/forecast_metrics_ostia.txt — per-lead RMSE, MAE, Pearson for SST.

Example:
    conda activate BoB_Surf_2
    python src/inference/run_metrics_ostia.py \\
        --name AFNO_BoB_Surf_E14 \\
        --config_file afno_bob_surf_e14.yaml \\
        --model_path results/models/AFNO_BoB_Surf_E14.pth
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from configmypy import ConfigPipeline, YamlConfig
from models.architectures.afno.afnonet import AFNONet
from data_pipeline.preprocessing.transformers.normalize_transform import PreprocessTransform
from inference.run_metrics import (
    load_normalization_stats,
    load_initial_ocean_state,
    load_atmospheric_forcing,
)
from inference.utils import date_to_day_index
from training.utils.experiment_logger import setup_logging, cleanup_logging


GLORYS_SHAPE = (229, 265)
OCEAN_VARS   = ['thetao', 'so', 'uo', 'vo', 'zos']
OSTIA_KELVIN_OFFSET = 273.15


def _load_model(config, model_path: Path, device: torch.device) -> torch.nn.Module:
    """Load AFNO model weights from a .pth checkpoint.

    Args:
        config: Configuration object.
        model_path (Path): Path to the weights file.
        device (torch.device): Target device.

    Returns:
        torch.nn.Module: Model in eval mode.

    Example:
        >>> model = _load_model(config, Path('results/models/AFNO_BoB_Surf_E14.pth'), device)
    """
    model = AFNONet(config)
    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


def _interp_to_glorys(arr_224: np.ndarray) -> np.ndarray:
    """Bilinearly interpolate a (224, 224) field to GLORYS grid (229, 265).

    Args:
        arr_224 (np.ndarray): Float32 array of shape (224, 224).

    Returns:
        np.ndarray: Float32 array of shape (229, 265).

    Example:
        >>> field_glorys = _interp_to_glorys(model_output)
    """
    t = torch.tensor(arr_224[np.newaxis, np.newaxis], dtype=torch.float32)
    return F.interpolate(t, size=GLORYS_SHAPE, mode='bilinear',
                         align_corners=False).squeeze().numpy()


def _precompute_ostia_on_glorys(ostia_file: Path,
                                 glorys_lat: np.ndarray,
                                 glorys_lon: np.ndarray) -> xr.DataArray:
    """Load OSTIA SST and pre-interpolate all daily fields to the GLORYS grid.

    Converts OSTIA from Kelvin to Celsius during loading.

    Args:
        ostia_file (Path): Path to ostia_2020.nc.
        glorys_lat (np.ndarray): GLORYS latitude coordinates, shape (229,).
        glorys_lon (np.ndarray): GLORYS longitude coordinates, shape (265,).

    Returns:
        xr.DataArray: OSTIA SST in °C on GLORYS grid, dims (time, latitude, longitude).

    Example:
        >>> ostia = _precompute_ostia_on_glorys(Path('ostia_2020.nc'), lat, lon)
        >>> ostia.shape
        (366, 229, 265)
    """
    ds = xr.open_dataset(ostia_file)
    sst_k = ds['analysed_sst']                # (366, 380, 440) in Kelvin
    sst_c = sst_k - OSTIA_KELVIN_OFFSET       # convert to °C
    sst_glorys = sst_c.interp(
        latitude=glorys_lat, longitude=glorys_lon, method='linear'
    )
    ds.close()
    return sst_glorys                          # (366, 229, 265)


def _compute_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Compute RMSE, MAE, and Pearson correlation over valid (non-NaN) pixels.

    Args:
        pred (np.ndarray): Predicted field, any shape.
        truth (np.ndarray): Ground truth field, same shape as pred.

    Returns:
        dict: Keys 'rmse', 'mae', 'pearson'. NaN if no valid pixels.

    Example:
        >>> m = _compute_metrics(sst_pred, sst_truth)
        >>> m['rmse']
        0.453
    """
    valid = ~np.isnan(pred) & ~np.isnan(truth)
    if valid.sum() < 2:
        return dict(rmse=np.nan, mae=np.nan, pearson=np.nan)
    p, t   = pred[valid], truth[valid]
    err    = p - t
    rmse   = float(np.sqrt(np.mean(err ** 2)))
    mae    = float(np.mean(np.abs(err)))
    pearson = float(np.corrcoef(p, t)[0, 1]) if (p.std() > 0 and t.std() > 0) else np.nan
    return dict(rmse=rmse, mae=mae, pearson=pearson)


def main():
    """Run AFNO SST evaluation against OSTIA over the 2020 test period.

    Args:
        None: All settings read from sys.argv (see module docstring).

    Returns:
        None: Writes forecast_metrics_ostia.txt as a side effect.

    Example:
        >>> # python src/inference/run_metrics_ostia.py --name AFNO_BoB_Surf_E14
    """
    parser = argparse.ArgumentParser(
        description='Evaluate AFNO SST forecasts against OSTIA L4 analysis'
    )
    parser.add_argument('--name',        default=None)
    parser.add_argument('--model_path',  default=None)
    parser.add_argument('--config_file', default='afno_bob_config.yaml')
    parser.add_argument('--ostia_file',  default='data/1993_2020/ostia_2020.nc')
    parser.add_argument('--output_dir',  default=None)
    parser.add_argument('--device',      default=None)
    args = parser.parse_args()

    # --- Config ---
    import argparse as _ap, sys as _sys
    _pre = _ap.ArgumentParser(add_help=False)
    _pre.add_argument('--config_file', default=None)
    _known, _ = _pre.parse_known_args(_sys.argv[1:])
    _exp_yaml  = _known.config_file or 'afno_bob_config.yaml'
    pipe = ConfigPipeline([
        YamlConfig('./afno_bob_config.yaml', config_name='default', config_folder='config/'),
        YamlConfig(f'./{_exp_yaml}',         config_name='default', config_folder='config/'),
    ])
    config = pipe.read_conf()
    if args.name:
        config.name = args.name

    tee_logger = setup_logging(config)

    device = torch.device(args.device or config.device)
    print(f"Device : {device}")

    # --- Model ---
    model_path = Path(args.model_path or f'results/models/{config.name}.pth')
    print(f"Model  : {model_path}")
    model = _load_model(config, model_path, device)

    # --- Output dir ---
    out_dir = Path(args.output_dir or f'results/{config.name}')
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # --- Normalisation stats ---
        mean_dict, variance_dict = load_normalization_stats(config)
        thetao_mean = np.squeeze(mean_dict['thetao'])    # (229, 265)
        transform   = PreprocessTransform(config)

        # --- Open GLORYS (for atm forcing path and lat/lon) ---
        data_dir  = Path(config.data.data_dir)
        ocean_ds  = xr.open_dataset(data_dir / f'{config.data.file_prefix}.nc')
        atm_ds    = xr.open_dataset(data_dir / f'{config.data.file_prefix_atm}.nc')
        glorys_lat = ocean_ds.latitude.values   # (229,)
        glorys_lon = ocean_ds.longitude.values  # (265,)
        land_mask  = np.isnan(ocean_ds['thetao'].values[0, 0])  # (229,265) True=land

        # --- Pre-interpolate OSTIA to GLORYS grid once ---
        ostia_file = Path(args.ostia_file)
        print(f"OSTIA  : {ostia_file}")
        print("Pre-interpolating OSTIA to GLORYS grid ...", flush=True)
        ostia_glorys = _precompute_ostia_on_glorys(ostia_file, glorys_lat, glorys_lon)
        # Build date → index map for OSTIA
        ostia_dates = [str(t)[:10] for t in ostia_glorys.time.values]
        ostia_date_idx = {d: i for i, d in enumerate(ostia_dates)}

        # --- Evaluation period (2020 test set from config) ---
        ref_date   = datetime.strptime(config.evaluation.reference_date, '%d-%m-%Y')
        start_date = datetime.strptime(config.evaluation.start_date, '%d-%m-%Y')
        num_days   = config.evaluation.num_days
        max_lead   = config.evaluation.max_forecast_days

        eval_days = num_days - max_lead
        print(f"\nEvaluation: {start_date.strftime('%Y-%m-%d')} + {eval_days} start dates, "
              f"{max_lead} lead days")

        # Accumulate metrics across start dates
        rmse_acc    = {lt: [] for lt in range(1, max_lead + 1)}
        mae_acc     = {lt: [] for lt in range(1, max_lead + 1)}
        pearson_acc = {lt: [] for lt in range(1, max_lead + 1)}

        # --- Main evaluation loop ---
        pbar = tqdm(range(eval_days), desc='OSTIA eval', unit='IC')

        for day_offset in pbar:
            ic_dt      = start_date + timedelta(days=day_offset)
            ic_idx     = date_to_day_index(ic_dt.strftime('%d-%m-%Y'),
                                           config.evaluation.reference_date)

            # --- Run 1-step-at-a-time AR forecast ---
            ocean_state = load_initial_ocean_state(
                config, ic_idx, ocean_ds, transform, mean_dict
            )

            for lead in range(1, max_lead + 1):
                atm = load_atmospheric_forcing(
                    config, ic_idx + lead, atm_ds, transform, mean_dict, variance_dict
                )
                ocean_tensor = torch.cat([ocean_state[v] for v in config.data.variable], dim=0)
                inp = torch.cat([atm, ocean_tensor], dim=0).unsqueeze(0).to(device)

                with torch.no_grad():
                    out = model(inp).squeeze(0).cpu()   # (5, 224, 224)

                # Update ocean state for next step
                for vi, var in enumerate(config.data.variable):
                    ocean_state[var] = out[vi:vi+1]

                # SST: denormalize and interpolate to GLORYS grid
                pred_norm  = out[0].numpy()            # (224, 224) thetao normalised
                pred_glorys = _interp_to_glorys(pred_norm)   # (229, 265)
                pred_sst    = pred_glorys + thetao_mean      # °C
                pred_sst[land_mask] = np.nan

                # OSTIA truth for the valid date
                valid_dt   = ic_dt + timedelta(days=lead)
                date_key   = valid_dt.strftime('%Y-%m-%d')
                if date_key not in ostia_date_idx:
                    continue
                truth_sst = ostia_glorys.values[ostia_date_idx[date_key]]  # (229,265) °C

                m = _compute_metrics(pred_sst, truth_sst)
                if not np.isnan(m['rmse']):
                    rmse_acc[lead].append(m['rmse'])
                    mae_acc[lead].append(m['mae'])
                    pearson_acc[lead].append(m['pearson'])

            pbar.set_postfix(IC=ic_dt.strftime('%Y-%m-%d'))

        ocean_ds.close()
        atm_ds.close()

        # --- Average and save ---
        out_path = out_dir / 'forecast_metrics_ostia.txt'
        with open(out_path, 'w') as f:
            f.write('=' * 70 + '\n')
            f.write('OSTIA SST EVALUATION METRICS\n')
            f.write(f'Model : {config.name}\n')
            f.write(f'Period: {start_date.strftime("%Y-%m-%d")} + {eval_days} IC dates\n')
            f.write(f'Truth : OSTIA L4 analysed_sst (converted to °C)\n')
            f.write('=' * 70 + '\n\n')
            f.write(f"{'Lead':>4}  {'N':>5}  {'RMSE':>8}  {'MAE':>8}  {'Pearson':>8}\n")
            f.write('-' * 42 + '\n')
            for lt in range(1, max_lead + 1):
                n = len(rmse_acc[lt])
                if n == 0:
                    f.write(f'{lt:>4}  {n:>5}  {"nan":>8}  {"nan":>8}  {"nan":>8}\n')
                else:
                    f.write(f'{lt:>4}  {n:>5}  '
                            f'{np.mean(rmse_acc[lt]):>8.4f}  '
                            f'{np.mean(mae_acc[lt]):>8.4f}  '
                            f'{np.nanmean(pearson_acc[lt]):>8.4f}\n')

        print(f'\nSaved: {out_path}')
        for lt in range(1, max_lead + 1):
            n = len(rmse_acc[lt])
            if n:
                print(f'  Lead +{lt}d  N={n:3d}  '
                      f'RMSE={np.mean(rmse_acc[lt]):.4f}  '
                      f'MAE={np.mean(mae_acc[lt]):.4f}  '
                      f'r={np.nanmean(pearson_acc[lt]):.4f}')

    finally:
        cleanup_logging(tee_logger)


if __name__ == '__main__':
    main()

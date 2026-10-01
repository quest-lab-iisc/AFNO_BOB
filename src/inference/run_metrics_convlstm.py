"""Evaluate a trained ConvLSTM model over the 2020-2025 test period.

Runs a stateful 9-day autoregressive rollout for each IC date.  The ConvLSTM
hidden state is initialised to zero at the start of each IC and carried forward
across all 9 lead days, giving the model full temporal memory over the forecast.

Reuses the postprocessing and metric utilities from ``run_metrics.py``.

Inputs:
    --name (str): Experiment name (default: ConvLSTM_BoB_Surf_E01).
    --model_path (str): Path to .pth weights (default: results/models/<name>.pth).
    --config_file (str): YAML config in config/ (default: convlstm_bob_config.yaml).
    --extra_data_dir (str): Directory with ocean/atm 2021-2025 NetCDF files.
    --extra_start_date (str): First date in extra data (dd-mm-yyyy).
    --extra_num_days (int): Days in extra evaluation window (default 1826).
    --extra_ref_date (str): Index-0 date for extra NetCDF (dd-mm-yyyy).
    --extra_ocean_file (str): Prefix for extra ocean NetCDF (default ocean_2021_2025).
    --extra_atm_file (str): Prefix for extra atm NetCDF (default atm_2021_2025).
    --combined_output (flag): Pool 2020 and 2021-2025 into a combined metrics file.
    --device (str): PyTorch device (default: cuda:0).

Outputs:
    results/<name>/forecast_metrics.txt           2020 metrics.
    results/<name>/forecast_metrics_extended.txt  2021-2025 metrics (if extra).
    results/<name>/forecast_metrics_combined.txt  Pooled metrics (if combined).
    logs/convlstm_metrics_<name>.log              Verbatim console log.

Example:
    conda activate BoB_Surf_2
    python src/inference/run_metrics_convlstm.py \\
        --name ConvLSTM_BoB_Surf_E01 \\
        --extra_data_dir data/2021_2025 \\
        --combined_output
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from configmypy import ConfigPipeline, YamlConfig
from data_pipeline.preprocessing.transformers.normalize_transform import PreprocessTransform
from inference.run_metrics import (
    load_normalization_stats, date_to_day_index,
    load_initial_ocean_state, load_atmospheric_forcing,
    postprocess_ocean_variable, load_ground_truth_batch,
    compute_forecast_metrics, average_raw_metrics, save_forecast_metrics,
)
from models.architectures.convlstm import ConvLSTMNet
from training.utils.experiment_logger import TeeLogger, cleanup_logging

N_NORTH_ROWS = 20


def load_convlstm(model_path: str, config, device: torch.device) -> ConvLSTMNet:
    """Load a trained ConvLSTMNet from a .pth weights file.

    Args:
        model_path (str): Path to the saved weights (.pth).
        config: Configuration namespace with ``convlstm`` sub-namespace.
        device (torch.device): Target device.

    Returns:
        ConvLSTMNet: Model in eval mode on ``device``.

    Example:
        >>> model = load_convlstm('results/models/ConvLSTM_BoB_Surf_E01.pth', cfg, dev)
    """
    c = config.convlstm
    model = ConvLSTMNet(
        in_channels=c.in_channels,
        hidden_channels=c.hidden_channels,
        num_layers=c.num_layers,
        out_channels=c.out_channels,
        kernel_size=c.kernel_size,
    ).to(device)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def run_convlstm_forecast(config, model: ConvLSTMNet,
                          start_day_index: int, forecast_days: int,
                          device: torch.device,
                          ocean_data, atm_data) -> tuple[dict, dict]:
    """Run a stateful ConvLSTM 9-day forecast from a single IC.

    The hidden state is initialised to zero at ``start_day_index`` and carried
    forward across all ``forecast_days`` autoregressive steps.

    Args:
        config: Configuration namespace.
        model (ConvLSTMNet): Trained model in eval mode.
        start_day_index (int): Index of the IC day in the dataset.
        forecast_days (int): Number of lead days (typically 9).
        device (torch.device): Compute device.
        ocean_data: xarray.Dataset with ocean variables.
        atm_data: xarray.Dataset with atmospheric variables.

    Returns:
        tuple:
            forecast_dict (dict): lead → {var: (224, 224) normalised prediction}.
            mean_dict (dict): Normalisation means for postprocessing.

    Example:
        >>> fc, md = run_convlstm_forecast(cfg, model, 0, 9, dev, oc, at)
        >>> fc[1]['thetao'].shape
        (224, 224)
    """
    mean_dict, variance_dict = load_normalization_stats(config)
    transform = PreprocessTransform(config)

    ocean_state = load_initial_ocean_state(
        config, start_day_index, ocean_data, transform, mean_dict)

    hidden = None   # initialised to zeros on first forward pass
    forecast_dict = {}

    for lead in range(1, forecast_days + 1):
        atm_forcing = load_atmospheric_forcing(
            config, start_day_index + lead,
            atm_data, transform, mean_dict, variance_dict)

        ocean_vars   = [ocean_state[v] for v in config.data.variable]
        ocean_tensor = torch.cat(ocean_vars, dim=0)           # (5, H, W)
        x = torch.cat([atm_forcing, ocean_tensor], dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred, hidden = model(x, hidden)

        pred_np = pred.squeeze(0).cpu().numpy()               # (5, H, W)
        step = {}
        for i, var in enumerate(config.data.out_variable):
            ch = pred_np[i]
            ch[-N_NORTH_ROWS:, :] = np.nan
            step[var] = ch
            ocean_state[var] = torch.tensor(pred_np[i:i+1])   # (1, H, W)

        forecast_dict[lead] = step

    return forecast_dict, mean_dict


def postprocess_convlstm(forecast_dict: dict, mean_dict: dict, config) -> dict:
    """Reverse normalisation and interpolate ConvLSTM forecasts to GLORYS grid.

    Args:
        forecast_dict (dict): lead → {var: (224, 224) normalised array}.
        mean_dict (dict): Normalisation means.
        config: Configuration namespace.

    Returns:
        dict: lead → {var: (229, 265) physical-units array, NaN on land}.

    Example:
        >>> pp = postprocess_convlstm(forecast_dict, mean_dict, config)
        >>> pp[1]['thetao'].shape
        (229, 265)
    """
    out = {}
    for lead, step in forecast_dict.items():
        out[lead] = {}
        for var in config.data.out_variable:
            out[lead][var] = postprocess_ocean_variable(step[var], mean_dict[var], var)
    return out


def run_convlstm_evaluation(config, model, device,
                             ocean_data, atm_data,
                             start_date: str, num_days: int,
                             reference_date: str,
                             desc: str = 'Evaluating') -> tuple[dict, dict, int, int]:
    """Evaluate ConvLSTM over a date range and return averaged + raw metrics.

    Args:
        config: Configuration namespace.
        model (ConvLSTMNet): Trained model in eval mode.
        device (torch.device): Compute device.
        ocean_data: xarray.Dataset (ocean).
        atm_data: xarray.Dataset (atmosphere).
        start_date (str): First IC date (dd-mm-yyyy).
        num_days (int): Evaluation window length.
        reference_date (str): Index-0 date in the datasets (dd-mm-yyyy).
        desc (str): tqdm progress bar label.

    Returns:
        tuple: (averaged_metrics, raw_metrics, start_day_index, num_eval_days).

    Example:
        >>> avg, raw, idx, n = run_convlstm_evaluation(
        ...     cfg, model, dev, oc, at, '01-01-2020', 365, '01-01-2020')
    """
    max_lead     = config.evaluation.max_forecast_days
    start_idx    = date_to_day_index(start_date, reference_date)
    dataset_size = len(ocean_data.time)
    num_eval     = min(num_days - max_lead + 1,
                       dataset_size - start_idx - max_lead)

    raw = {lt: {var: [] for var in config.data.out_variable}
           for lt in range(1, max_lead + 1)}

    for eval_day in tqdm(range(num_eval), desc=desc, file=sys.stdout,
                          dynamic_ncols=True):
        idx = start_idx + eval_day
        fc, md = run_convlstm_forecast(config, model, idx, max_lead, device,
                                       ocean_data, atm_data)
        pp      = postprocess_convlstm(fc, md, config)
        gt      = load_ground_truth_batch(config, idx, max_lead, ocean_data)
        metrics = compute_forecast_metrics(pp, gt, config)

        for lt in range(1, max_lead + 1):
            for var in config.data.out_variable:
                raw[lt][var].append(metrics[lt][var])

    averaged = average_raw_metrics(raw, config.data.out_variable, max_lead)
    return averaged, raw, start_idx, num_eval


def main() -> None:
    """Entry point: load model, evaluate, write metrics files.

    Example:
        python src/inference/run_metrics_convlstm.py \\
            --name ConvLSTM_BoB_Surf_E01 \\
            --extra_data_dir data/2021_2025 \\
            --combined_output
    """
    parser = argparse.ArgumentParser(
        description='ConvLSTM multi-day forecast evaluation')
    parser.add_argument('--name',             default='ConvLSTM_BoB_Surf_E01')
    parser.add_argument('--model_path',       default=None)
    parser.add_argument('--config_file',      default='convlstm_bob_config.yaml')
    parser.add_argument('--extra_data_dir',   default=None)
    parser.add_argument('--extra_start_date', default='01-01-2021')
    parser.add_argument('--extra_num_days',   type=int, default=1826)
    parser.add_argument('--extra_ref_date',   default='01-01-2021')
    parser.add_argument('--extra_ocean_file', default='ocean_2021_2025')
    parser.add_argument('--extra_atm_file',   default='atm_2021_2025')
    parser.add_argument('--combined_output',  action='store_true')
    parser.add_argument('--device',           default='cuda:0')
    args = parser.parse_args()

    pipe = ConfigPipeline([
        YamlConfig(f'./{args.config_file}', config_name='default',
                   config_folder='config/'),
        YamlConfig(config_folder='config/'),
    ])
    config = pipe.read_conf()
    if args.name:
        config.name = args.name

    Path('logs').mkdir(exist_ok=True)
    tee = TeeLogger(f'logs/convlstm_metrics_{config.name}.log')
    sys.stdout = tee

    try:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        model_path = (args.model_path
                      or str(Path(config.results.model_dir) / f'{config.name}.pth'))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model not found: {model_path}')
        print(f'Loading {model_path} …')
        model = load_convlstm(model_path, config, device)

        data_dir   = config.data.data_dir
        ocean_data = xr.open_dataset(Path(data_dir) / f'{config.data.file_prefix}.nc')
        atm_data   = xr.open_dataset(Path(data_dir) / f'{config.data.file_prefix_atm}.nc')

        out_dir = Path(config.results.save_dir) / config.name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f'\n=== Primary evaluation ({config.evaluation.start_date}) ===')
        avg, primary_raw, _, n1 = run_convlstm_evaluation(
            config, model, device, ocean_data, atm_data,
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

            print(f'\n=== Extended evaluation ({args.extra_start_date}) ===')
            ext_avg, ext_raw, _, n2 = run_convlstm_evaluation(
                config, model, device, extra_ocean, extra_atm,
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

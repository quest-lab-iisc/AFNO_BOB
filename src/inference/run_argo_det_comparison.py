"""Validate deterministic AFNO ocean forecasts against Argo float observations (2020–2025).

Runs autoregressive 9-day deterministic forecasts for IC dates spanning 2020–2025
(at a configurable stride), collocates the model predictions with Argo float
near-surface profiles, and computes bias, RMSE, MAE, and correlation as functions
of lead day, year-month, and season.

The model (default: E11p1) is loaded once and reused for all IC dates.
Atmospheric forcing is read from ERA5 daily fields: atm.nc for ≤2020-12-31 and
atm_2021_2025.nc for 2021-01-01 onwards.  Ocean ICs follow the same split using
ocean.nc and ocean_2021_2025.nc.

Argo profiles are discovered recursively under --argo_root by globbing
``**/*YYYYMMDD*_prof.nc``.  Profiles with the ``--download_argo`` flag are
fetched on demand from the NCEI GADR Indian Ocean archive.

Inputs:
    --model_path (str): Path to the model .pth weights file.
    --config (str): Config YAML filename in config/ directory.
    --ocean_file (str): Path to ocean.nc (1993–2020).
    --ocean_2021_file (str): Path to ocean_2021_2025.nc (2021–2025).
    --atm_file (str): Path to atm.nc (1993–2020).
    --atm_2021_file (str): Path to atm_2021_2025.nc (2021–2025).
    --mean_dir (str): Directory containing normalisation mean .npy files.
    --argo_root (str): Root directory for *YYYYMMDD_prof.nc Argo files.
    --date_start (str): First IC date in dd-mm-yyyy (default: 01-01-2020).
    --date_end (str): Last IC date in dd-mm-yyyy (default: 07-09-2025).
    --stride_days (int): Interval between consecutive IC dates (default: 7).
    --max_depth_dbar (float): ARGO near-surface depth cutoff in dbar (default: 20).
    --download_argo (flag): Download missing ARGO files from NCEI.
    --output_dir (str): Output directory (default: results/argo_det_comparison).
    --device (str): PyTorch device string (default: cuda:0).

Outputs:
    logs/argo_det_comparison.log          Verbatim console output with timestamps.
    {output_dir}/collocations.csv         All collocated pairs.
    {output_dir}/summary_by_lead.csv      RMSE/bias/MAE/n by lead day.
    {output_dir}/summary_by_yearmonth.csv RMSE/bias/MAE/n by year-month.
    {output_dir}/rmse_lead.png            RMSE and bias vs lead day.
    {output_dir}/scatter.png              Model vs ARGO scatter (all collocations).
    {output_dir}/temporal.png             Monthly RMSE time series 2020–2025.
    {output_dir}/map.png                  Map of all ARGO collocation positions.

Example:
    conda activate BoB_Surf_2
    python src/inference/run_argo_det_comparison.py \\
        --argo_root data/argo \\
        --date_start 01-01-2020 \\
        --date_end 31-12-2020 \\
        --output_dir results/argo_det_2020
"""

import argparse
import csv
import sys
import urllib.request
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from tqdm import tqdm

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))

from configmypy import ConfigPipeline, YamlConfig
from data_pipeline.preprocessing.transformers.normalize_transform import PreprocessTransform
from inference.run_argo_validation import load_argo_profiles, glorys_index
from inference.run_ensemble_verification import interp_to_glorys, denormalize
from models.architectures.afno.afnonet import AFNONet
from training.utils.experiment_logger import TeeLogger, cleanup_logging

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OCEAN_VARS   = ['thetao', 'so', 'uo', 'vo', 'zos']
ATM_VARS     = ['ssr', 'tp', 'u10', 'v10', 'msl', 'tcc']
ATM_ZSCORE   = ['ssr', 'tp', 'msl']
ATM_MEAN_SUB = ['u10', 'v10', 'tcc']
N_NORTH_ROWS = 20

GLORYS_LATS = np.linspace(4.0, 23.0, 229)
GLORYS_LONS = np.linspace(77.0, 99.0, 265)

REF_DATE_2020 = datetime(1993, 1, 1)
REF_DATE_2021 = datetime(2021, 1, 1)

NCEI_URL_TPL = (
    'https://www.ncei.noaa.gov/data/oceans/argo/gadr/data/indian/'
    '{year:04d}/{month:02d}/{date}_prof.nc'
)

_SEASON_MAP = {1: 'Winter', 2: 'Winter', 3: 'Winter',
               4: 'PreMonsoon', 5: 'PreMonsoon',
               6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon',
               10: 'PostMonsoon', 11: 'PostMonsoon', 12: 'PostMonsoon'}


# ---------------------------------------------------------------------------
# Norm stats
# ---------------------------------------------------------------------------

def load_norm_stats(mean_dir: Path) -> tuple[dict, dict]:
    """Load climatological means and stds used for model normalisation.

    Ocean variables (thetao, so) were normalised by mean subtraction.
    Atmospheric variables ssr, tp, msl were z-scored; u10, v10, tcc by mean
    subtraction only; no stats exist for uo, vo, zos.

    Args:
        mean_dir (Path): Directory containing mean_{var}_*.npy and {var}_mean_*.npy files.

    Returns:
        tuple[dict, dict]: (mean_dict, std_dict).  Keys are variable names; values are
            float32 arrays.  Missing files yield None.

    Example:
        >>> mean_dict, std_dict = load_norm_stats(Path('data/1993_2020/mean'))
        >>> mean_dict['thetao'].shape
        (1, 229, 265)
    """
    mean_dict = {}
    std_dict  = {}

    for var in OCEAN_VARS:
        f = mean_dir / f'mean_{var}_1993_2018_all_months.npy'
        mean_dict[var] = np.load(f).astype(np.float32) if f.exists() else None

    for var in ATM_VARS:
        f = mean_dir / f'{var}_mean_1993_2018_all_months.npy'
        mean_dict[var] = np.load(f).astype(np.float32) if f.exists() else None

    for var in ATM_ZSCORE:
        f = mean_dir / f'{var}_var_1993_2018_all_months.npy'
        std_dict[var] = np.sqrt(np.load(f)).astype(np.float32) if f.exists() else None

    return mean_dict, std_dict


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: Path, config, device: torch.device) -> torch.nn.Module:
    """Load an AFNO model from a .pth state-dict file.

    Args:
        model_path (Path): Path to a .pth file containing model_state_dict or raw
            state dict.
        config: configmypy config object with afno2d / data sections.
        device (torch.device): Target device.

    Returns:
        torch.nn.Module: Model in eval mode on device.

    Example:
        >>> model = load_model(Path('results/models/AFNO_BoB_Surf_E11p1.pth'), cfg, device)
    """
    model = AFNONet(config)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Atmospheric and ocean data access helpers
# ---------------------------------------------------------------------------

def _day_index(date: datetime) -> tuple[str, int]:
    """Return ('2020'|'2021', day_index) for a given date.

    Args:
        date (datetime): The calendar date.

    Returns:
        tuple[str, int]: Period string and 0-based day index into that file.

    Example:
        >>> _day_index(datetime(2021, 3, 15))
        ('2021', 73)
    """
    if date.year <= 2020:
        return '2020', (date - REF_DATE_2020).days
    return '2021', (date - REF_DATE_2021).days


def preprocess_ocean_ic(date: datetime,
                        ocean_ds_2020, ocean_ds_2021,
                        transform: PreprocessTransform,
                        mean_dict: dict) -> torch.Tensor:
    """Load and preprocess the 5-channel ocean IC for a given date.

    Applies mean subtraction (thetao, so), bilinear interpolation to 224×224,
    and zeros the top 20 rows (north boundary masking, matching training).

    Args:
        date (datetime): IC date.
        ocean_ds_2020: xarray.Dataset for ocean.nc (1993–2020).
        ocean_ds_2021: xarray.Dataset for ocean_2021_2025.nc (2021–2025).
        transform (PreprocessTransform): Preprocessing transform.
        mean_dict (dict): Normalisation means.

    Returns:
        torch.Tensor: Shape (5, 224, 224), float32, on CPU.

    Example:
        >>> ic = preprocess_ocean_ic(datetime(2020, 1, 14), ds20, ds21, tr, md)
        >>> ic.shape
        torch.Size([5, 224, 224])
    """
    period, idx = _day_index(date)
    ds = ocean_ds_2020 if period == '2020' else ocean_ds_2021

    channels = []
    for var in OCEAN_VARS:
        # isel drops time dim; thetao/so/uo/vo retain depth=1 → (1, 229, 265)
        # zos has no depth dim                                  → (229, 265)
        raw = ds[var].isel(time=idx).values
        raw = raw[np.newaxis]    # add batch: (1,1,229,265) or (1,229,265) for zos
        mean = mean_dict.get(var)
        ch = transform(raw, mean, variable=var)  # → (1, 1, 224, 224)
        ch = ch.squeeze(0)                        # (1, 224, 224)
        channels.append(ch)

    ocean_tensor = torch.cat(channels, dim=0)  # (5, 224, 224)
    ocean_tensor[:, -N_NORTH_ROWS:, :] = 0.0  # mask north boundary
    return ocean_tensor


def preprocess_atm_forcing(date: datetime,
                            atm_ds_2020, atm_ds_2021,
                            transform: PreprocessTransform,
                            mean_dict: dict, std_dict: dict) -> torch.Tensor | None:
    """Load and preprocess the 6-channel atmospheric forcing for a given date.

    Args:
        date (datetime): Date for which to load atmospheric forcing.
        atm_ds_2020: xarray.Dataset for atm.nc (1993–2020).
        atm_ds_2021: xarray.Dataset for atm_2021_2025.nc (2021–2025).
        transform (PreprocessTransform): Preprocessing transform.
        mean_dict (dict): Normalisation means.
        std_dict (dict): Normalisation stds (ssr, tp, msl only).

    Returns:
        torch.Tensor | None: Shape (6, 224, 224), or None if date is out of range.

    Example:
        >>> atm = preprocess_atm_forcing(datetime(2020, 1, 15), ds20, ds21, tr, md, sd)
        >>> atm.shape if atm is not None else None
        torch.Size([6, 224, 224])
    """
    period, idx = _day_index(date)
    ds = atm_ds_2020 if period == '2020' else atm_ds_2021

    if idx < 0 or idx >= ds.dims['time']:
        return None

    channels = []
    for var in ATM_VARS:
        # isel drops time dim; all atm vars retain depth=1 → (1, 77, 89)
        raw = ds[var].isel(time=idx).values
        raw = raw[np.newaxis]   # add batch → (1, 1, 77, 89)
        mean = mean_dict.get(var)
        std  = std_dict.get(var) if var in ATM_ZSCORE else None

        if var in ATM_ZSCORE:
            ch = transform(raw, mean, variable=var, type='atm', variance=std)
        else:
            ch = transform(raw, mean, variable=var, type='atm')
        ch = ch.squeeze(0)   # (1, 224, 224)
        channels.append(ch)

    return torch.cat(channels, dim=0)  # (6, 224, 224)


# ---------------------------------------------------------------------------
# Autoregressive deterministic forecast
# ---------------------------------------------------------------------------

_N_NORTH_ROWS = 20


def fill_north_rows_224(preds: np.ndarray) -> np.ndarray:
    """Extrapolate the zeroed north-boundary rows using the adjacent row value.

    Training zeroes the top 20 rows of every 224×224 prediction, causing a
    visual artifact when interpolated to GLORYS.  This function replaces those
    rows with the constant value of row 203 (the last unmasked row), which is
    the same strategy applied in run_ensemble_inference_ic.py before aggregation.

    Args:
        preds (np.ndarray): Array of any shape (..., 224, 224).

    Returns:
        np.ndarray: Same shape as input, with rows [-20:] filled from row 203.

    Example:
        >>> filled = fill_north_rows_224(preds_9x5x224x224)
        >>> np.any(filled[..., -20:, :] == 0.0)
        False
    """
    out = preds.copy()
    boundary = out[..., -(_N_NORTH_ROWS + 1): -_N_NORTH_ROWS, :]  # (..., 1, 224)
    out[..., -_N_NORTH_ROWS:, :] = boundary
    return out


def run_forecast(model: torch.nn.Module,
                 init_date: datetime,
                 ocean_ds_2020, ocean_ds_2021,
                 atm_ds_2020, atm_ds_2021,
                 transform: PreprocessTransform,
                 mean_dict: dict, std_dict: dict,
                 device: torch.device,
                 num_days: int = 9) -> np.ndarray | None:
    """Run a deterministic 9-day autoregressive forecast from init_date.

    Args:
        model (torch.nn.Module): Trained AFNO model in eval mode.
        init_date (datetime): Forecast initialisation date.
        ocean_ds_2020: xarray.Dataset for ocean.nc.
        ocean_ds_2021: xarray.Dataset for ocean_2021_2025.nc.
        atm_ds_2020: xarray.Dataset for atm.nc.
        atm_ds_2021: xarray.Dataset for atm_2021_2025.nc.
        transform (PreprocessTransform): Preprocessing transform.
        mean_dict (dict): Normalisation mean arrays.
        std_dict (dict): Normalisation std arrays.
        device (torch.device): Compute device.
        num_days (int): Forecast length in days (default: 9).

    Returns:
        np.ndarray | None: Shape (num_days, 5, 224, 224) in normalised space,
            or None if the IC or any forcing is out of available date range.

    Example:
        >>> preds = run_forecast(model, datetime(2020, 1, 14), ...)
        >>> preds.shape
        (9, 5, 224, 224)
    """
    try:
        ocean_state = preprocess_ocean_ic(
            init_date, ocean_ds_2020, ocean_ds_2021, transform, mean_dict
        )
    except (IndexError, KeyError, Exception):
        return None

    preds_list = []
    for step in range(1, num_days + 1):
        lead_date = init_date + timedelta(days=step)
        atm = preprocess_atm_forcing(
            lead_date, atm_ds_2020, atm_ds_2021, transform, mean_dict, std_dict
        )
        if atm is None:
            break

        x = torch.cat([atm, ocean_state], dim=0).unsqueeze(0).to(device)  # (1, 11, 224, 224)

        with torch.no_grad():
            pred = model(x).squeeze(0).cpu()   # (5, 224, 224)

        preds_list.append(pred.numpy())
        ocean_state = pred  # feed back in normalised space

    if len(preds_list) < num_days:
        return None

    return np.stack(preds_list, axis=0)   # (num_days, 5, 224, 224)


# ---------------------------------------------------------------------------
# ARGO index and downloading
# ---------------------------------------------------------------------------

def build_argo_index(argo_root: Path) -> dict[str, list[Path]]:
    """Recursively discover all Argo daily profile files and index them by date.

    Searches for files matching ``*YYYYMMDD*_prof.nc`` under argo_root.

    Args:
        argo_root (Path): Root directory to search.

    Returns:
        dict[str, list[Path]]: Maps 'YYYYMMDD' strings to lists of matching files.

    Example:
        >>> idx = build_argo_index(Path('data/argo'))
        >>> sorted(idx.keys())[:2]
        ['20200114', '20200115']
    """
    index: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(argo_root.rglob('*_prof.nc')):
        # Extract YYYYMMDD from stem (format: YYYYMMDD_prof)
        stem = f.stem
        date_part = stem[:8]
        if len(date_part) == 8 and date_part.isdigit():
            index[date_part].append(f)
    return dict(index)


def download_argo_day(date: datetime, argo_root: Path) -> list[Path]:
    """Download the ARGO profile file for one day from NCEI and save to argo_root.

    Args:
        date (datetime): Date for which to download the Argo file.
        argo_root (Path): Directory where the downloaded file will be saved.

    Returns:
        list[Path]: List containing the saved file Path if successful, else [].

    Example:
        >>> paths = download_argo_day(datetime(2020, 1, 14), Path('data/argo'))
    """
    date_str = date.strftime('%Y%m%d')
    url = NCEI_URL_TPL.format(year=date.year, month=date.month, date=date_str)
    dest = argo_root / f'{date_str}_prof.nc'
    if dest.exists():
        return [dest]
    try:
        urllib.request.urlretrieve(url, dest)
        return [dest]
    except Exception as exc:
        tqdm.write(f'  [ARGO download] {date_str}: {exc}')
        return []


# ---------------------------------------------------------------------------
# Collocation
# ---------------------------------------------------------------------------

def collocate_ic(forecast: np.ndarray,
                 init_date: datetime,
                 argo_index: dict[str, list[Path]],
                 mean_dict: dict,
                 max_depth_dbar: float,
                 download_argo: bool,
                 argo_root: Path) -> list[dict]:
    """Collocate a 9-day forecast against all available Argo profiles in the window.

    Args:
        forecast (np.ndarray): Shape (9, 5, 224, 224) in normalised space.
        init_date (datetime): Forecast initialisation date (not included in window).
        argo_index (dict[str, list[Path]]): Maps 'YYYYMMDD' to Argo file paths.
        mean_dict (dict): Normalisation means for denormalisation.
        max_depth_dbar (float): Shallowest acceptable ARGO pressure (dbar).
        download_argo (bool): Attempt NCEI download for missing dates.
        argo_root (Path): ARGO root directory (used for downloads).

    Returns:
        list[dict]: One record per valid collocation with keys ic_date, date,
            lead_day, lat, lon, pres, argo_temp, argo_psal, model_thetao,
            model_so, year, season.

    Example:
        >>> pairs = collocate_ic(forecast, datetime(2020, 1, 14), idx, md, 20, False, path)
        >>> pairs[0]['lead_day']
        1
    """
    # Denormalise thetao and so for all 9 lead days → (9, 229, 265)
    filled = fill_north_rows_224(forecast)

    pred_thetao = np.stack([
        denormalize(interp_to_glorys(filled[ld, 0]), 'thetao', mean_dict)
        for ld in range(9)
    ])  # (9, 229, 265)
    pred_so = np.stack([
        denormalize(interp_to_glorys(filled[ld, 1]), 'so', mean_dict)
        for ld in range(9)
    ])  # (9, 229, 265)

    ic_str = init_date.strftime('%Y%m%d')
    pairs = []

    for lead in range(1, 10):
        obs_date = init_date + timedelta(days=lead)
        obs_str  = obs_date.strftime('%Y%m%d')
        step     = lead - 1   # 0-indexed

        # Ensure Argo files are available for this day
        if obs_str not in argo_index:
            if download_argo:
                new_files = download_argo_day(obs_date, argo_root)
                if new_files:
                    argo_index[obs_str] = new_files
            if obs_str not in argo_index:
                continue

        profiles = []
        for fp in argo_index[obs_str]:
            profiles.extend(load_argo_profiles(fp, max_depth_dbar=max_depth_dbar))

        for rec in profiles:
            li, lj = glorys_index(rec['lat'], rec['lon'])
            mt = float(pred_thetao[step, li, lj])
            ms = float(pred_so[step, li, lj])
            if np.isnan(mt) or np.isnan(ms):
                continue
            obs_dt = datetime.strptime(rec['date'], '%Y%m%d')
            pairs.append({
                'ic_date'      : ic_str,
                'date'         : rec['date'],
                'lead_day'     : lead,
                'lat'          : rec['lat'],
                'lon'          : rec['lon'],
                'pres'         : rec['pres'],
                'argo_temp'    : rec['temp'],
                'argo_psal'    : rec['psal'],
                'model_thetao' : mt,
                'model_so'     : ms,
                'year'         : obs_dt.year,
                'season'       : _SEASON_MAP[obs_dt.month],
            })

    return pairs


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _error_stats(errors: list[float]) -> dict:
    """Compute bias, RMSE, MAE, and n from a list of signed errors.

    Args:
        errors (list[float]): List of (model − obs) values.

    Returns:
        dict: Keys bias, rmse, mae, n.

    Example:
        >>> _error_stats([0.1, -0.2, 0.3])
        {'bias': 0.067, 'rmse': 0.216, 'mae': 0.2, 'n': 3}
    """
    e = np.array(errors, dtype=np.float64)
    return {'bias': float(np.mean(e)),
            'rmse': float(np.sqrt(np.mean(e ** 2))),
            'mae' : float(np.mean(np.abs(e))),
            'n'   : int(len(e))}


def summarise_by_lead(pairs: list[dict]) -> dict:
    """Aggregate collocation errors by lead day for thetao and so.

    Args:
        pairs (list[dict]): Output of collocate_ic aggregated over all IC dates.

    Returns:
        dict: Nested dict keyed by variable ('thetao', 'so') then lead day (1–9).
            Each leaf is a stats dict (bias, rmse, mae, n).

    Example:
        >>> stats = summarise_by_lead(all_pairs)
        >>> stats['thetao'][3]['rmse']
        0.45
    """
    result = {'thetao': {}, 'so': {}}
    by_lead_t: dict[int, list] = defaultdict(list)
    by_lead_s: dict[int, list] = defaultdict(list)

    for p in pairs:
        ld = p['lead_day']
        by_lead_t[ld].append(p['model_thetao'] - p['argo_temp'])
        by_lead_s[ld].append(p['model_so']     - p['argo_psal'])

    for ld in sorted(by_lead_t):
        result['thetao'][ld] = _error_stats(by_lead_t[ld])
        result['so'][ld]     = _error_stats(by_lead_s[ld])

    return result


def summarise_by_yearmonth(pairs: list[dict]) -> dict:
    """Aggregate collocation errors by (year, month) for thetao and so.

    Args:
        pairs (list[dict]): All collocated pairs.

    Returns:
        dict: Keyed (year, month) → {'thetao': stats_dict, 'so': stats_dict}.

    Example:
        >>> stats = summarise_by_yearmonth(all_pairs)
        >>> stats[(2020, 1)]['thetao']['rmse']
        0.38
    """
    by_ym: dict = defaultdict(lambda: {'thetao': [], 'so': []})

    for p in pairs:
        obs_dt = datetime.strptime(p['date'], '%Y%m%d')
        key = (obs_dt.year, obs_dt.month)
        by_ym[key]['thetao'].append(p['model_thetao'] - p['argo_temp'])
        by_ym[key]['so'].append(p['model_so']     - p['argo_psal'])

    result = {}
    for key in sorted(by_ym):
        result[key] = {
            'thetao': _error_stats(by_ym[key]['thetao']),
            'so'    : _error_stats(by_ym[key]['so']),
        }
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rmse_lead(lead_stats: dict, output_path: Path) -> None:
    """Plot RMSE and bias vs lead day for thetao and so.

    Args:
        lead_stats (dict): Output of summarise_by_lead().
        output_path (Path): Where to save the PNG.

    Returns:
        None: Writes PNG to output_path.

    Example:
        >>> plot_rmse_lead(stats, Path('results/rmse_lead.png'))
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    var_cfg = [
        ('thetao', 'SST (°C)',   '#1565C0'),
        ('so',     'SSS (PSU)',  '#E65100'),
    ]

    for col, (var, unit, color) in enumerate(var_cfg):
        vstats = lead_stats[var]
        leads  = sorted(vstats)
        rmse   = [vstats[ld]['rmse'] for ld in leads]
        bias   = [vstats[ld]['bias'] for ld in leads]
        ns     = [vstats[ld]['n']    for ld in leads]

        for row, (metric, values, ylabel) in enumerate([
            ('RMSE', rmse, f'RMSE ({unit})'),
            ('Bias', bias, f'Bias model−Argo ({unit})'),
        ]):
            ax = axes[row, col]
            ax.plot(leads, values, marker='o', linewidth=2, markersize=7,
                    color=color, label=var)
            for x, y, n in zip(leads, values, ns):
                ax.annotate(f'n={n}', (x, y), textcoords='offset points',
                            xytext=(4, 4), fontsize=6, color='#444444')
            if row == 1:
                ax.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
            ax.set_xlabel('Lead day')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{var}  —  {metric}', fontsize=10)
            ax.set_xticks(range(1, 10))
            ax.set_xlim(0.5, 9.5)
            ax.grid(True, alpha=0.3)

    fig.suptitle('Deterministic AFNO vs Argo: RMSE and Bias by lead day',
                 fontsize=12, fontweight='bold')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    tqdm.write(f'  Saved: {output_path}')


def plot_rmse_lead_comparison(stats_by_label: dict[str, dict],
                              output_path: Path) -> None:
    """Overlay RMSE and bias vs lead day for multiple subsets (e.g. Jan vs Oct).

    Args:
        stats_by_label (dict[str, dict]): Maps a display label (e.g. 'Jan 2020')
            to a lead_stats dict as returned by summarise_by_lead().
        output_path (Path): Where to save the PNG.

    Returns:
        None: Writes PNG to output_path.

    Example:
        >>> plot_rmse_lead_comparison({'Jan': jan_stats, 'Oct': oct_stats},
        ...                           Path('results/rmse_lead_comparison.png'))
    """
    palette = ['#1565C0', '#E65100', '#2E7D32', '#6A1B9A']
    markers = ['o', 's', '^', 'D']
    var_cfg = [
        ('thetao', 'SST (°C)'),
        ('so',     'SSS (PSU)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for col, (var, unit) in enumerate(var_cfg):
        for row, (metric_key, ylabel) in enumerate([
            ('rmse', f'RMSE ({unit})'),
            ('bias', f'Bias model−Argo ({unit})'),
        ]):
            ax = axes[row, col]
            for i, (label, lead_stats) in enumerate(stats_by_label.items()):
                vstats = lead_stats[var]
                leads  = sorted(vstats)
                values = [vstats[ld][metric_key] for ld in leads]
                ns     = [vstats[ld]['n']         for ld in leads]
                color  = palette[i % len(palette)]
                marker = markers[i % len(markers)]
                ax.plot(leads, values, marker=marker, linewidth=2, markersize=7,
                        color=color, label=label)
                for x, y, n in zip(leads, values, ns):
                    ax.annotate(f'n={n}', (x, y), textcoords='offset points',
                                xytext=(4, 4), fontsize=6, color=color)
            if row == 1:
                ax.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
            ax.set_xlabel('Lead day')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{var}  —  {metric_key.upper()}', fontsize=10)
            ax.set_xticks(range(1, 10))
            ax.set_xlim(0.5, 9.5)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

    fig.suptitle('Deterministic AFNO vs Argo: RMSE and Bias by lead day',
                 fontsize=12, fontweight='bold')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    tqdm.write(f'  Saved: {output_path}')


def plot_scatter(pairs: list[dict], output_path: Path) -> None:
    """Model vs Argo scatter for thetao and so, coloured by lead day.

    Args:
        pairs (list[dict]): All collocated pairs.
        output_path (Path): Where to save the PNG.

    Returns:
        None: Writes PNG to output_path.

    Example:
        >>> plot_scatter(all_pairs, Path('results/scatter.png'))
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    cfg = [
        ('thetao', 'argo_temp', 'model_thetao',
         'Argo TEMP (°C)', 'Model SST (°C)', (22, 32)),
        ('so', 'argo_psal', 'model_so',
         'Argo PSAL (PSU)', 'Model SSS (PSU)', (27, 37)),
    ]
    for ax, (var, obs_key, mod_key, xlabel, ylabel, xlim) in zip(axes, cfg):
        obs   = np.array([p[obs_key]   for p in pairs])
        model = np.array([p[mod_key]   for p in pairs])
        leads = np.array([p['lead_day'] for p in pairs])
        sc = ax.scatter(obs, model, c=leads, cmap='viridis', s=18,
                        edgecolors='none', alpha=0.6, vmin=1, vmax=9)
        plt.colorbar(sc, ax=ax, label='Lead day', fraction=0.046, pad=0.04)
        lo, hi = xlim
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
        bias = float(np.mean(model - obs))
        rmse = float(np.sqrt(np.mean((model - obs) ** 2)))
        r    = float(np.corrcoef(obs, model)[0, 1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f'{var}  (n={len(obs)})\n'
            f'Bias={bias:+.2f}  RMSE={rmse:.2f}  r={r:.3f}',
            fontsize=10
        )
    fig.suptitle('Deterministic AFNO vs Argo — all collocations',
                 fontsize=12, fontweight='bold')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    tqdm.write(f'  Saved: {output_path}')


def plot_temporal(ym_stats: dict, output_path: Path) -> None:
    """Monthly RMSE time series for thetao and so over the evaluation period.

    Args:
        ym_stats (dict): Output of summarise_by_yearmonth().
        output_path (Path): Where to save the PNG.

    Returns:
        None: Writes PNG to output_path.

    Example:
        >>> plot_temporal(ym_stats, Path('results/temporal.png'))
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), constrained_layout=True,
                             sharex=True)
    cfg = [
        ('thetao', 'RMSE SST (°C)',   '#1565C0'),
        ('so',     'RMSE SSS (PSU)',  '#E65100'),
    ]
    xdates = [datetime(y, m, 15) for y, m in sorted(ym_stats)]

    for ax, (var, ylabel, color) in zip(axes, cfg):
        rmse = [ym_stats[k][var]['rmse'] for k in sorted(ym_stats)]
        n    = [ym_stats[k][var]['n']    for k in sorted(ym_stats)]
        ax.plot(xdates, rmse, color=color, linewidth=1.5, marker='o',
                markersize=4, label='RMSE')
        ax2 = ax.twinx()
        ax2.bar(xdates, n, width=25, color=color, alpha=0.15, label='n obs')
        ax2.set_ylabel('n collocations', color=color, alpha=0.6, fontsize=9)
        ax2.tick_params(labelcolor=color, labelsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

    axes[0].set_title('Monthly RMSE — Deterministic AFNO vs Argo',
                      fontsize=11, fontweight='bold')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    tqdm.write(f'  Saved: {output_path}')


def plot_map(pairs: list[dict], output_path: Path) -> None:
    """Scatter map of all ARGO collocation positions, coloured by lead day.

    Uses a cartopy PlateCarree projection to overlay land mask and coastlines.

    Args:
        pairs (list[dict]): All collocated pairs.
        output_path (Path): Where to save the PNG.

    Returns:
        None: Writes PNG to output_path.

    Example:
        >>> plot_map(all_pairs, Path('results/map.png'))
    """
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw={'projection': proj})

    ax.set_extent([77, 99, 4, 23], crs=proj)
    ax.add_feature(cfeature.LAND, facecolor='#d3d3d3', edgecolor='#555555',
                   linewidth=0.4, zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor='#888888',
                   linestyle='--', zorder=4)
    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=0.4,
                      color='grey', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    lats  = [p['lat']      for p in pairs]
    lons  = [p['lon']      for p in pairs]
    leads = [p['lead_day'] for p in pairs]
    sc = ax.scatter(lons, lats, c=leads, cmap='viridis', s=16, transform=proj,
                    edgecolors='k', linewidths=0.3, alpha=0.8, vmin=1, vmax=9,
                    zorder=3)
    plt.colorbar(sc, ax=ax, label='Lead day', fraction=0.04, pad=0.04)

    ax.set_title(f'Argo collocation positions (n={len(pairs):,})', fontsize=11)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    tqdm.write(f'  Saved: {output_path}')


# ---------------------------------------------------------------------------
# CSV output helpers
# ---------------------------------------------------------------------------

def save_collocations(pairs: list[dict], csv_path: Path) -> None:
    """Write all collocated pairs to a CSV file.

    Args:
        pairs (list[dict]): List of collocation records.
        csv_path (Path): Output path.

    Returns:
        None: Writes CSV to csv_path.

    Example:
        >>> save_collocations(all_pairs, Path('results/collocations.csv'))
    """
    if not pairs:
        return
    fieldnames = list(pairs[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in pairs:
            row = {k: round(v, 4) if isinstance(v, float) else v
                   for k, v in p.items()}
            writer.writerow(row)


def save_lead_summary(lead_stats: dict, csv_path: Path) -> None:
    """Write lead-day summary statistics to CSV.

    Args:
        lead_stats (dict): Output of summarise_by_lead().
        csv_path (Path): Output path.

    Returns:
        None: Writes CSV to csv_path.

    Example:
        >>> save_lead_summary(lead_stats, Path('results/summary_by_lead.csv'))
    """
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['variable', 'lead_day', 'n', 'bias', 'rmse', 'mae'])
        for var in ['thetao', 'so']:
            for ld, s in sorted(lead_stats[var].items()):
                writer.writerow([var, ld, s['n'],
                                  round(s['bias'], 4),
                                  round(s['rmse'], 4),
                                  round(s['mae'],  4)])


def save_ym_summary(ym_stats: dict, csv_path: Path) -> None:
    """Write year-month summary statistics to CSV.

    Args:
        ym_stats (dict): Output of summarise_by_yearmonth().
        csv_path (Path): Output path.

    Returns:
        None: Writes CSV to csv_path.

    Example:
        >>> save_ym_summary(ym_stats, Path('results/summary_by_yearmonth.csv'))
    """
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['year', 'month', 'variable', 'n', 'bias', 'rmse', 'mae'])
        for (yr, mo), stats in sorted(ym_stats.items()):
            for var in ['thetao', 'so']:
                s = stats[var]
                writer.writerow([yr, mo, var, s['n'],
                                  round(s['bias'], 4),
                                  round(s['rmse'], 4),
                                  round(s['mae'],  4)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Parse CLI arguments and run the full deterministic Argo validation pipeline.

    Iterates over IC dates at --stride_days intervals, runs 9-day deterministic
    forecasts, collocates predictions with available Argo profiles, saves the
    combined CSV and four diagnostic figures.

    Args:
        None: All parameters read from sys.argv via argparse.

    Returns:
        None: Side effects are CSV files and PNG figures in --output_dir.

    Example:
        >>> # python src/inference/run_argo_det_comparison.py \\
        >>> #     --argo_root data/argo --date_start 01-01-2020 --date_end 31-12-2020
    """
    parser = argparse.ArgumentParser(
        description='Deterministic AFNO forecast vs Argo 2020–2025'
    )
    parser.add_argument('--model_path', default='results/models/AFNO_BoB_Surf_E11p1.pth')
    parser.add_argument('--config',     default='afno_bob_surf_e11p1.yaml',
                        help='Config YAML filename in config/')
    parser.add_argument('--ocean_file', default='data/1993_2020/ocean.nc')
    parser.add_argument('--ocean_2021_file',
                        default='data/2021_2025/ocean_2021_2025.nc')
    parser.add_argument('--atm_file',   default='data/1993_2020/atm.nc')
    parser.add_argument('--atm_2021_file',
                        default='data/2021_2025/atm_2021_2025.nc')
    parser.add_argument('--mean_dir',   default='data/1993_2020/mean')
    parser.add_argument('--argo_root',  required=True,
                        help='Root directory tree containing *YYYYMMDD_prof.nc files')
    parser.add_argument('--date_start', default='01-01-2020')
    parser.add_argument('--date_end',   default='07-09-2025')
    parser.add_argument('--stride_days', type=int, default=7)
    parser.add_argument('--max_depth_dbar', type=float, default=20.0)
    parser.add_argument('--download_argo', action='store_true',
                        help='Download missing ARGO files from NCEI')
    parser.add_argument('--output_dir', default='results/argo_det_comparison')
    parser.add_argument('--device',     default='cuda:0')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Logging ---
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / 'argo_det_comparison.log'
    tee = TeeLogger(str(log_path))
    sys.stdout = tee

    try:
        _run(args, output_dir)
    except Exception as exc:
        import traceback
        print(f'\n[FATAL] {exc}')
        traceback.print_exc()
    finally:
        cleanup_logging(tee)


def _run(args, output_dir: Path) -> None:
    """Core pipeline executed inside the logging wrapper.

    Args:
        args: Parsed argparse namespace.
        output_dir (Path): Where outputs are written.

    Returns:
        None

    Example:
        >>> _run(args, Path('results/argo_det_comparison'))
    """
    device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu'
    )
    print(f'Device: {device}')

    # --- Config ---
    pipe = ConfigPipeline([
        YamlConfig(args.config, config_name='default', config_folder='config/'),
        YamlConfig(config_folder='config/'),
    ])
    config = pipe.read_conf()

    # --- Model ---
    print(f'Loading model from {args.model_path} …')
    model = load_model(Path(args.model_path), config, device)

    # --- Norm stats ---
    print('Loading normalisation stats …')
    mean_dict, std_dict = load_norm_stats(Path(args.mean_dir))

    # --- Transform ---
    transform = PreprocessTransform(config)

    # --- Open datasets ---
    print('Opening ocean/atm datasets …')
    ocean_ds_2020 = xr.open_dataset(args.ocean_file,       mask_and_scale=True)
    ocean_ds_2021 = xr.open_dataset(args.ocean_2021_file,  mask_and_scale=True)
    atm_ds_2020   = xr.open_dataset(args.atm_file,         mask_and_scale=True)
    atm_ds_2021   = xr.open_dataset(args.atm_2021_file,    mask_and_scale=True)

    # --- ARGO index ---
    argo_root = Path(args.argo_root)
    print(f'Indexing Argo files under {argo_root} …')
    argo_index = build_argo_index(argo_root)
    print(f'  {len(argo_index)} unique dates found ({sum(len(v) for v in argo_index.values())} files)')

    # --- IC date range ---
    date_start = datetime.strptime(args.date_start, '%d-%m-%Y')
    date_end   = datetime.strptime(args.date_end,   '%d-%m-%Y')

    ic_dates = []
    d = date_start
    while d <= date_end:
        ic_dates.append(d)
        d += timedelta(days=args.stride_days)
    print(f'Evaluating {len(ic_dates)} IC dates '
          f'({date_start.date()} → {date_end.date()}, stride={args.stride_days}d)')

    # --- Forecast + collocation loop ---
    all_pairs: list[dict] = []
    n_skipped_no_argo  = 0
    n_skipped_no_data  = 0
    n_forecast_ok      = 0

    for init_date in tqdm(ic_dates, desc='IC dates', unit='IC',
                           file=sys.stdout, dynamic_ncols=True):
        # Quick pre-check: is there any ARGO data in this IC's 9-day window?
        window_dates = [
            (init_date + timedelta(days=ld)).strftime('%Y%m%d')
            for ld in range(1, 10)
        ]
        has_argo = any(d in argo_index for d in window_dates)
        if not has_argo and not args.download_argo:
            n_skipped_no_argo += 1
            continue

        # Run forecast
        forecast = run_forecast(
            model, init_date,
            ocean_ds_2020, ocean_ds_2021,
            atm_ds_2020,   atm_ds_2021,
            transform, mean_dict, std_dict, device
        )
        if forecast is None:
            n_skipped_no_data += 1
            continue

        n_forecast_ok += 1

        # Collocate
        pairs = collocate_ic(
            forecast, init_date, argo_index, mean_dict,
            args.max_depth_dbar, args.download_argo, argo_root
        )
        all_pairs.extend(pairs)

        if pairs:
            tqdm.write(
                f'  {init_date.strftime("%d %b %Y")}  '
                f'{len(pairs):3d} collocations'
            )

    ocean_ds_2020.close()
    ocean_ds_2021.close()
    atm_ds_2020.close()
    atm_ds_2021.close()

    # --- Summary ---
    print(f'\n=== Run summary ===')
    print(f'IC dates evaluated   : {n_forecast_ok}')
    print(f'Skipped (no Argo)    : {n_skipped_no_argo}')
    print(f'Skipped (no IC data) : {n_skipped_no_data}')
    print(f'Total collocations   : {len(all_pairs)}')

    if not all_pairs:
        print('No collocations found — check --argo_root and date range.')
        return

    # --- Save collocation CSV ---
    coll_csv = output_dir / 'collocations.csv'
    save_collocations(all_pairs, coll_csv)
    print(f'Saved: {coll_csv}')

    # --- Statistics ---
    lead_stats = summarise_by_lead(all_pairs)
    ym_stats   = summarise_by_yearmonth(all_pairs)

    save_lead_summary(lead_stats, output_dir / 'summary_by_lead.csv')
    save_ym_summary(ym_stats,     output_dir / 'summary_by_yearmonth.csv')

    # --- Print lead table ---
    print(f'\n{"Lead":>5}  {"n_T":>6}  {"T_RMSE":>8}  {"T_bias":>8}'
          f'  {"n_S":>6}  {"S_RMSE":>8}  {"S_bias":>8}')
    for ld in sorted(lead_stats['thetao']):
        st = lead_stats['thetao'][ld]
        ss = lead_stats['so'][ld]
        print(f'+{ld:>3}d  {st["n"]:>6d}  {st["rmse"]:>8.3f}  {st["bias"]:>+8.3f}'
              f'  {ss["n"]:>6d}  {ss["rmse"]:>8.3f}  {ss["bias"]:>+8.3f}')

    # --- Per-month comparison plot ---
    month_pairs: dict[str, list[dict]] = defaultdict(list)
    for p in all_pairs:
        label = datetime.strptime(p['ic_date'], '%Y%m%d').strftime('%b %Y')
        month_pairs[label].append(p)

    if len(month_pairs) > 1:
        stats_by_label = {lbl: summarise_by_lead(ps)
                          for lbl, ps in sorted(month_pairs.items())}

    # --- Plots ---
    print('\nGenerating figures …')
    plot_rmse_lead(lead_stats,       output_dir / 'rmse_lead.png')
    if len(month_pairs) > 1:
        plot_rmse_lead_comparison(stats_by_label,
                                  output_dir / 'rmse_lead_comparison.png')
    plot_scatter(all_pairs,          output_dir / 'scatter.png')
    plot_temporal(ym_stats,          output_dir / 'temporal.png')
    plot_map(all_pairs,              output_dir / 'map.png')

    print(f'\nAll outputs written to {output_dir}/')


if __name__ == '__main__':
    main()

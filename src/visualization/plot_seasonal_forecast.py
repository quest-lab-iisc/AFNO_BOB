"""
Seasonal one-day-ahead forecast comparison panel.

Produces a 4-row × 6-column figure comparing GLORYS reanalysis against AFNO
predictions for SST (thetao) and SSS (so) across four seasonal regimes. Each
row shows one season; columns show [GLORYS | AFNO | |Error|] for SST then SSS.

Inputs:
    config_file (str): YAML config filename in config/ directory
    model_path (str): Path to .pth weights; defaults to results/models/{name}.pth
    dates (str): Comma-separated input dates dd-mm-yyyy (one per row)
    season_labels (str): Comma-separated row labels matching dates
    device (str): Compute device override, e.g. cuda:0 or cpu
    output (str): Output file path without extension
    sst_range (str): Optional "vmin,vmax" for SST colourbar (°C); default auto
    sss_range (str): Optional "vmin,vmax" for SSS colourbar (psu); default auto

Outputs:
    <output>.pdf (file): Vector figure for paper submission
    <output>.png (file): Raster figure at 300 dpi

Example:
    python src/visualization/plot_seasonal_forecast.py \\
        --config_file afno_bob_surf_e06p1.yaml \\
        --dates 14-01-2020,19-04-2020,14-07-2020,19-10-2020 \\
        --season_labels "Winter,Pre-monsoon,Monsoon,Post-monsoon" \\
        --device cuda:0
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import torch
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import FixedLocator

sys.path.append(str(Path(__file__).parent.parent))

from configmypy import ConfigPipeline, YamlConfig
from models.architectures.afno.afnonet import AFNONet
from data_pipeline.preprocessing.transformers.normalize_transform import PreprocessTransform
from inference.utils import postprocess_ocean_variable, date_to_day_index

try:
    import cmocean
    _SST_CMAP = cmocean.cm.thermal
    _SSS_CMAP = cmocean.cm.haline
except ImportError:
    _SST_CMAP = 'RdYlBu_r'
    _SSS_CMAP = 'viridis'

_ERR_CMAP = 'RdBu_r'
_LON_RANGE = (77, 99)
_LAT_RANGE = (4, 23)
_REF_DATE = '01-01-1993'

_NAME_MAP = {
    'AFNO_BoB_Surf_E14': 'AFNO RT',
    'AFNO_BoB_Surf_E13': 'AFNO 1T',
    'TFNO_BoB_Surf_E04': 'TFNO RT',
    'TFNO_BoB_Surf_E03': 'TFNO 1T',
    'FNO_BoB_Surf_E04':  'FNO RT',
    'UNO_BoB_Surf_E04':  'UNO RT',
}


def model_display_name(name):
    """Return a human-readable model label for use in figure titles and legends.

    Args:
        name (str): Raw experiment name (e.g. 'AFNO_BoB_Surf_E14').

    Returns:
        str: Display label ('AFNO RT', 'AFNO 1T', or the name unchanged).

    Example:
        >>> model_display_name('AFNO_BoB_Surf_E14')
        'AFNO RT'
    """
    return _NAME_MAP.get(name, name)

DEFAULT_DATES = ['14-01-2020', '19-04-2020', '14-07-2020', '19-10-2020']
DEFAULT_LABELS = ['Winter', 'Pre-monsoon', 'Monsoon', 'Post-monsoon']


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.

    Example:
        >>> args = parse_args()
    """
    p = argparse.ArgumentParser(description='Seasonal one-day-ahead SST/SSS forecast panel')
    p.add_argument('--config_file', default='afno_bob_surf_e06p1.yaml',
                   help='YAML config filename inside config/ (selects model + data paths)')
    p.add_argument('--model_path', default=None,
                   help='Explicit path to .pth weights; default: results/models/{name}.pth')
    p.add_argument('--dates', default=','.join(DEFAULT_DATES),
                   help='Comma-separated input dates dd-mm-yyyy, one per panel row')
    p.add_argument('--season_labels', default=','.join(DEFAULT_LABELS),
                   help='Comma-separated row labels matching --dates')
    p.add_argument('--device', default=None,
                   help='Compute device, e.g. cuda:0 or cpu (overrides config)')
    p.add_argument('--output', default=None,
                   help='Output path without extension; default: results/plots/{name}_seasonal_forecast')
    p.add_argument('--sst_range', default=None,
                   help='SST colourbar "vmin,vmax" in °C; default: auto from data')
    p.add_argument('--sss_range', default=None,
                   help='SSS colourbar "vmin,vmax" in psu; default: auto from data')
    return p.parse_args()


def load_config(config_file):
    """Load model and data configuration from a YAML file in config/.

    Args:
        config_file (str): Filename of the YAML config, e.g. afno_bob_surf_e06p1.yaml

    Returns:
        config: configmypy configuration object.

    Example:
        >>> config = load_config('afno_bob_surf_e06p1.yaml')
    """
    pipe = ConfigPipeline([
        YamlConfig(config_file, config_name='default', config_folder='config/'),
    ])
    return pipe.read_conf()


def load_model(config, model_path, device):
    """Load the appropriate model from a .pth weights file.

    Detects the architecture from the config (``tfno`` section → TFNOWrapper;
    otherwise → AFNONet) and loads saved weights.

    Args:
        config: Configuration object (defines model architecture).
        model_path (str): Path to the .pth file.
        device (torch.device): Device to load weights onto.

    Returns:
        torch.nn.Module: Model in eval mode on the specified device.

    Example:
        >>> model = load_model(config, 'results/models/AFNO_BoB_Surf_E06p1.pth', torch.device('cpu'))
    """
    arch = _detect_arch(config)

    if arch == 'tfno':
        import torch.nn.functional as F
        from neuralop.models import TFNO

        class TFNOWrapper(torch.nn.Module):
            """Thin wrapper matching neuralop TFNO to the pipeline's forward(x) interface."""

            def __init__(self, cfg):
                """Initialise TFNO from config.

                Args:
                    cfg: configmypy config object with a ``tfno`` section.

                Example:
                    >>> m = TFNOWrapper(config)
                """
                super().__init__()
                nl_map = {'gelu': F.gelu, 'relu': F.relu, 'tanh': torch.tanh}
                self.tfno = TFNO(
                    n_modes=tuple(cfg.tfno.n_modes),
                    in_channels=cfg.data.in_chs,
                    out_channels=cfg.data.out_chs,
                    hidden_channels=cfg.tfno.hidden_channels,
                    lifting_channel_ratio=cfg.tfno.lifting_channel_ratio,
                    projection_channel_ratio=cfg.tfno.projection_channel_ratio,
                    n_layers=cfg.tfno.n_layers,
                    factorization=cfg.tfno.factorization,
                    rank=cfg.tfno.rank,
                    non_linearity=nl_map.get(cfg.tfno.non_linearity, F.gelu),
                    use_channel_mlp=cfg.tfno.use_channel_mlp,
                    channel_mlp_expansion=cfg.tfno.channel_mlp_expansion,
                    channel_mlp_dropout=cfg.tfno.channel_mlp_dropout,
                    channel_mlp_skip=cfg.tfno.channel_mlp_skip,
                    fno_skip=cfg.tfno.fno_skip,
                    positional_embedding=cfg.tfno.positional_embedding,
                    norm=cfg.tfno.norm,
                )

            def forward(self, x):
                """Forward pass.

                Args:
                    x (torch.Tensor): (batch, in_channels, H, W).

                Returns:
                    torch.Tensor: (batch, out_channels, H, W).

                Example:
                    >>> y = m.forward(x)
                """
                return self.tfno(x)

        model = TFNOWrapper(config)
    else:
        model = AFNONet(config)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    # _metadata is an internal PyTorch OrderedDict attribute that occasionally
    # gets serialised as a regular key; strip it before load_state_dict.
    state_dict = {k: v for k, v in state_dict.items() if k != '_metadata'}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def load_norm_stats(config):
    """Load normalisation mean and variance statistics from mean_dir.

    Args:
        config: Configuration object with data.mean_dir.

    Returns:
        mean (dict): Per-variable mean arrays (numpy); None if file missing.
        variance (dict): Per-variable std arrays keyed as '{var}_std'; None if missing.

    Example:
        >>> mean, variance = load_norm_stats(config)
    """
    d = Path(config.data.mean_dir)
    mean, variance = {}, {}

    for var in ['thetao', 'so', 'uo', 'vo', 'zos']:
        f = d / f'mean_{var}_1993_2018_all_months.npy'
        mean[var] = np.load(f) if f.exists() else None

    for var in ['ssr', 'tp', 'u10', 'v10', 'msl', 'tcc']:
        f = d / f'{var}_mean_1993_2018_all_months.npy'
        mean[var] = np.load(f) if f.exists() else None

    for var in ['ssr', 'tp', 'msl']:
        f = d / f'{var}_var_1993_2018_all_months.npy'
        variance[f'{var}_std'] = np.sqrt(np.load(f)) if f.exists() else None

    return mean, variance


def _is_coupled(config):
    """Return True if config describes a coupled atm+ocean model (out_chs == 11)."""
    return config.data.out_chs == 11


def _detect_arch(config):
    """Return the model architecture identifier from the config.

    Checks for architecture-specific config sections: ``tfno`` → ``'tfno'``;
    otherwise defaults to ``'afno'``.

    Args:
        config: configmypy configuration object.

    Returns:
        str: One of ``'tfno'``, ``'afno'``.

    Example:
        >>> arch = _detect_arch(config)
    """
    # configmypy raises KeyError (not AttributeError) for missing keys,
    # so hasattr() does not work here — use an explicit try/except.
    try:
        _ = config.tfno
        return 'tfno'
    except (KeyError, AttributeError):
        return 'afno'


def _preprocess_atm(raw, var, mean, variance, transform):
    """Normalise one atmospheric variable slice to a (1, 224, 224) tensor.

    Args:
        raw (np.ndarray): Raw data slice from xarray, shape (1, H, W) or (1, 1, H, W).
        var (str): Variable name.
        mean (dict): Normalisation means.
        variance (dict): Normalisation standard deviations.
        transform (PreprocessTransform): Normalisation + interpolation transform.

    Returns:
        torch.Tensor: (1, 224, 224) normalised tensor.
    """
    if var == 'ssr':
        out = transform(raw, mean[var], variable=var, type='atm', variance=variance['ssr_std'])
    elif var == 'tp':
        out = transform(raw, mean[var], variable=var, type='atm', variance=variance['tp_std'])
    elif var == 'msl':
        out = transform(raw, mean[var], variable=var, type='atm', variance=variance['msl_std'])
    else:
        out = transform(raw, mean[var], variable=var, type='atm')
    return out.squeeze(1)   # (1, 224, 224)


def _preprocess_ocean(raw, var, mean, transform):
    """Normalise one ocean variable slice to a (1, 224, 224) tensor.

    Args:
        raw (np.ndarray): Raw data slice from xarray.
        var (str): Variable name.
        mean (dict): Normalisation means.
        transform (PreprocessTransform): Normalisation + interpolation transform.

    Returns:
        torch.Tensor: (1, 224, 224) normalised tensor.
    """
    out = transform(raw, mean[var], variable=var)
    return out.squeeze(1)   # (1, 224, 224)


def build_input(config, day_idx, ocean_ds, atm_ds, mean, variance, transform):
    """Assemble the 11-channel input tensor for one forward step.

    Ocean-only model (out_chs=5): atm(t+1) + ocean(t) — uses future atmospheric
    forcing, matching the NetCDFDataset training convention.
    Coupled model (out_chs=11): atm(t) + ocean(t).

    Args:
        config: Configuration object.
        day_idx (int): Absolute day index of the input ocean state (t).
        ocean_ds (xr.Dataset): Ocean variable dataset.
        atm_ds (xr.Dataset): Atmospheric variable dataset.
        mean (dict): Normalisation means.
        variance (dict): Normalisation standard deviations.
        transform (PreprocessTransform): Normalisation transform.

    Returns:
        torch.Tensor: (11, 224, 224) input tensor ready for model.forward().

    Example:
        >>> x = build_input(config, 9876, ocean_ds, atm_ds, mean, variance, transform)
    """
    atm_t = day_idx if _is_coupled(config) else day_idx + 1

    channels = []
    for var in config.data.atm_variable:
        raw = atm_ds[var][atm_t:atm_t + 1].values
        channels.append(_preprocess_atm(raw, var, mean, variance, transform))

    for var in config.data.variable:
        raw = ocean_ds[var][day_idx:day_idx + 1].values
        channels.append(_preprocess_ocean(raw, var, mean, transform))

    return torch.cat(channels, dim=0)   # (11, 224, 224)


def model_forward(model, x, device):
    """Run one model forward pass.

    Args:
        model (AFNONet): Loaded model in eval mode.
        x (torch.Tensor): (11, 224, 224) input tensor.
        device (torch.device): Compute device.

    Returns:
        np.ndarray: (out_chs, 224, 224) prediction in normalised space.

    Example:
        >>> output = model_forward(model, x, torch.device('cpu'))
    """
    with torch.no_grad():
        return model(x.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()


def postprocess_var(output_arr, config, var, mean):
    """Extract one variable from the model output and convert to physical units.

    Args:
        output_arr (np.ndarray): (out_chs, 224, 224) model output.
        config: Configuration object with data.out_variable list.
        var (str): Variable name, e.g. 'thetao' or 'so'.
        mean (dict): Normalisation means.

    Returns:
        np.ndarray: Field at native GLORYS grid resolution in physical units.

    Example:
        >>> sst = postprocess_var(output, config, 'thetao', mean)
    """
    ch = list(config.data.out_variable).index(var)
    return postprocess_ocean_variable(output_arr[ch], mean[var], var)


def load_truth(ocean_ds, var, day_idx):
    """Load ground-truth field at day_idx+1 in native GLORYS units.

    Args:
        ocean_ds (xr.Dataset): Ocean variable dataset.
        var (str): Variable name.
        day_idx (int): Index of the input day; truth is at day_idx + 1.

    Returns:
        np.ndarray: (H, W) field in physical units; NaN on land pixels.

    Example:
        >>> truth = load_truth(ocean_ds, 'thetao', 9876)
    """
    return ocean_ds[var][day_idx + 1].values.squeeze()


def get_coords(ocean_ds, var='thetao'):
    """Extract 1-D longitude and latitude coordinate arrays from the dataset.

    Falls back to linearly-spaced arrays over the Bay of Bengal domain if
    coordinate names cannot be determined.

    Args:
        ocean_ds (xr.Dataset): Ocean variable dataset.
        var (str): Variable to read coordinates from.

    Returns:
        lons (np.ndarray): 1-D longitude array.
        lats (np.ndarray): 1-D latitude array.

    Example:
        >>> lons, lats = get_coords(ocean_ds)
    """
    da = ocean_ds[var]
    lons = lats = None

    for name in ['longitude', 'lon', 'x', 'nav_lon']:
        if name in da.coords:
            arr = da.coords[name].values
            lons = arr[0] if arr.ndim > 1 else arr
            break

    for name in ['latitude', 'lat', 'y', 'nav_lat']:
        if name in da.coords:
            arr = da.coords[name].values
            lats = arr[:, 0] if arr.ndim > 1 else arr
            break

    if lons is None:
        lons = np.linspace(_LON_RANGE[0], _LON_RANGE[1], da.shape[-1])
    if lats is None:
        lats = np.linspace(_LAT_RANGE[0], _LAT_RANGE[1], da.shape[-2])

    return lons, lats


def run_season_forecasts(config, model, dates, ocean_ds, atm_ds, mean, variance, device):
    """Run one-day-ahead forecasts for each input date and collect SST/SSS results.

    Args:
        config: Configuration object.
        model (AFNONet): Loaded model in eval mode.
        dates (list[str]): Input dates in dd-mm-yyyy format.
        ocean_ds (xr.Dataset): Ocean variable dataset.
        atm_ds (xr.Dataset): Atmospheric variable dataset.
        mean (dict): Normalisation means.
        variance (dict): Normalisation standard deviations.
        device (torch.device): Compute device.

    Returns:
        list[dict]: One dict per date with keys: pred_date (str), sst_pred,
            sst_truth, sst_err, sss_pred, sss_truth, sss_err, lons, lats
            (all arrays at native GLORYS resolution).

    Example:
        >>> results = run_season_forecasts(config, model, ['14-01-2020'], ...)
    """
    transform = PreprocessTransform(config)
    lons, lats = get_coords(ocean_ds)
    results = []

    for date_str in dates:
        day_idx = date_to_day_index(date_str, _REF_DATE)
        pred_date = (
            datetime.strptime(date_str, '%d-%m-%Y') + timedelta(days=1)
        ).strftime('%d-%b-%Y')
        print(f'  {date_str} → {pred_date}')

        x = build_input(config, day_idx, ocean_ds, atm_ds, mean, variance, transform)
        output_arr = model_forward(model, x, device)

        sst_pred  = postprocess_var(output_arr, config, 'thetao', mean)
        sss_pred  = postprocess_var(output_arr, config, 'so',     mean)
        sst_truth = load_truth(ocean_ds, 'thetao', day_idx)
        sss_truth = load_truth(ocean_ds, 'so',     day_idx)

        # NaN propagates from truth (land pixels) into the error naturally
        sst_err = sst_pred - sst_truth
        sss_err = sss_pred - sss_truth

        results.append({
            'pred_date': pred_date,
            'sst_pred':  sst_pred,
            'sst_truth': sst_truth,
            'sst_err':   sst_err,
            'sss_pred':  sss_pred,
            'sss_truth': sss_truth,
            'sss_err':   sss_err,
            'lons':      lons,
            'lats':      lats,
        })

    return results


def compute_color_ranges(results, sst_range=None, sss_range=None):
    """Compute shared colourbar ranges from all season data.

    Percentiles are computed over valid (non-NaN) ocean pixels pooled across
    all seasons. Field panels use the 2nd–98th percentile range; error panels
    run from 0 to the 98th percentile.

    Args:
        results (list[dict]): Output of run_season_forecasts.
        sst_range (tuple | None): (vmin, vmax) override for SST field panels.
        sss_range (tuple | None): (vmin, vmax) override for SSS field panels.

    Returns:
        dict: Keys sst_vmin, sst_vmax, sst_err_vmax, sss_vmin, sss_vmax, sss_err_vmax.

    Example:
        >>> ranges = compute_color_ranges(results, sst_range=(28.0, 31.0))
    """
    def _pool(arrays):
        return np.concatenate([a[~np.isnan(a)].ravel() for a in arrays])

    sst_all  = _pool([r['sst_pred']  for r in results] + [r['sst_truth'] for r in results])
    sss_all  = _pool([r['sss_pred']  for r in results] + [r['sss_truth'] for r in results])
    sst_errs = _pool([r['sst_err']   for r in results])
    sss_errs = _pool([r['sss_err']   for r in results])

    if sst_range:
        s_vmin, s_vmax = sst_range
    else:
        s_vmin, s_vmax = np.percentile(sst_all, 2), np.percentile(sst_all, 98)

    if sss_range:
        ss_vmin, ss_vmax = sss_range
    else:
        ss_vmin, ss_vmax = np.percentile(sss_all, 2), np.percentile(sss_all, 98)

    # Symmetric error range: ±98th percentile of absolute errors
    sst_elim = np.percentile(np.abs(sst_errs), 98)
    sss_elim = np.percentile(np.abs(sss_errs), 98)

    return {
        'sst_vmin':     s_vmin,
        'sst_vmax':     s_vmax,
        'sst_err_lim':  sst_elim,
        'sss_vmin':     ss_vmin,
        'sss_vmax':     ss_vmax,
        'sss_err_lim':  sss_elim,
    }


def setup_map(ax, left_labels=False, bottom_labels=False):
    """Configure a cartopy axes for the Bay of Bengal domain.

    Args:
        ax: Cartopy axes with PlateCarree projection.
        left_labels (bool): Draw latitude gridline labels on the left.
        bottom_labels (bool): Draw longitude gridline labels on the bottom.

    Example:
        >>> setup_map(ax, left_labels=True, bottom_labels=True)
    """
    ax.set_extent([_LON_RANGE[0], _LON_RANGE[1], _LAT_RANGE[0], _LAT_RANGE[1]],
                  crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      facecolor='#d3d3d3', edgecolor='#555555',
                   linewidth=0.4, zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.6, color='gray',
                      alpha=0.7, linestyle='--')
    gl.xlocator     = FixedLocator([80, 85, 90, 95, 100])
    gl.ylocator     = FixedLocator([5, 10, 15, 20, 25])
    gl.top_labels    = False
    gl.right_labels  = False
    gl.left_labels   = left_labels
    gl.bottom_labels = bottom_labels
    gl.xlabel_style  = {'size': 6}
    gl.ylabel_style  = {'size': 6}


def draw_panel(ax, data, lons, lats, cmap, norm):
    """Draw one map panel using pcolormesh.

    Args:
        ax: Cartopy axes already configured with setup_map.
        data (np.ndarray): (H, W) field; NaN values are transparent.
        lons (np.ndarray): 1-D longitude array of length W.
        lats (np.ndarray): 1-D latitude array of length H.
        cmap: Matplotlib colormap.
        norm (mcolors.Normalize): Colour normalisation.

    Returns:
        matplotlib.collections.QuadMesh: The rendered mesh (for colourbar).

    Example:
        >>> mesh = draw_panel(ax, sst_field, lons, lats, cmap, norm)
    """
    lon2d, lat2d = np.meshgrid(lons, lats)
    return ax.pcolormesh(lon2d, lat2d, data, cmap=cmap, norm=norm,
                         transform=ccrs.PlateCarree(), zorder=1, shading='auto')


def _error_metrics(err):
    """Compute RMSE and MAE over valid (non-NaN) pixels of an error field.

    Args:
        err (np.ndarray): (H, W) signed error array (pred − truth); NaN on land.

    Returns:
        rmse (float): Root-mean-square error over ocean pixels.
        mae (float): Mean absolute error over ocean pixels.

    Example:
        >>> rmse, mae = _error_metrics(sst_err)
    """
    valid = err[~np.isnan(err)]
    rmse = np.sqrt(np.mean(valid ** 2))
    mae  = np.mean(np.abs(valid))
    return rmse, mae


def build_figure(results, season_labels, ranges, config, output_path, arch_label='Model'):
    """Assemble and save the seasonal 4×6 forecast panel figure.

    Args:
        results (list[dict]): Season forecast data from run_season_forecasts.
        season_labels (list[str]): Row labels (one per season / date).
        ranges (dict): Shared colourbar ranges from compute_color_ranges.
        config: Configuration object (for the figure title).
        output_path (str): Output file path without extension.
        arch_label (str): Short model-type label used in the column header (e.g. ``'AFNO'``, ``'TFNO'``).

    Returns:
        str: Path to the saved PDF file.

    Example:
        >>> build_figure(results, DEFAULT_LABELS, ranges, config, 'results/plots/seasonal', arch_label='AFNO')
    """
    n = len(results)

    sst_norm  = mcolors.Normalize(ranges['sst_vmin'],    ranges['sst_vmax'])
    sst_enorm = mcolors.TwoSlopeNorm(vcenter=0,
                                     vmin=-ranges['sst_err_lim'],
                                     vmax= ranges['sst_err_lim'])
    sss_norm  = mcolors.Normalize(ranges['sss_vmin'],    ranges['sss_vmax'])
    sss_enorm = mcolors.TwoSlopeNorm(vcenter=0,
                                     vmin=-ranges['sss_err_lim'],
                                     vmax= ranges['sss_err_lim'])

    # column order: SST truth, SST pred, SST err, SSS truth, SSS pred, SSS err
    col_cmaps = [_SST_CMAP, _SST_CMAP, _ERR_CMAP, _SSS_CMAP, _SSS_CMAP, _ERR_CMAP]
    col_norms = [sst_norm,  sst_norm,  sst_enorm, sss_norm,  sss_norm,  sss_enorm]

    fig = plt.figure(figsize=(18, 3.6 * n + 1.4))
    gs  = GridSpec(n + 1, 6,
                   height_ratios=[1] * n + [0.06],
                   hspace=0.12, wspace=0.07,
                   top=0.91, bottom=0.08, left=0.08, right=0.99)

    # --- map axes ---
    axes = []
    for r in range(n):
        row = []
        for c in range(6):
            ax = fig.add_subplot(gs[r, c], projection=ccrs.PlateCarree())
            setup_map(ax, left_labels=(c == 0), bottom_labels=(r == n - 1))
            row.append(ax)
        axes.append(row)

    # --- draw panels ---
    for r, res in enumerate(results):
        lons, lats = res['lons'], res['lats']
        panels = [
            res['sst_truth'], res['sst_pred'], res['sst_err'],
            res['sss_truth'], res['sss_pred'], res['sss_err'],
        ]
        for c, (data, cmap, norm) in enumerate(zip(panels, col_cmaps, col_norms)):
            draw_panel(axes[r][c], data, lons, lats, cmap, norm)

    # --- RMSE / MAE annotations on error panels (cols 2 and 5) ---
    for r, res in enumerate(results):
        for err_arr, col in [(res['sst_err'], 2), (res['sss_err'], 5)]:
            rmse, mae = _error_metrics(err_arr)
            unit = '°C' if col == 2 else 'psu'
            axes[r][col].text(
                0.03, 0.97,
                f'RMSE={rmse:.3f} {unit}\nMAE ={mae:.3f} {unit}',
                transform=axes[r][col].transAxes,
                ha='left', va='top', fontsize=7, fontfamily='monospace',
                bbox=dict(facecolor='white', alpha=0.65, edgecolor='none', pad=2),
            )

    # --- column sub-headers (above first row only) ---
    for c, title in enumerate(['GLORYS', arch_label, 'Error',
                                'GLORYS', arch_label, 'Error']):
        axes[0][c].annotate(
            title, xy=(0.5, 1.04), xycoords='axes fraction',
            ha='center', va='bottom', fontsize=9, fontweight='bold',
            annotation_clip=False,
        )

    # --- group headers (SST / SSS) above middle column of each group ---
    for col_idx, label in [(1, 'SST  (°C)'), (4, 'SSS  (psu)')]:
        axes[0][col_idx].annotate(
            label, xy=(0.5, 1.18), xycoords='axes fraction',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            annotation_clip=False,
        )

    # --- season labels on the left of each row (vertical, bottom-to-top) ---
    for r, (res, label) in enumerate(zip(results, season_labels)):
        axes[r][0].text(
            -0.14, 0.5, f'{label}  ({res["pred_date"]})',
            transform=axes[r][0].transAxes,
            ha='center', va='center', fontsize=9, fontweight='bold',
            rotation=90,
        )

    # --- colourbar axes (last GridSpec row, merged cells) ---
    cax_sst_f = fig.add_subplot(gs[n, 0:2])
    cax_sst_e = fig.add_subplot(gs[n, 2])
    cax_sss_f = fig.add_subplot(gs[n, 3:5])
    cax_sss_e = fig.add_subplot(gs[n, 5])

    for sm, cax, label in [
        (plt.cm.ScalarMappable(cmap=_SST_CMAP, norm=sst_norm),  cax_sst_f, 'SST (°C)'),
        (plt.cm.ScalarMappable(cmap=_ERR_CMAP,  norm=sst_enorm), cax_sst_e, 'Error (°C)'),
        (plt.cm.ScalarMappable(cmap=_SSS_CMAP,  norm=sss_norm),  cax_sss_f, 'SSS (psu)'),
        (plt.cm.ScalarMappable(cmap=_ERR_CMAP,  norm=sss_enorm), cax_sss_e, 'Error (psu)'),
    ]:
        sm.set_array([])
        fig.colorbar(sm, cax=cax, orientation='horizontal', label=label)

    fig.suptitle(
        f'One-day-ahead SST and SSS forecasts — {arch_label} Model',
        fontsize=12, fontweight='bold', y=0.965,
    )

    for ext in ('pdf', 'png'):
        path = f'{output_path}.{ext}'
        fig.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f'Saved: {path}')

    plt.close(fig)
    return f'{output_path}.pdf'


def main():
    """Entry point: parse arguments, load model, run forecasts, build figure.

    Example:
        >>> # run from project root:
        >>> # python src/visualization/plot_seasonal_forecast.py --config_file afno_bob_surf_e06p1.yaml
    """
    args = parse_args()

    dates         = [d.strip() for d in args.dates.split(',')]
    season_labels = [s.strip() for s in args.season_labels.split(',')]

    if len(dates) != len(season_labels):
        raise ValueError(
            f'--dates has {len(dates)} items but --season_labels has {len(season_labels)}'
        )

    print('=== Loading configuration ===')
    config = load_config(args.config_file)

    device_str = args.device or config.device
    device = torch.device(
        device_str if torch.cuda.is_available() or 'cpu' in device_str else 'cpu'
    )
    print(f'Device: {device}')

    arch = _detect_arch(config)
    arch_label = model_display_name(config.name) if config.name in _NAME_MAP \
        else {'afno': 'AFNO', 'tfno': 'TFNO'}.get(arch, arch.upper())
    print(f'Architecture: {arch_label}')

    model_path = args.model_path or f'{config.results.model_dir}/{config.name}.pth'
    if not Path(model_path).exists():
        raise FileNotFoundError(f'Model weights not found: {model_path}')
    print(f'=== Loading model: {model_path} ===')
    model = load_model(config, model_path, device)

    print('=== Loading normalisation statistics ===')
    mean, variance = load_norm_stats(config)

    print('=== Opening datasets ===')
    data_dir = Path(config.data.data_dir)
    ocean_ds = xr.open_dataset(data_dir / f'{config.data.file_prefix}.nc')
    atm_ds   = xr.open_dataset(data_dir / f'{config.data.file_prefix_atm}.nc')

    print('=== Running seasonal forecasts ===')
    results = run_season_forecasts(
        config, model, dates, ocean_ds, atm_ds, mean, variance, device
    )

    sst_range = (
        tuple(float(v) for v in args.sst_range.split(',')) if args.sst_range else None
    )
    sss_range = (
        tuple(float(v) for v in args.sss_range.split(',')) if args.sss_range else None
    )
    ranges = compute_color_ranges(results, sst_range=sst_range, sss_range=sss_range)

    output_path = args.output or f'{config.results.plot_dir}/{config.name}_seasonal_forecast'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print('=== Building figure ===')
    build_figure(results, season_labels, ranges, config, output_path, arch_label=arch_label)
    print('=== Done ===')


if __name__ == '__main__':
    main()

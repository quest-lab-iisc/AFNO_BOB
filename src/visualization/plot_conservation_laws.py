"""Analyse and plot three physical conservation law diagnostics from AFNO RT forecasts.

Produces a 3-row × 2-column figure (Figure~\ref{fig:conservation_laws}) assessing
how well the model respects ocean physics implicitly learned from GLORYS12 reanalysis:

  Row 1 — Geostrophic balance:
      Scatter of SSH-derived geostrophic speed against predicted current speed at each
      ocean grid point, pooled over all evaluation dates.  A high Pearson r indicates
      the model's SSH and velocity fields satisfy geostrophic balance.

  Row 2 — Frontal co-location:
      2-D spatial map of SST gradient magnitude (colour) overlaid with predicted surface
      current streamlines (arrows), for a representative monsoon forecast.  SST fronts
      should align with current convergence / shear zones.

  Row 3 — Spatial power spectra:
      Azimuthally-averaged 2-D power spectral density of predicted SST and SSH vs.
      GLORYS12 ground truth, showing that the model preserves realistic spatial length
      scales (Rossby radius).

Lead days 3 and 7 are shown in the left and right columns respectively.

Inputs:
    --config_file (str)  : Config YAML in config/ (default: afno_bob_surf_e12p1.yaml,
                           overridden to use E11p1 weights via --model_path).
    --model_path (str)   : Path to AFNO RT weights (default: results/models/AFNO_BoB_Surf_E11p1.pth).
    --name (str)         : Experiment name (default: AFNO_BoB_Surf_E11p1).
    --eval_dates (str list): IC dates (dd-mm-yyyy) to use for geostrophic / spectra
                             diagnostics (default: one per month of 2020).
    --monsoon_date (str) : IC date for the frontal co-location map (default: 01-07-2020).
    --output (str)       : Output path without extension
                           (default: results/figures/conservation_laws).
    --device (str)       : PyTorch device (default: cuda:0).
    --dpi (int)          : Raster DPI (default: 300).
    --width (float)      : Figure width in inches (default: 8.0).

Outputs:
    <output>.pdf   Publication-quality vector figure.
    <output>.png   Raster copy at --dpi.

Example:
    python src/visualization/plot_conservation_laws.py \\
        --model_path results/models/AFNO_BoB_Surf_E11p1.pth \\
        --name AFNO_BoB_Surf_E11p1 \\
        --output results/figures/conservation_laws
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import xarray as xr
from scipy import signal
from scipy.stats import pearsonr
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

sys.path.append(str(Path(__file__).parent.parent))

import xarray as xr

from configmypy import ConfigPipeline, YamlConfig
from inference.run_metrics import (
    load_model, date_to_day_index,
    run_autoregressive_forecast, postprocess_forecasts, load_ground_truth_batch,
)
from inference.utils import postprocess_ocean_variable


# ── physical constants ────────────────────────────────────────────────────────
G     = 9.81          # m s-2
OMEGA = 7.2921e-5     # rad s-1


# ── grid helpers ─────────────────────────────────────────────────────────────

def make_grid(n_lat, n_lon,
              lat_min=4.0, lat_max=23.0, lon_min=77.0, lon_max=99.0):
    """Build a regular lat/lon grid matching the postprocessed field dimensions.

    Args:
        n_lat (int): Number of latitude points (rows) in the postprocessed field.
        n_lon (int): Number of longitude points (cols) in the postprocessed field.
        lat_min (float): Southern boundary (degrees N).
        lat_max (float): Northern boundary (degrees N).
        lon_min (float): Western boundary (degrees E).
        lon_max (float): Eastern boundary (degrees E).

    Returns:
        tuple: (lat2d, lon2d, dy_m, dx_m) — 2-D coordinate arrays and
               metre spacing arrays, shapes (n_lat, n_lon).

    Example:
        >>> lat2d, lon2d, dy_m, dx_m = make_grid(229, 265)
    """
    lat1d = np.linspace(lat_min, lat_max, n_lat)
    lon1d = np.linspace(lon_min, lon_max, n_lon)
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)

    dlat = (lat_max - lat_min) / (n_lat - 1)
    dlon = (lon_max - lon_min) / (n_lon - 1)
    dy_m = dlat * 111_000.0 * np.ones_like(lat2d)
    dx_m = dlon * 111_000.0 * np.cos(np.deg2rad(lat2d))

    return lat2d, lon2d, dy_m, dx_m


def coriolis(lat2d):
    """Compute the Coriolis parameter f = 2Ω sin(φ).

    Args:
        lat2d (np.ndarray): 2-D latitude array in degrees.

    Returns:
        np.ndarray: Coriolis parameter in rad s⁻¹, same shape as lat2d.

    Example:
        >>> f = coriolis(lat2d)
    """
    return 2.0 * OMEGA * np.sin(np.deg2rad(lat2d))


def geostrophic_currents(ssh, dy_m, dx_m, f):
    """Compute geostrophic surface currents from SSH via centred finite differences.

    u_g = -(g/f) ∂η/∂y
    v_g =  (g/f) ∂η/∂x

    Interior points only; boundary rows/columns are set to NaN.

    Args:
        ssh (np.ndarray): SSH field (m), shape (H, W).
        dy_m (np.ndarray): Grid spacing in y (m), shape (H, W).
        dx_m (np.ndarray): Grid spacing in x (m), shape (H, W).
        f (np.ndarray): Coriolis parameter (rad s⁻¹), shape (H, W).

    Returns:
        tuple: (ug, vg) each shape (H, W), in m s⁻¹.  Boundary = NaN.

    Example:
        >>> ug, vg = geostrophic_currents(ssh, dy_m, dx_m, f)
    """
    H, W = ssh.shape
    ug = np.full_like(ssh, np.nan)
    vg = np.full_like(ssh, np.nan)

    deta_dy = (ssh[2:, 1:-1] - ssh[:-2, 1:-1]) / (2.0 * dy_m[1:-1, 1:-1])
    deta_dx = (ssh[1:-1, 2:] - ssh[1:-1, :-2]) / (2.0 * dx_m[1:-1, 1:-1])

    f_int = f[1:-1, 1:-1]
    safe  = np.abs(f_int) > 1e-10

    ug[1:-1, 1:-1] = np.where(safe, -(G / f_int) * deta_dy, np.nan)
    vg[1:-1, 1:-1] = np.where(safe, (G / f_int) * deta_dx,  np.nan)

    return ug, vg


def gradient_magnitude(field):
    """Compute the horizontal gradient magnitude of a 2-D scalar field.

    Args:
        field (np.ndarray): 2-D scalar field, shape (H, W).

    Returns:
        np.ndarray: |∇field|, shape (H, W).

    Example:
        >>> mag = gradient_magnitude(sst)
    """
    gy, gx = np.gradient(field)
    return np.sqrt(gx**2 + gy**2)


def power_spectrum_1d(field, dx=1.0):
    """Compute the azimuthally-averaged 2-D power spectral density of a 2-D field.

    Applies a 2-D Hann window before FFT to reduce spectral leakage.

    Args:
        field (np.ndarray): 2-D real-valued field, shape (H, W).
        dx (float): Isotropic grid spacing in km (default 1.0).

    Returns:
        tuple: (k_km, psd) — wavenumber array (cycles per km) and
               radially-averaged PSD (units²·km), both 1-D.

    Example:
        >>> k, psd = power_spectrum_1d(sst_field, dx=10.0)
    """
    H, W = field.shape
    win  = np.outer(np.hanning(H), np.hanning(W))
    f2d  = np.fft.fftshift(np.fft.fft2(field * win))
    psd2d = (np.abs(f2d) ** 2) / (H * W)

    kx = np.fft.fftshift(np.fft.fftfreq(W, d=dx))
    ky = np.fft.fftshift(np.fft.fftfreq(H, d=dx))
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

    k_bins  = np.linspace(0, K.max(), min(H, W) // 2)
    psd_1d  = np.zeros(len(k_bins) - 1)
    for i in range(len(k_bins) - 1):
        mask = (K >= k_bins[i]) & (K < k_bins[i + 1])
        if mask.any():
            psd_1d[i] = psd2d[mask].mean()

    k_centres = 0.5 * (k_bins[:-1] + k_bins[1:])
    return k_centres, psd_1d


# ── northern boundary taper ──────────────────────────────────────────────────

def apply_north_taper(field, target=0.0, n_rows=20):
    """Blend the northern n_rows of a 2-D field toward a target value via a
    cosine taper, suppressing model boundary artefacts smoothly.

    The weight w(t) = 0.5*(1 + cos(π t/(n_rows-1))) runs from 1 at the
    southern edge of the taper zone (t=0, row -n_rows) to 0 at the northern
    edge (t=n_rows-1, row -1):

        field_tapered[row] = w * field[row] + (1 - w) * target

    Use target=0 for currents and gradient fields; use target=np.nanmean(field[:-n_rows])
    for SST/SSS so the transition is toward a physically plausible background.

    Args:
        field (np.ndarray): 2-D field, shape (H, W).
        target (float or np.ndarray): Value(s) to taper toward (default 0).
            Scalar or broadcastable array of shape (H, W).
        n_rows (int): Number of rows from the northern edge to taper (default 20).

    Returns:
        np.ndarray: Tapered field, same shape as input.

    Example:
        >>> tapered = apply_north_taper(sst, target=np.nanmean(sst[:-20]), n_rows=20)
        >>> tapered = apply_north_taper(front_mag, target=0.0, n_rows=20)
    """
    result = field.copy()
    t = np.arange(n_rows, dtype=float)
    weights = 0.5 * (1.0 + np.cos(np.pi * t / (n_rows - 1)))  # 1 → 0
    for i, w in enumerate(weights):
        row = -(n_rows - i)          # -20, -19, …, -1
        result[row, :] = w * field[row, :] + (1.0 - w) * target
    return result


# ── northern boundary fill ───────────────────────────────────────────────────

def fill_north_rows(field, land_mask, n_rows=20):
    """Fill the northernmost n_rows of an ocean field by cubic interpolation from
    valid ocean pixels south of the fill zone, exactly as done in
    plot_ensemble_season_comparison.py.

    The model output carries artefacts in the north rows because those pixels
    were trained on mean-filled land values.  Replacing them with values
    extrapolated from clean ocean pixels to the south gives a physically
    meaningful extension that produces smooth gradients.

    Args:
        field (np.ndarray): 2-D field, shape (H, W).  Values may be artefacts
            in the top n_rows rows.
        land_mask (np.ndarray): Boolean (H, W); True = land.
        n_rows (int): Number of rows from the northern edge to fill (default 20).

    Returns:
        np.ndarray: Field with top n_rows ocean pixels replaced by interpolated
            values; land pixels and southern rows are unchanged.

    Example:
        >>> filled = fill_north_rows(raw_224, land_mask_224, n_rows=20)
    """
    out = field.copy()

    # Source: valid ocean pixels south of the fill zone
    src = ~land_mask & np.isfinite(out)
    src[-n_rows:, :] = False

    # Target: ocean pixels inside the fill zone
    tgt = np.zeros(land_mask.shape, dtype=bool)
    tgt[-n_rows:, :] = True
    tgt &= ~land_mask

    if src.any() and tgt.any():
        ys, xs = np.where(src)
        src_vals = out[ys, xs]
        fy, fx = np.where(tgt)

        filled = griddata(np.stack([ys, xs], axis=1), src_vals,
                          np.stack([fy, fx], axis=1), method='cubic')

        # Fall back to nearest where cubic returns NaN (outside convex hull)
        nan_idx = np.isnan(filled)
        if nan_idx.any():
            filled[nan_idx] = griddata(
                np.stack([ys, xs], axis=1), src_vals,
                np.stack([fy[nan_idx], fx[nan_idx]], axis=1), method='nearest'
            )

        out[fy, fx] = filled

    return out


# ── data loading ─────────────────────────────────────────────────────────────

def load_config_and_model(config_file, model_path, name, device_str):
    """Load configuration and model weights.

    Args:
        config_file (str): YAML config filename in config/.
        model_path (str): Path to model .pth weights file.
        name (str): Experiment name override.
        device_str (str): PyTorch device string (e.g. 'cuda:0').

    Returns:
        tuple: (config, model, device) ready for inference.

    Example:
        >>> config, model, device = load_config_and_model(
        ...     'afno_bob_surf_e12p1.yaml',
        ...     'results/models/AFNO_BoB_Surf_E11p1.pth',
        ...     'AFNO_BoB_Surf_E11p1', 'cuda:0')
    """
    pipe = ConfigPipeline([
        YamlConfig(f'./{config_file}', config_name='default', config_folder='config/'),
        YamlConfig(config_folder='config/'),
    ])
    config = pipe.read_conf()
    config.name = name
    config.device = device_str

    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    model  = load_model(config, model_path, device)
    model.eval()
    return config, model, device


def forecast_at_lead(config, model, device, ocean_data, atm_data,
                     date_str, lead_day, mask_north=True, return_raw=False):
    """Run autoregressive forecast from a given IC date and return the field at lead_day.

    Args:
        config: Model configuration object.
        model: Loaded AFNO model.
        device (torch.device): Compute device.
        ocean_data (xr.Dataset): Ocean NetCDF dataset.
        atm_data (xr.Dataset): Atmospheric NetCDF dataset.
        date_str (str): IC date as 'dd-mm-yyyy'.
        lead_day (int): Lead day to extract (1-indexed).
        mask_north (bool): If True (default), set the northern 20 rows to NaN as in
            standard postprocessing.  Ignored when return_raw=True.
        return_raw (bool): If True, return the model's native 224×224 output with
            mean restored (for SST/SSS) but without any spatial interpolation.
            Use this for diagnostics that must be computed on the clean model grid
            (e.g. frontal co-location) to avoid interpolation artefacts.

    Returns:
        tuple: (pred_dict, gt_dict) each mapping variable name → 2-D np.ndarray.
               Shape is 224×224 when return_raw=True, else the original grid size.

    Example:
        >>> pred, gt = forecast_at_lead(config, model, device,
        ...                             ocean_data, atm_data, '01-07-2020', 3)
    """
    import torch.nn.functional as _F
    ref_date = config.evaluation.reference_date
    idx      = date_to_day_index(date_str, ref_date)

    fc, md = run_autoregressive_forecast(
        config, model, idx, lead_day, device, ocean_data, atm_data
    )

    if return_raw:
        # Stay on the clean 224×224 model grid; restore mean for SST/SSS so that
        # gradient magnitudes are in physical units (°C, psu).
        pp_raw = {}
        for v in config.data.out_variable:
            raw = fc[lead_day][v]          # (224, 224), normalised
            if md[v] is not None:
                # Downsample the stored mean from original grid to 224×224
                mean_t = torch.tensor(md[v]).unsqueeze(0).float()  # (1,1,H,W) or (1,H,W)
                if mean_t.dim() == 3:
                    mean_t = mean_t.unsqueeze(0)
                mean_224 = _F.interpolate(mean_t, size=(224, 224),
                                          mode='bilinear',
                                          align_corners=False).squeeze().numpy()
                pp_raw[v] = raw + mean_224
            else:
                pp_raw[v] = raw
        gt_raw = load_ground_truth_batch(config, idx, lead_day, ocean_data)
        return pp_raw, {v: gt_raw[lead_day][v] for v in config.data.out_variable}

    if mask_north:
        pp = postprocess_forecasts(fc, md, config)
    else:
        # Postprocess and apply a cosine taper toward climatology over the
        # northern boundary rows instead of hard-clamping to NaN.
        # n_rows comes from config so models without north masking (n_rows=0)
        # skip the taper entirely.
        n_north = getattr(config.data, 'north_mask_rows', 20)
        pp = {}
        for lead_time, preds in fc.items():
            pp[lead_time] = {}
            for v in config.data.out_variable:
                field = postprocess_ocean_variable(preds[v], md[v], v)
                if n_north > 0:
                    clim = md[v].squeeze() if md[v] is not None else None
                    pp[lead_time][v] = apply_north_taper(field, clim, n_rows=n_north)
                else:
                    pp[lead_time][v] = field

    gt_raw = load_ground_truth_batch(config, idx, lead_day, ocean_data)
    pred_dict = {v: pp[lead_day][v] for v in config.data.out_variable}
    gt_dict   = {v: gt_raw[lead_day][v] for v in config.data.out_variable}
    return pred_dict, gt_dict


# ── diagnostic computations ───────────────────────────────────────────────────

def compute_geostrophic_scatter(config, model, device, ocean_data, atm_data,
                                 eval_dates, lead_day, lat2d, dx_m, dy_m, f):
    """Compare geostrophic speeds derived from GLORYS SSH vs AFNO-predicted SSH.

    For each IC date, derives geostrophic current speed from both the ground-truth
    SSH (GLORYS) and the model-predicted SSH using the same geostrophic operator,
    then pools the point-wise pairs across all dates for scatter plotting.

    Args:
        config: Model config.
        model: AFNO model.
        device (torch.device): Compute device.
        ocean_data (xr.Dataset): Ocean NetCDF.
        atm_data (xr.Dataset): Atmospheric NetCDF.
        eval_dates (list[str]): IC dates as 'dd-mm-yyyy'.
        lead_day (int): Forecast lead day.
        lat2d (np.ndarray): 2-D latitude grid.
        dx_m (np.ndarray): Zonal grid spacing in metres.
        dy_m (np.ndarray): Meridional grid spacing in metres.
        f (np.ndarray): Coriolis parameter.

    Returns:
        tuple: (geo_glorys, geo_afno, r) — 1-D pooled arrays and Pearson r.

    Example:
        >>> gs, ga, r = compute_geostrophic_scatter(config, model, device,
        ...     od, ad, dates, 3, lat2d, dx_m, dy_m, f)
    """
    geo_glorys_all = []
    geo_afno_all   = []

    for date_str in eval_dates:
        try:
            pred, gt = forecast_at_lead(config, model, device, ocean_data, atm_data,
                                        date_str, lead_day)
        except Exception as e:
            print(f'  Skipping {date_str}: {e}')
            continue

        ssh_afno   = pred.get('zos', pred.get('ZOS'))
        ssh_glorys = gt.get('zos',  gt.get('ZOS'))
        if ssh_afno is None or ssh_glorys is None:
            continue

        ug_afno,   vg_afno   = geostrophic_currents(ssh_afno,   dy_m, dx_m, f)
        ug_glorys, vg_glorys = geostrophic_currents(ssh_glorys, dy_m, dx_m, f)
        spd_afno   = np.sqrt(ug_afno**2   + vg_afno**2)
        spd_glorys = np.sqrt(ug_glorys**2 + vg_glorys**2)

        mask = np.isfinite(spd_afno) & np.isfinite(spd_glorys)
        geo_glorys_all.append(spd_glorys[mask][::4])
        geo_afno_all.append(spd_afno[mask][::4])

    geo_glorys = np.concatenate(geo_glorys_all)
    geo_afno   = np.concatenate(geo_afno_all)
    r, _       = pearsonr(geo_glorys, geo_afno)
    return geo_glorys, geo_afno, r


def compute_frontal_colocation(pred_dict, land_mask, n_rows=20):
    """Compute SST gradient magnitude and current vorticity for frontal co-location plot.

    Args:
        pred_dict (dict): Variable name → 2-D predicted field.
        land_mask (np.ndarray): Boolean mask (True = land), shape (224, 224).
        n_rows (int): Number of northern boundary rows to cosine-taper toward 0.
            Pass config.data.north_mask_rows; use 0 for models with no north masking.

    Returns:
        tuple: (front_mag, vorticity) — both 2-D arrays shape (224, 224).
            front_mag: |∇SST| (proxy for frontal strength).
            vorticity: ∂v/∂x − ∂u/∂y (relative vorticity, normalised).

    Example:
        >>> fmag, vort = compute_frontal_colocation(pred_dict, land_mask, n_rows=0)
    """
    sst = pred_dict.get('thetao', pred_dict.get('THETAO')).copy()
    uo  = pred_dict.get('uo',     pred_dict.get('UO')).copy()
    vo  = pred_dict.get('vo',     pred_dict.get('VO')).copy()

    # Compute gradients on the full field (north rows already filled by caller)
    front_mag = gradient_magnitude(sst)
    dv_dx     = np.gradient(vo, axis=1)
    du_dy     = np.gradient(uo, axis=0)
    vorticity = dv_dx - du_dy

    # Cosine-taper the gradient fields in the northern boundary rows toward 0:
    # np.gradient uses one-sided differences at the array boundary which can
    # leave edge artefacts even on a clean input field.  n_rows=0 skips this.
    if n_rows > 0:
        front_mag = apply_north_taper(front_mag, target=0.0, n_rows=n_rows)
        vorticity = apply_north_taper(vorticity, target=0.0, n_rows=n_rows)

    # Apply land mask last
    front_mag[land_mask] = np.nan
    vorticity[land_mask] = np.nan
    return front_mag, vorticity


def _fill_nan_with_mean(field):
    """Replace NaN pixels with the field mean to avoid zero-padding artifacts in FFT.

    Args:
        field (np.ndarray): 2-D array possibly containing NaN (land / masked rows).

    Returns:
        np.ndarray: Field with NaN replaced by nanmean(field).

    Example:
        >>> filled = _fill_nan_with_mean(sst)
    """
    out = field.copy()
    m = np.nanmean(out)
    out[~np.isfinite(out)] = m if np.isfinite(m) else 0.0
    return out


def compute_spectra_pair(pred_dict, gt_dict, var, dx_km):
    """Compute 1-D power spectra for a predicted and ground-truth field.

    NaN pixels (land, northern masked rows) are filled with the field mean before
    FFT to avoid artificial zero-padding discontinuities at coastlines.

    Args:
        pred_dict (dict): Predicted fields, variable → 2-D array.
        gt_dict (dict): Ground truth fields, variable → 2-D array.
        var (str): Variable key (e.g. 'thetao').
        dx_km (float): Isotropic grid spacing in km.

    Returns:
        tuple: (k, psd_pred, psd_gt) — wavenumber and PSD arrays.

    Example:
        >>> k, p_pred, p_gt = compute_spectra_pair(pred, gt, 'thetao', 10.0)
    """
    field_pred = _fill_nan_with_mean(pred_dict.get(var, pred_dict.get(var.upper())))
    field_gt   = _fill_nan_with_mean(gt_dict.get(var,  gt_dict.get(var.upper())))
    k,   psd_pred = power_spectrum_1d(field_pred, dx=dx_km)
    k_g, psd_gt   = power_spectrum_1d(field_gt,   dx=dx_km)
    return k, psd_pred, psd_gt


# ── figure ───────────────────────────────────────────────────────────────────

def make_figure(geo_data, frontal_data, spectra_data,
                lead_labels, lat2d, lon2d, width, land_mask,
                lat2d_front=None, lon2d_front=None, land_mask_front=None):
    """Assemble the 3-row × 2-column conservation laws figure.

    Args:
        geo_data (list): Two (geo_speed, pred_speed, r) tuples (one per lead column).
        frontal_data (list): Two (front_mag, vorticity, uo, vo) tuples per lead column.
        spectra_data (list): Two lists of (k, psd_pred, psd_gt, var_label) per lead.
        lead_labels (list[str]): Column labels, e.g. ['Lead day 3', 'Lead day 7'].
        lat2d (np.ndarray): 2-D latitude for geostrophic/spectra panels.
        lon2d (np.ndarray): 2-D longitude for geostrophic/spectra panels.
        width (float): Figure width in inches.
        land_mask (np.ndarray): Boolean land mask for geostrophic panels.
        lat2d_front (np.ndarray): 2-D latitude for frontal panels (224×224).
        lon2d_front (np.ndarray): 2-D longitude for frontal panels (224×224).
        land_mask_front (np.ndarray): Boolean land mask for frontal panels (224×224).

    Returns:
        plt.Figure: Completed figure.

    Example:
        >>> fig = make_figure(geo_data, frontal_data, spectra_data,
        ...                   ['Lead day 3', 'Lead day 7'], lat2d, lon2d, 8.0, land_mask)
    """
    # Fall back to main grid if frontal grid not supplied
    if lat2d_front is None:
        lat2d_front, lon2d_front, land_mask_front = lat2d, lon2d, land_mask
    fig, axes = plt.subplots(3, 2, figsize=(width, width * 1.4))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.08,
                        hspace=0.40, wspace=0.35)

    col_titles = lead_labels
    row_labels = [
        '(a) Geostrophic speed: GLORYS vs AFNO',
        '(b) Frontal co-location',
        '(c) Spatial power spectra',
    ]

    # ── Column titles ─────────────────────────────────────────────────────
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight='bold', pad=6)

    # ── Row 1: Geostrophic scatter ────────────────────────────────────────
    for col, (gs, ps, r) in enumerate(geo_data):
        ax = axes[0, col]
        ax.scatter(gs, ps, s=0.5, alpha=0.15, color='#1f77b4', rasterized=True)
        lim = max(gs.max(), ps.max()) * 1.05
        ax.plot([0, lim], [0, lim], 'k--', lw=0.8, label='1:1')
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel('GLORYS geostrophic speed (m s$^{-1}$)', fontsize=8)
        ax.set_ylabel('AFNO geostrophic speed (m s$^{-1}$)', fontsize=8)
        ax.text(0.05, 0.92, f'$r = {r:.3f}$', transform=ax.transAxes,
                fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=8)
        ax.set_aspect('equal')
        for sp in ax.spines.values():
            sp.set_visible(True)

    # ── Row 2: Frontal co-location ────────────────────────────────────────
    for col, (fmag, vort, uo, vo) in enumerate(frontal_data):
        ax = axes[1, col]
        # Smooth for display only — the 224×224 grid is too coarse for
        # contourf to look smooth without a mild Gaussian blur (sigma=1.5 px).
        fmag_s = gaussian_filter(np.where(land_mask_front, np.nan, fmag), sigma=0.1,
                                 mode='nearest')
        vmax_f = np.nanpercentile(fmag_s, 95)
        levels = np.linspace(0, vmax_f, 12)
        fmag_m = np.ma.masked_where(land_mask_front, fmag_s)
        im = ax.contourf(lon2d_front, lat2d_front, fmag_m,
                         levels=levels, cmap='YlOrRd', extend='max')

        # Gray land background
        from matplotlib.colors import ListedColormap as _LCM
        ax.pcolormesh(lon2d_front, lat2d_front,
                      np.where(land_mask_front, 1.0, np.nan),
                      cmap=_LCM(['lightgray']), shading='auto',
                      rasterized=True, zorder=2)

        # Quiver: subsample coarsely, mask land so no arrows drawn over it
        step = 6
        uo_m = np.ma.masked_where(land_mask_front, uo)
        vo_m = np.ma.masked_where(land_mask_front, vo)
        arrow_scale = 1.5  # multiply vectors to control apparent length
        qs = ax.quiver(
            lon2d_front[::step, ::step], lat2d_front[::step, ::step],
            uo_m[::step, ::step] * arrow_scale,
            vo_m[::step, ::step] * arrow_scale,
            color='black', alpha=0.85,
            scale=None,
            width=0.0022, headwidth=3.0, headlength=3.5, headaxislength=3.0,
            zorder=3,
        )
        ax.quiverkey(qs, X=0.08, Y=0.92, U=0.3 * arrow_scale,
                     label='0.3 m s$^{-1}$', labelpos='E',
                     fontproperties={'size': 6}, color='black', labelcolor='black')

        cb = plt.colorbar(im, ax=ax, label='|∇SST| (°C / grid cell)', pad=0.02,
                          fraction=0.046)
        cb.ax.tick_params(labelsize=7)
        cb.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax.set_xlabel('Longitude (°E)', fontsize=8)
        ax.set_ylabel('Latitude (°N)', fontsize=8)
        ax.tick_params(labelsize=8)
        for sp in ax.spines.values():
            sp.set_visible(True)

    # ── Row 3: Power spectra ──────────────────────────────────────────────
    colors = {'thetao': '#e6194b', 'zos': '#4363d8'}
    for col, spec_list in enumerate(spectra_data):
        ax = axes[2, col]
        glorys_colors = {'SST': '#b0b0b0', 'SSS': '#a8d5a2', 'SSH': '#78b4d0'}
        for k, psd_pred, psd_gt, var_label, color in spec_list:
            valid = (k > 0) & (psd_pred > 0) & (psd_gt > 0)
            g_color = glorys_colors.get(var_label, '#999999')
            kv, gv = k[valid], psd_gt[valid]
            step_m = max(1, len(kv) // 10)
            ax.loglog(kv, gv, '--', color=g_color, lw=1.4,
                      marker='o', markevery=step_m, markersize=4,
                      markerfacecolor='none', markeredgewidth=1.2,
                      label=f'GLORYS {var_label}')
            ax.loglog(k[valid], psd_pred[valid], '-', color=color, lw=1.6,
                      label=f'AFNO RT {var_label}')
        ax.set_xlabel('Wavenumber (cycles km$^{-1}$)', fontsize=8)
        ax.set_ylabel('PSD (units² km)', fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        ax.tick_params(labelsize=8)
        ax.grid(True, which='both', lw=0.3, color='#e0e0e0')
        for sp in ax.spines.values():
            sp.set_visible(True)

    # ── Row labels on left margin ─────────────────────────────────────────
    for row, label in enumerate(row_labels):
        axes[row, 0].annotate(
            label,
            xy=(-0.22, 0.5), xycoords='axes fraction',
            fontsize=8, fontweight='bold',
            ha='center', va='center', rotation=90,
        )

    return fig


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    """Run conservation law diagnostics and save figure.

    Example:
        python src/visualization/plot_conservation_laws.py \\
            --model_path results/models/AFNO_BoB_Surf_E11p1.pth \\
            --name AFNO_BoB_Surf_E11p1
    """
    parser = argparse.ArgumentParser(description='Conservation law diagnostics figure')
    parser.add_argument('--config_file',   default='afno_bob_surf_e12p1.yaml')
    parser.add_argument('--model_path',    default='results/models/AFNO_BoB_Surf_E11p1.pth')
    parser.add_argument('--name',          default='AFNO_BoB_Surf_E11p1')
    parser.add_argument('--eval_dates',    nargs='+',
                        default=[f'{d:02d}-{m:02d}-2020'
                                 for m, d in [(1,15),(2,15),(3,15),(4,15),
                                              (5,15),(6,15),(7,15),(8,15),
                                              (9,15),(10,15),(11,15),(12,15)]])
    parser.add_argument('--monsoon_date',  default='01-07-2020')
    parser.add_argument('--output',        default='results/figures/conservation_laws')
    parser.add_argument('--device',        default='cuda:0')
    parser.add_argument('--dpi',           type=int,   default=300)
    parser.add_argument('--width',         type=float, default=8.0)
    args = parser.parse_args()

    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 8,
        'axes.linewidth': 0.6, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    })

    print('=== Loading config and model ===')
    config, model, device = load_config_and_model(
        args.config_file, args.model_path, args.name, args.device
    )

    print('=== Loading data ===')
    from pathlib import Path as _Path
    ocean_data = xr.open_dataset(_Path(config.data.data_dir) / f'{config.data.file_prefix}.nc')
    atm_data   = xr.open_dataset(_Path(config.data.data_dir) / f'{config.data.file_prefix_atm}.nc')

    # Probe a single forecast to determine the actual postprocessed field shape
    probe_pred, _ = forecast_at_lead(config, model, device, ocean_data, atm_data,
                                     args.eval_dates[0], 1)
    probe_ssh = probe_pred.get('zos', probe_pred.get('ZOS'))
    n_lat, n_lon = probe_ssh.shape
    print(f'  Postprocessed field shape: {n_lat}×{n_lon}')

    # Build grid matching postprocessed resolution
    lat2d, lon2d, dy_m, dx_m = make_grid(n_lat, n_lon)
    f = coriolis(lat2d)
    dx_km = ((99.0 - 77.0) / (n_lon - 1)) * 111.0 * np.cos(np.deg2rad(np.mean(lat2d)))

    # Land mask at postprocessed resolution (for geostrophic scatter)
    ref_gt = load_ground_truth_batch(config, 9862, 1, ocean_data)
    sst_ref = ref_gt[1].get('thetao', ref_gt[1].get('THETAO'))
    land_mask = ~np.isfinite(sst_ref)

    # 224×224 grid and land mask for frontal co-location (raw model output space)
    import torch.nn.functional as _F
    lat2d_224, lon2d_224, _, _ = make_grid(224, 224)
    lm_t = torch.tensor(land_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    land_mask_224 = _F.interpolate(lm_t, size=(224, 224),
                                   mode='nearest').squeeze().numpy().astype(bool)

    lead_days   = [3, 7]
    lead_labels = ['Lead day 3', 'Lead day 7']
    geo_data    = []
    frontal_data = []
    spectra_data = []

    for lead in lead_days:
        print(f'\n=== Lead day {lead} ===')

        # ── Geostrophic scatter ───────────────────────────────────────────
        print('  Computing geostrophic balance scatter...')
        gs, ps, r = compute_geostrophic_scatter(
            config, model, device, ocean_data, atm_data,
            args.eval_dates, lead, lat2d, dx_m, dy_m, f
        )
        geo_data.append((gs, ps, r))
        print(f'  Pearson r (geo vs pred speed) = {r:.3f}')

        # ── Frontal co-location on raw 224×224 model output ───────────────
        print(f'  Computing frontal co-location for {args.monsoon_date}...')
        pred_m, _ = forecast_at_lead(config, model, device, ocean_data, atm_data,
                                     args.monsoon_date, lead, return_raw=True)
        # Cosine-taper the northern boundary rows only for models trained with
        # north masking (north_mask_rows > 0); skip for E14 and later.
        n_north_224 = round(
            getattr(config.data, 'north_mask_rows', 0) * 224 / 224
        )
        if n_north_224 > 0:
            for v in pred_m:
                bg = float(np.nanmean(pred_m[v][:-n_north_224, :])) \
                     if v in ('thetao', 'so') else 0.0
                pred_m[v] = apply_north_taper(pred_m[v], target=bg, n_rows=n_north_224)
        fmag, vort = compute_frontal_colocation(pred_m, land_mask_224,
                                                n_rows=n_north_224)
        uo_m = pred_m.get('uo', pred_m.get('UO')).copy()
        vo_m = pred_m.get('vo', pred_m.get('VO')).copy()
        uo_m[land_mask_224] = np.nan
        vo_m[land_mask_224] = np.nan
        frontal_data.append((fmag, vort, uo_m, vo_m))

        # ── Power spectra (average over all eval dates) ──────────────────
        print('  Computing power spectra (averaging over all eval dates)...')
        spec_accum = {var: {'psd_pred': [], 'psd_gt': [], 'k': None}
                      for var in ['thetao', 'zos']}
        for date_str in args.eval_dates:
            try:
                pred_s, gt_s = forecast_at_lead(config, model, device, ocean_data,
                                                atm_data, date_str, lead)
            except Exception as e:
                print(f'  Skipping spectra for {date_str}: {e}')
                continue
            for var in ['thetao', 'zos']:
                k, psd_pred, psd_gt = compute_spectra_pair(pred_s, gt_s, var, dx_km)
                spec_accum[var]['k'] = k
                spec_accum[var]['psd_pred'].append(psd_pred)
                spec_accum[var]['psd_gt'].append(psd_gt)

        spec_list = []
        for var, vlabel, color in [('thetao', 'SST', '#e6194b'),
                                    ('zos',    'SSH', '#4363d8')]:
            k = spec_accum[var]['k']
            psd_pred = np.mean(spec_accum[var]['psd_pred'], axis=0)
            psd_gt   = np.mean(spec_accum[var]['psd_gt'],   axis=0)
            spec_list.append((k, psd_pred, psd_gt, vlabel, color))
        spectra_data.append(spec_list)

    print('\n=== Building figure ===')
    fig = make_figure(geo_data, frontal_data, spectra_data,
                      lead_labels, lat2d, lon2d, args.width, land_mask,
                      lat2d_front=lat2d_224, lon2d_front=lon2d_224,
                      land_mask_front=land_mask_224)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.pdf'), dpi=args.dpi, bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=args.dpi, bbox_inches='tight')
    print(f'Saved: {out}.pdf / .png')
    plt.close(fig)


if __name__ == '__main__':
    main()

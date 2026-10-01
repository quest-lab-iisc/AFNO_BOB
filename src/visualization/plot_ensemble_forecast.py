"""Visualise TIGGE-driven ensemble ocean forecast: mean and spread.

Reads pre-computed ensemble predictions produced by run_ensemble_inference.py
and generates one A4 PNG per forecast lead day. Each figure shows the
ensemble mean and ensemble standard deviation for SST, SSS, current speed
(with quiver arrows on the mean panel), and SSH.

Postprocessing applied here mirrors run_inference.py / postprocess_ocean_variable():
    thetao, so : mean added back (reverse of training normalisation)
    uo, vo, zos: no normalisation reversal needed (raw model output)
    All fields  : bilinearly interpolated from 224×224 to GLORYS grid (229×265)

Inputs:
    --ensemble_dir (Path): Directory containing ensemble_predictions.npy,
        ensemble_mean.npy, ensemble_std.npy, ensemble_metadata.json.
    --mean_dir (Path): Directory with normalisation .npy stats files.
    --output_dir (Path): Where PNGs are written (default: {ensemble_dir}/plots).
    --quiver_stride (int): Sub-sample stride for UV quiver arrows (default 14).

Outputs:
    {output_dir}/ensemble_day{lt:02d}.png (PNG, 300 dpi, A4 portrait):
        One file per lead day (lt = 01 – 09).
        Layout: 4 rows × 2 columns — left=mean, right=std.
        Rows: SST | SSS | Current speed (+ quivers on mean) | SSH.

Example:
    conda activate BoB_Surf_2
    python src/visualization/plot_ensemble_forecast.py \\
        --ensemble_dir results/AFNO_BoB_Surf_E00/ensemble_Jan_14012020 \\
        --mean_dir data/1993_2020/mean
"""

import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    import cmocean
    CMAP_SST   = cmocean.cm.thermal
    CMAP_SSS   = cmocean.cm.haline
    CMAP_SPEED = cmocean.cm.speed
    CMAP_SSH   = cmocean.cm.matter
except ImportError:
    CMAP_SST   = plt.cm.RdYlBu_r
    CMAP_SSS   = plt.cm.viridis
    CMAP_SPEED = plt.cm.YlOrBr
    CMAP_SSH   = plt.cm.YlOrBr

CMAP_STD = plt.cm.Reds  # std is always non-negative

# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------

OCEAN_VARS = ['thetao', 'so', 'uo', 'vo', 'zos']
VAR_IDX    = {v: i for i, v in enumerate(OCEAN_VARS)}

# GLORYS ocean grid for Bay of Bengal
TARGET_H, TARGET_W = 229, 265
TARGET_SIZE = (TARGET_H, TARGET_W)

# Map extent [lon_min, lon_max, lat_min, lat_max]
EXTENT = [77, 99, 4, 23]

# Lat/lon arrays for quiver arrows (matching TARGET_SIZE, S→N to match data orientation)
_LATS = np.linspace(4, 23,  TARGET_H)
_LONS = np.linspace(77, 99, TARGET_W)
_LON_GRID, _LAT_GRID = np.meshgrid(_LONS, _LATS)

# A4 portrait in inches
A4_W, A4_H = 8.27, 11.69


# ---------------------------------------------------------------------------
# Postprocessing helpers
# ---------------------------------------------------------------------------

def _interp_to_target(arr_224: np.ndarray) -> np.ndarray:
    """Bilinearly interpolate a (224, 224) array to TARGET_SIZE.

    Args:
        arr_224 (np.ndarray): float32 array of shape (224, 224).

    Returns:
        np.ndarray: float32 array of shape (TARGET_H, TARGET_W).

    Example:
        >>> out = _interp_to_target(pred)
        >>> out.shape
        (229, 265)
    """
    t = torch.tensor(arr_224[np.newaxis, np.newaxis], dtype=torch.float32)
    t = F.interpolate(t, size=TARGET_SIZE, mode='bilinear', align_corners=False)
    return t.squeeze().numpy()


def postprocess_mean(arr_224: np.ndarray, var: str,
                     mean_dict: dict,
                     land_mask: np.ndarray) -> np.ndarray:
    """Reverse normalisation, interpolate to physical grid, and apply land mask.

    Args:
        arr_224 (np.ndarray): Normalised mean field from ensemble_mean.npy,
            shape (224, 224).
        var (str): Ocean variable name (one of OCEAN_VARS).
        mean_dict (dict): Normalisation mean arrays keyed by variable name.
        land_mask (np.ndarray): Boolean array of shape (TARGET_H, TARGET_W),
            True where pixel is land. Land pixels are set to NaN.

    Returns:
        np.ndarray: float32 array of shape (TARGET_H, TARGET_W) in physical
            units. Land pixels are NaN for all variables.

    Example:
        >>> sst_phys = postprocess_mean(mean_norm[0, 0], 'thetao', mean_dict, lmask)
    """
    out = _interp_to_target(arr_224.astype(np.float32))

    if var in ('thetao', 'so') and mean_dict.get(var) is not None:
        mean = np.squeeze(mean_dict[var]).astype(np.float32)  # (TARGET_H, TARGET_W)
        out = out + mean

    out[land_mask] = np.nan
    return out


def postprocess_std(arr_224: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    """Interpolate ensemble std to physical grid and apply land mask.

    Std is scale-invariant under mean subtraction, so only interpolation and
    masking are needed.

    Args:
        arr_224 (np.ndarray): Std field from ensemble_std.npy, shape (224, 224).
        land_mask (np.ndarray): Boolean array of shape (TARGET_H, TARGET_W),
            True where pixel is land.

    Returns:
        np.ndarray: float32 array of shape (TARGET_H, TARGET_W), NaN over land.

    Example:
        >>> sst_std = postprocess_std(std_norm[0, 0], lmask)
    """
    out = _interp_to_target(arr_224.astype(np.float32))
    out[land_mask] = np.nan
    return out


def load_norm_stats(mean_dir: Path) -> dict:
    """Load ocean normalisation mean arrays from .npy files.

    Args:
        mean_dir (Path): Directory containing mean_{var}_1993_2018_all_months.npy.

    Returns:
        dict: Variable name → numpy array (or None if file missing).

    Example:
        >>> mean_dict = load_norm_stats(Path('/data/mean'))
        >>> mean_dict['thetao'].shape
        (1, 229, 265)
    """
    mean_dict = {}
    for var in OCEAN_VARS:
        f = mean_dir / f"mean_{var}_1993_2018_all_months.npy"
        mean_dict[var] = np.load(f) if f.exists() else None
    return mean_dict


def compute_speed_fields(ensemble_preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute current speed mean and std from all ensemble member predictions.

    Args:
        ensemble_preds (np.ndarray): Full ensemble predictions of shape
            (N_members, N_steps, 5, 224, 224) in normalised space.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            speed_mean (N_steps, 224, 224): ensemble-mean speed (m/s).
            speed_std  (N_steps, 224, 224): ensemble std of speed (m/s).

    Example:
        >>> spd_mean, spd_std = compute_speed_fields(preds)
        >>> spd_mean.shape
        (9, 224, 224)
    """
    uo = ensemble_preds[:, :, VAR_IDX['uo']]  # (N_members, N_steps, 224, 224)
    vo = ensemble_preds[:, :, VAR_IDX['vo']]
    speed = np.sqrt(uo ** 2 + vo ** 2)        # (N_members, N_steps, 224, 224)
    return speed.mean(axis=0), speed.std(axis=0)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _add_panel(ax, data: np.ndarray, cmap, vmin: float, vmax: float,
               title: str, cbar_label: str) -> None:
    """Draw one map panel with a colorbar attached to the right.

    Args:
        ax: Matplotlib Axes object.
        data (np.ndarray): 2-D field of shape (TARGET_H, TARGET_W).
        cmap: Matplotlib colormap.
        vmin (float): Colorbar lower limit.
        vmax (float): Colorbar upper limit.
        title (str): Panel title.
        cbar_label (str): Colorbar label string.

    Returns:
        None: Draws into ax as a side effect.

    Example:
        >>> _add_panel(ax, sst_mean, CMAP_SST, 26, 30, 'SST mean', '°C')
    """
    im = ax.imshow(data, extent=EXTENT, origin='lower',
                   cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto',
                   interpolation='bilinear')
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xlabel('Longitude (°E)', fontsize=6, labelpad=2)
    ax.set_ylabel('Latitude (°N)', fontsize=6, labelpad=2)
    ax.tick_params(labelsize=6)

    # Lat/lon gridlines every 5°
    ax.set_xticks(np.arange(80, 100, 5))
    ax.set_yticks(np.arange(5, 24, 5))
    ax.grid(color='grey', linewidth=0.3, linestyle='--', alpha=0.5)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(cbar_label, fontsize=6)
    cb.ax.tick_params(labelsize=5)


def _add_speed_panel(ax, speed: np.ndarray, uo: np.ndarray, vo: np.ndarray,
                     cmap, vmin: float, vmax: float, title: str,
                     stride: int) -> None:
    """Draw current speed panel with UV quiver arrows overlaid.

    Args:
        ax: Matplotlib Axes object.
        speed (np.ndarray): Speed field (TARGET_H, TARGET_W).
        uo (np.ndarray): U-component field (TARGET_H, TARGET_W).
        vo (np.ndarray): V-component field (TARGET_H, TARGET_W).
        cmap: Matplotlib colormap.
        vmin (float): Colorbar lower limit (m/s).
        vmax (float): Colorbar upper limit (m/s).
        title (str): Panel title.
        stride (int): Sub-sampling stride for quiver arrows.

    Returns:
        None: Draws into ax as a side effect.

    Example:
        >>> _add_speed_panel(ax, spd, uo, vo, CMAP_SPEED, 0, 0.8, 'Speed', 14)
    """
    _add_panel(ax, speed, cmap, vmin, vmax, title, 'm/s')

    # Quiver — sub-sampled and normalised for direction only
    qs = stride
    lons_q = _LONS[::qs]
    lats_q = _LATS[::qs]
    U = uo[::qs, ::qs]
    V = vo[::qs, ::qs]
    mag = np.sqrt(U ** 2 + V ** 2)
    mag[mag == 0] = 1e-9
    ax.quiver(lons_q, lats_q, U / mag, V / mag,
              scale=28, width=0.003, color='black', alpha=0.6)


def _clim(data: np.ndarray, pct_lo: float = 2, pct_hi: float = 98,
          force_zero_lo: bool = False) -> tuple[float, float]:
    """Compute robust colormap limits from percentiles, ignoring NaN.

    Args:
        data (np.ndarray): Array of arbitrary shape.
        pct_lo (float): Lower percentile for vmin.
        pct_hi (float): Upper percentile for vmax.
        force_zero_lo (bool): If True, set vmin = 0 (for speed/std panels).

    Returns:
        tuple[float, float]: (vmin, vmax).

    Example:
        >>> vmin, vmax = _clim(sst_all_days)
    """
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        return 0.0, 1.0
    vmin = 0.0 if force_zero_lo else float(np.percentile(finite, pct_lo))
    vmax = float(np.percentile(finite, pct_hi))
    if vmin == vmax:
        vmax = vmin + 1e-6
    return vmin, vmax


# ---------------------------------------------------------------------------
# Per-day figure
# ---------------------------------------------------------------------------

def plot_day(lead_time: int, pred_date: str,
             sst_mean: np.ndarray, sst_std: np.ndarray,
             sss_mean: np.ndarray, sss_std: np.ndarray,
             spd_mean: np.ndarray, spd_std: np.ndarray,
             uo_mean:  np.ndarray, vo_mean:  np.ndarray,
             ssh_mean: np.ndarray, ssh_std: np.ndarray,
             clim: dict, quiver_stride: int,
             output_path: Path) -> None:
    """Create and save one A4 PNG for a given forecast lead day.

    Args:
        lead_time (int): Lead time in days (1–9).
        pred_date (str): Prediction date string, e.g. '15-01-2020'.
        sst_mean (np.ndarray): SST ensemble mean, shape (TARGET_H, TARGET_W), °C.
        sst_std  (np.ndarray): SST ensemble std,  shape (TARGET_H, TARGET_W), °C.
        sss_mean (np.ndarray): SSS ensemble mean, shape (TARGET_H, TARGET_W), psu.
        sss_std  (np.ndarray): SSS ensemble std,  shape (TARGET_H, TARGET_W), psu.
        spd_mean (np.ndarray): Speed ensemble mean, shape (TARGET_H, TARGET_W), m/s.
        spd_std  (np.ndarray): Speed ensemble std,  shape (TARGET_H, TARGET_W), m/s.
        uo_mean  (np.ndarray): U-component mean for quivers, (TARGET_H, TARGET_W).
        vo_mean  (np.ndarray): V-component mean for quivers, (TARGET_H, TARGET_W).
        ssh_mean (np.ndarray): SSH ensemble mean, shape (TARGET_H, TARGET_W), m.
        ssh_std  (np.ndarray): SSH ensemble std,  shape (TARGET_H, TARGET_W), m.
        clim (dict): Colormap limits keyed by variable and 'mean'/'std'.
        quiver_stride (int): UV quiver sub-sampling stride.
        output_path (Path): Where to save the PNG.

    Returns:
        None: Writes PNG as a side effect.

    Example:
        >>> plot_day(1, '15-01-2020', sst_m, sst_s, ..., clim, 14, Path('day01.png'))
    """
    fig = plt.figure(figsize=(A4_W, A4_H))
    fig.suptitle(
        f'Ensemble forecast  |  Lead time +{lead_time}d  |  Valid: {pred_date}',
        fontsize=10, fontweight='bold', y=0.995
    )

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.08, right=0.95,
                           top=0.965, bottom=0.04)

    axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(4)]

    # Row 0: SST
    _add_panel(axes[0][0], sst_mean, CMAP_SST,
               *clim['sst_mean'], 'SST  ensemble mean', '°C')
    _add_panel(axes[0][1], sst_std,  CMAP_STD,
               *clim['sst_std'],  'SST  ensemble std',  '°C')

    # Row 1: SSS
    _add_panel(axes[1][0], sss_mean, CMAP_SSS,
               *clim['sss_mean'], 'SSS  ensemble mean', 'psu')
    _add_panel(axes[1][1], sss_std,  CMAP_STD,
               *clim['sss_std'],  'SSS  ensemble std',  'psu')

    # Row 2: Current speed (mean with quivers, std without)
    _add_speed_panel(axes[2][0], spd_mean, uo_mean, vo_mean, CMAP_SPEED,
                     *clim['spd_mean'],
                     'Current speed  ensemble mean', quiver_stride)
    _add_panel(axes[2][1], spd_std, CMAP_STD,
               *clim['spd_std'], 'Current speed  ensemble std', 'm/s')

    # Row 3: SSH
    _add_panel(axes[3][0], ssh_mean, CMAP_SSH,
               *clim['ssh_mean'], 'SSH  ensemble mean', 'm')
    _add_panel(axes[3][1], ssh_std,  CMAP_STD,
               *clim['ssh_std'],  'SSH  ensemble std',  'm')

    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', format='png')
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Load ensemble outputs, postprocess, and save one PNG per lead day.

    Args:
        None: Reads --ensemble_dir, --mean_dir, --output_dir, --quiver_stride
            from sys.argv via argparse.

    Returns:
        None: Writes one PNG per lead day to output_dir.

    Example:
        >>> # python src/visualization/plot_ensemble_forecast.py \\
        >>> #     --ensemble_dir results/AFNO_BoB_Surf_E00/ensemble_Jan_14012020 \\
        >>> #     --mean_dir data/1993_2020/mean
    """
    parser = argparse.ArgumentParser(
        description='Plot TIGGE ensemble ocean forecast — mean and std per day'
    )
    parser.add_argument('--ensemble_dir', required=True,
                        help='Directory with ensemble_*.npy and metadata.json')
    parser.add_argument('--mean_dir', default='data/1993_2020/mean',
                        help='Directory with normalisation mean .npy files')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory for PNGs (default: ensemble_dir/plots)')
    parser.add_argument('--quiver_stride', type=int, default=14,
                        help='Quiver sub-sampling stride (default: 14)')
    args = parser.parse_args()

    ens_dir = Path(args.ensemble_dir)
    out_dir = Path(args.output_dir) if args.output_dir else ens_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load metadata ---
    meta = json.load(open(ens_dir / 'ensemble_metadata.json'))
    init_date   = meta['input_date']          # e.g. '14-01-2020'
    lead_times  = meta['lead_times_h']         # [24, 48, ..., 216]
    n_steps     = len(lead_times)

    print(f"Init date : {init_date}")
    print(f"Lead times: {lead_times}")
    print(f"Output dir: {out_dir}")

    # --- Load arrays ---
    print("\nLoading ensemble arrays...", flush=True)
    ens_mean  = np.load(ens_dir / 'ensemble_mean.npy')   # (N_steps, 5, 224, 224)
    ens_std   = np.load(ens_dir / 'ensemble_std.npy')    # (N_steps, 5, 224, 224)
    ens_preds = np.load(ens_dir / 'ensemble_predictions.npy')  # (50, N_steps, 5, 224, 224)

    # --- Load normalisation stats ---
    mean_dict = load_norm_stats(Path(args.mean_dir))

    # --- Land mask: NaN in the thetao climatological mean marks land pixels ---
    land_mask = np.isnan(np.squeeze(mean_dict['thetao']))  # (TARGET_H, TARGET_W)
    print(f"Land mask: {land_mask.sum()} land pixels out of {land_mask.size} total")

    # --- Compute speed statistics from full ensemble ---
    print("Computing speed statistics...", flush=True)
    spd_mean_norm, spd_std_norm = compute_speed_fields(ens_preds)  # (N_steps, 224, 224)

    # --- Postprocess all steps upfront ---
    print("Postprocessing fields...", flush=True)
    sst_mean_all = np.stack([postprocess_mean(ens_mean[s, VAR_IDX['thetao']],
                                              'thetao', mean_dict, land_mask) for s in range(n_steps)])
    sst_std_all  = np.stack([postprocess_std(ens_std[s, VAR_IDX['thetao']], land_mask)
                              for s in range(n_steps)])

    sss_mean_all = np.stack([postprocess_mean(ens_mean[s, VAR_IDX['so']],
                                              'so', mean_dict, land_mask) for s in range(n_steps)])
    sss_std_all  = np.stack([postprocess_std(ens_std[s, VAR_IDX['so']], land_mask)
                              for s in range(n_steps)])

    spd_mean_all = np.stack([postprocess_std(spd_mean_norm[s], land_mask) for s in range(n_steps)])
    spd_std_all  = np.stack([postprocess_std(spd_std_norm[s],  land_mask) for s in range(n_steps)])

    uo_mean_all  = np.stack([postprocess_mean(ens_mean[s, VAR_IDX['uo']],
                                              'uo', mean_dict, land_mask) for s in range(n_steps)])
    vo_mean_all  = np.stack([postprocess_mean(ens_mean[s, VAR_IDX['vo']],
                                              'vo', mean_dict, land_mask) for s in range(n_steps)])

    ssh_mean_all = np.stack([postprocess_mean(ens_mean[s, VAR_IDX['zos']],
                                              'zos', mean_dict, land_mask) for s in range(n_steps)])
    ssh_std_all  = np.stack([postprocess_std(ens_std[s, VAR_IDX['zos']], land_mask)
                              for s in range(n_steps)])

    # --- Compute consistent colormap limits across all days ---
    clim = {
        'sst_mean': _clim(sst_mean_all),
        'sst_std' : _clim(sst_std_all,  force_zero_lo=True),
        'sss_mean': _clim(sss_mean_all),
        'sss_std' : _clim(sss_std_all,  force_zero_lo=True),
        'spd_mean': _clim(spd_mean_all, force_zero_lo=True),
        'spd_std' : _clim(spd_std_all,  force_zero_lo=True),
        'ssh_mean': _clim(ssh_mean_all),
        'ssh_std' : _clim(ssh_std_all,  force_zero_lo=True),
    }

    print("\nColormap limits (consistent across all lead days):")
    for k, (vmin, vmax) in clim.items():
        print(f"  {k:12s}: [{vmin:.3f}, {vmax:.3f}]")

    # --- Plot one PNG per lead day ---
    print(f"\nPlotting {n_steps} figures...", flush=True)
    init_dt = datetime.strptime(init_date, '%d-%m-%Y')

    for s in range(n_steps):
        lead_time = s + 1
        pred_dt   = init_dt + timedelta(hours=int(lead_times[s]))
        pred_date = pred_dt.strftime('%d-%m-%Y')
        out_path  = out_dir / f'ensemble_day{lead_time:02d}.png'

        plot_day(
            lead_time   = lead_time,
            pred_date   = pred_date,
            sst_mean    = sst_mean_all[s],
            sst_std     = sst_std_all[s],
            sss_mean    = sss_mean_all[s],
            sss_std     = sss_std_all[s],
            spd_mean    = spd_mean_all[s],
            spd_std     = spd_std_all[s],
            uo_mean     = uo_mean_all[s],
            vo_mean     = vo_mean_all[s],
            ssh_mean    = ssh_mean_all[s],
            ssh_std     = ssh_std_all[s],
            clim        = clim,
            quiver_stride = args.quiver_stride,
            output_path = out_path,
        )

    print(f"\nDone. {n_steps} PNGs written to {out_dir}/")


if __name__ == '__main__':
    main()

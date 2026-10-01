"""Visualise ensemble forecast verification results against GLORYS reanalysis.

Reads the outputs of run_ensemble_verification.py and produces:

  1. skill_curves.png    — RMSE, MAE, CRPS and ensemble spread vs lead day,
                           one subplot per ocean variable (5-panel column).
  2. spread_skill.png    — Spread-skill ratio and Pearson correlation vs lead day,
                           one subplot per variable.
  3. error_maps.png      — Spatial map of time-mean absolute error for each variable,
                           averaged across all lead days (5-panel column, A4 portrait).
  4. bias_maps.png       — Spatial map of time-mean signed bias for each variable
                           (5-panel column, A4 portrait).

Inputs:
    --verification_dir (Path): Directory produced by run_ensemble_verification.py,
        containing verification_metrics.csv, error_fields.npy, bias_fields.npy.
    --mean_dir (Path): Directory with mean_{var} .npy files (for land mask).
    --output_dir (Path): Where PNGs are written (default: verification_dir).

Outputs:
    skill_curves.png   (300 dpi)
    spread_skill.png   (300 dpi)
    error_maps.png     (300 dpi, A4 portrait)
    bias_maps.png      (300 dpi, A4 portrait)

Example:
    conda activate BoB_Surf_2
    python src/visualization/plot_ensemble_verification.py \\
        --verification_dir results/AFNO_BoB_Surf_E00/ensemble_Jan_14012020/verification \\
        --mean_dir data/1993_2020/mean
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    import cmocean
    CMAP_ERR  = cmocean.cm.amp
    CMAP_BIAS = cmocean.cm.balance
except ImportError:
    CMAP_ERR  = plt.cm.YlOrRd
    CMAP_BIAS = plt.cm.RdBu_r


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OCEAN_VARS = ['thetao', 'so', 'uo', 'vo', 'zos']

VAR_LABELS = {
    'thetao': ('SST (thetao)', '°C'),
    'so'    : ('SSS (so)',     'psu'),
    'uo'    : ('U-current (uo)', 'm/s'),
    'vo'    : ('V-current (vo)', 'm/s'),
    'zos'   : ('SSH (zos)',    'm'),
}

EXTENT = [77, 99, 4, 23]   # [lon_min, lon_max, lat_min, lat_max]
A4_W, A4_H = 8.27, 11.69


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics(csv_path: Path) -> dict:
    """Load verification_metrics.csv into a nested dict indexed by [var][metric].

    Args:
        csv_path (Path): Path to verification_metrics.csv from run_ensemble_verification.py.

    Returns:
        dict: metrics[var][metric] = list of values (one per lead day, sorted).

    Example:
        >>> m = load_metrics(Path('verification/verification_metrics.csv'))
        >>> m['thetao']['rmse']
        [0.45, 0.52, ...]
    """
    raw = defaultdict(lambda: defaultdict(list))
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            var = row['variable']
            for col in ['lead_day', 'rmse', 'mae', 'bias', 'corr',
                        'spread', 'spread_skill', 'crps']:
                val = row[col]
                raw[var][col].append(float(val) if val else np.nan)

    # Sort by lead_day and convert to arrays
    metrics = {}
    for var in OCEAN_VARS:
        order = np.argsort(raw[var]['lead_day'])
        metrics[var] = {col: np.array(raw[var][col])[order]
                        for col in raw[var]}
    return metrics


def load_land_mask(mean_dir: Path) -> np.ndarray:
    """Derive land mask from the thetao climatological mean NaN pattern.

    Args:
        mean_dir (Path): Directory containing mean_thetao_1993_2018_all_months.npy.

    Returns:
        np.ndarray: Boolean array of shape (229, 265), True = land.

    Example:
        >>> lmask = load_land_mask(Path('data/1993_2020/mean'))
        >>> lmask.shape
        (229, 265)
    """
    f = mean_dir / 'mean_thetao_1993_2018_all_months.npy'
    return np.isnan(np.squeeze(np.load(f)))


# ---------------------------------------------------------------------------
# Plot 1: Skill curves
# ---------------------------------------------------------------------------

def plot_skill_curves(metrics: dict, init_date: str, output_path: Path) -> None:
    """Plot RMSE, MAE, CRPS and ensemble spread vs lead day for all variables.

    Args:
        metrics (dict): Output of load_metrics().
        init_date (str): Initialisation date string for the figure title.
        output_path (Path): Where to save the PNG.

    Returns:
        None: Writes PNG as side effect.

    Example:
        >>> plot_skill_curves(metrics, '14-01-2020', Path('skill_curves.png'))
    """
    fig, axes = plt.subplots(5, 1, figsize=(8, 14), sharex=True)
    fig.suptitle(f'Ensemble forecast skill vs GLORYS  |  Init: {init_date}',
                 fontsize=11, fontweight='bold')

    for ax, var in zip(axes, OCEAN_VARS):
        m   = metrics[var]
        lts = m['lead_day']
        label, unit = VAR_LABELS[var]

        ax.plot(lts, m['rmse'],   'b-o',  ms=4, lw=1.5, label='RMSE')
        ax.plot(lts, m['mae'],    'g--s', ms=4, lw=1.2, label='MAE')
        ax.plot(lts, m['crps'],   'r:^',  ms=4, lw=1.2, label='CRPS')
        ax.plot(lts, m['spread'], 'k-.d', ms=4, lw=1.2, label='Spread (std)')

        ax.set_ylabel(f'{label}\n({unit})', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=4, loc='upper left')
        ax.set_title(label, fontsize=9, pad=2)

    axes[-1].set_xlabel('Lead time (days)', fontsize=9)
    axes[-1].set_xticks(m['lead_day'])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ---------------------------------------------------------------------------
# Plot 2: Spread–skill ratio and correlation
# ---------------------------------------------------------------------------

def plot_spread_skill(metrics: dict, init_date: str, output_path: Path) -> None:
    """Plot spread-skill ratio and Pearson correlation vs lead day.

    A spread-skill ratio of 1 indicates a perfectly calibrated ensemble.
    Values below 1 mean the ensemble is overconfident (spread < RMSE).

    Args:
        metrics (dict): Output of load_metrics().
        init_date (str): Initialisation date string for the figure title.
        output_path (Path): Where to save the PNG.

    Returns:
        None: Writes PNG as side effect.

    Example:
        >>> plot_spread_skill(metrics, '14-01-2020', Path('spread_skill.png'))
    """
    fig, axes = plt.subplots(5, 2, figsize=(10, 14), sharex=True)
    fig.suptitle(f'Spread–skill ratio & correlation  |  Init: {init_date}',
                 fontsize=11, fontweight='bold')

    for row, var in enumerate(OCEAN_VARS):
        m   = metrics[var]
        lts = m['lead_day']
        label, _ = VAR_LABELS[var]

        # Left: spread-skill ratio
        ax = axes[row][0]
        ax.plot(lts, m['spread_skill'], 'b-o', ms=4, lw=1.5)
        ax.axhline(1.0, color='red', lw=1.0, ls='--', label='Perfect calibration')
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        if row == 0:
            ax.set_title('Spread-skill ratio\n(spread / RMSE)', fontsize=9)

        # Right: correlation
        ax = axes[row][1]
        ax.plot(lts, m['corr'], 'g-s', ms=4, lw=1.5)
        ax.axhline(0.0, color='grey', lw=0.7, ls=':')
        ax.set_ylim(-0.1, 1.05)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        if row == 0:
            ax.set_title('Pearson correlation\n(ensemble mean vs truth)', fontsize=9)

    for ax in axes[-1]:
        ax.set_xlabel('Lead time (days)', fontsize=9)
        ax.set_xticks(metrics[OCEAN_VARS[0]]['lead_day'])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ---------------------------------------------------------------------------
# Plot 3 & 4: Spatial maps
# ---------------------------------------------------------------------------

def _draw_map(ax, field: np.ndarray, title: str, unit: str,
              cmap, vmin: float, vmax: float) -> None:
    """Draw one spatial map panel with attached colorbar.

    Args:
        ax: Matplotlib Axes.
        field (np.ndarray): 2-D array of shape (229, 265).
        title (str): Panel title.
        unit (str): Colorbar unit label.
        cmap: Matplotlib colormap.
        vmin (float): Colorbar lower limit.
        vmax (float): Colorbar upper limit.

    Returns:
        None: Draws into ax as side effect.

    Example:
        >>> _draw_map(ax, err, 'SST |error|', '°C', cmap, 0, 1.5)
    """
    im = ax.imshow(field, extent=EXTENT, origin='lower',
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect='auto', interpolation='bilinear')
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xlabel('Longitude (°E)', fontsize=6, labelpad=2)
    ax.set_ylabel('Latitude (°N)',  fontsize=6, labelpad=2)
    ax.tick_params(labelsize=6)
    ax.set_xticks(np.arange(80, 100, 5))
    ax.set_yticks(np.arange(5, 24, 5))
    ax.grid(color='grey', lw=0.3, ls='--', alpha=0.4)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.05)
    cb  = plt.colorbar(im, cax=cax)
    cb.set_label(unit, fontsize=6)
    cb.ax.tick_params(labelsize=5)


def plot_spatial_maps(fields: np.ndarray, title_prefix: str, unit_map: dict,
                      cmap, init_date: str, output_path: Path,
                      symmetric: bool = False) -> None:
    """Plot time-mean spatial field for each ocean variable (5-panel A4 portrait).

    Args:
        fields (np.ndarray): Shape (n_steps, 5, 229, 265). Time-mean is computed
            inside this function over the n_steps axis.
        title_prefix (str): Prefix for each panel title, e.g. 'Mean |error|'.
        unit_map (dict): Variable name → unit string.
        cmap: Matplotlib colormap.
        init_date (str): Initialisation date string for figure suptitle.
        output_path (Path): Where to save the PNG.
        symmetric (bool): If True, use symmetric colormap limits (for bias maps).

    Returns:
        None: Writes PNG as side effect.

    Example:
        >>> plot_spatial_maps(error_fields, 'Mean |error|', units, CMAP_ERR,
        ...                   '14-01-2020', Path('error_maps.png'))
    """
    mean_fields = np.nanmean(fields, axis=0)   # (5, 229, 265)

    fig = plt.figure(figsize=(A4_W, A4_H))
    fig.suptitle(f'{title_prefix} — averaged over all lead days  |  Init: {init_date}',
                 fontsize=9, fontweight='bold', y=0.995)

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.50, wspace=0.38,
                           left=0.08, right=0.95, top=0.96, bottom=0.06)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]

    for (r, c), var in zip(positions, OCEAN_VARS):
        ax    = fig.add_subplot(gs[r, c])
        label, unit = VAR_LABELS[var]
        vi    = list(OCEAN_VARS).index(var)
        data  = mean_fields[vi]

        finite = data[np.isfinite(data)]
        if len(finite) == 0:
            continue
        if symmetric:
            vmax = float(np.percentile(np.abs(finite), 98))
            vmin = -vmax
        else:
            vmin = 0.0
            vmax = float(np.percentile(finite, 98))

        _draw_map(ax, data, f'{title_prefix} — {label}', unit, cmap, vmin, vmax)

    # Empty sixth slot — hide axes
    ax_empty = fig.add_subplot(gs[2, 1])
    ax_empty.axis('off')

    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Load verification outputs and produce all diagnostic plots.

    Args:
        None: Reads --verification_dir, --mean_dir, --output_dir from sys.argv.

    Returns:
        None: Writes skill_curves.png, spread_skill.png, error_maps.png,
            bias_maps.png to output_dir.

    Example:
        >>> # python src/visualization/plot_ensemble_verification.py \\
        >>> #     --verification_dir results/AFNO_BoB_Surf_E00/ensemble_Jan_14012020/verification \\
        >>> #     --mean_dir data/1993_2020/mean
    """
    parser = argparse.ArgumentParser(
        description='Plot ensemble forecast verification results'
    )
    parser.add_argument('--verification_dir', required=True,
                        help='Directory with verification_metrics.csv and *.npy files')
    parser.add_argument('--mean_dir', default='data/1993_2020/mean',
                        help='Directory with mean_*.npy normalisation files')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory for PNGs (default: verification_dir)')
    args = parser.parse_args()

    ver_dir = Path(args.verification_dir)
    out_dir = Path(args.output_dir) if args.output_dir else ver_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    metrics      = load_metrics(ver_dir / 'verification_metrics.csv')
    error_fields = np.load(ver_dir / 'error_fields.npy')   # (n_steps, 5, 229, 265)
    bias_fields  = np.load(ver_dir / 'bias_fields.npy')    # (n_steps, 5, 229, 265)

    # Extract init_date from CSV (first row)
    with open(ver_dir / 'verification_metrics.csv') as f:
        init_date = next(csv.DictReader(f))['init_date']

    print(f"Init date      : {init_date}")
    print(f"Lead steps     : {len(metrics[OCEAN_VARS[0]]['lead_day'])}")
    print(f"Output dir     : {out_dir}")
    print()

    unit_map = {var: VAR_LABELS[var][1] for var in OCEAN_VARS}

    # --- Plot 1: Skill curves ---
    print("Plotting skill curves ...")
    plot_skill_curves(metrics, init_date, out_dir / 'skill_curves.png')

    # --- Plot 2: Spread–skill ratio & correlation ---
    print("Plotting spread-skill ...")
    plot_spread_skill(metrics, init_date, out_dir / 'spread_skill.png')

    # --- Plot 3: Spatial error maps ---
    print("Plotting error maps ...")
    plot_spatial_maps(error_fields, 'Mean |error|', unit_map, CMAP_ERR,
                      init_date, out_dir / 'error_maps.png', symmetric=False)

    # --- Plot 4: Spatial bias maps ---
    print("Plotting bias maps ...")
    plot_spatial_maps(bias_fields, 'Mean bias', unit_map, CMAP_BIAS,
                      init_date, out_dir / 'bias_maps.png', symmetric=True)

    print(f"\nDone. All plots written to {out_dir}/")


if __name__ == '__main__':
    main()

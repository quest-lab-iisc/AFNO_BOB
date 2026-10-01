"""Plot all ocean variables for one ensemble member — diagnostic visualization.

Shows raw model output (normalised space, no mean-reversal) for all five
ocean variables across all forecast lead days. Useful for verifying grid
orientation, value ranges, and spatial structure before full postprocessing.

One A4 PNG per lead day. Layout: 3 rows × 2 columns (5 variables, one slot
used for a text summary). Grid is displayed south-at-bottom / north-at-top
(origin='lower', lat increases S→N in the GLORYS grid).

Inputs:
    --ensemble_dir (Path): Directory with ensemble_predictions.npy and
        ensemble_metadata.json (output of run_ensemble_inference.py).
    --member (int): 1-based ensemble member index to visualise (default: 1).
    --output_dir (Path): Where to write PNGs (default: ensemble_dir/member_plots).

Outputs:
    {output_dir}/member{mm:02d}_day{lt:02d}.png (PNG, 300 dpi, A4 portrait):
        One file per lead day showing thetao, so, uo, vo, zos in normalised
        (model output) space.

Example:
    conda activate BoB_Surf_2
    python src/visualization/plot_ensemble_member.py \\
        --ensemble_dir results/AFNO_BoB_Surf_E00/ensemble_Jan_14012020 \\
        --member 1
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OCEAN_VARS  = ['thetao', 'so', 'uo', 'vo', 'zos']
VAR_LABELS  = {
    'thetao': ('SST (thetao)', 'normalised °C'),
    'so'    : ('SSS (so)',     'normalised psu'),
    'uo'    : ('U-current (uo)', 'm/s'),
    'vo'    : ('V-current (vo)', 'm/s'),
    'zos'   : ('SSH (zos)',     'm'),
}

# Bay of Bengal extent — lat increases S→N (origin='lower')
EXTENT = [77, 99, 4, 23]   # [lon_min, lon_max, lat_min, lat_max]

A4_W, A4_H = 8.27, 11.69   # A4 portrait inches


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clim_symmetric(data: np.ndarray) -> tuple[float, float]:
    """Return symmetric ±vmax limits from 99th percentile of |data|.

    Args:
        data (np.ndarray): Array of any shape, may contain NaN.

    Returns:
        tuple[float, float]: (-vmax, +vmax).

    Example:
        >>> vmin, vmax = _clim_symmetric(arr)
    """
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        return -1.0, 1.0
    vmax = float(np.percentile(np.abs(finite), 99))
    return -vmax, vmax


def _clim_positive(data: np.ndarray) -> tuple[float, float]:
    """Return 1st–99th percentile limits for a non-negative field.

    Args:
        data (np.ndarray): Non-negative array, may contain NaN.

    Returns:
        tuple[float, float]: (vmin, vmax).

    Example:
        >>> vmin, vmax = _clim_positive(speed)
    """
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, 1)), float(np.percentile(finite, 99))


def _draw_panel(ax, data2d: np.ndarray, title: str, unit: str,
                vmin: float, vmax: float, cmap) -> None:
    """Draw one map panel with a colorbar.

    Grid origin is 'lower' so that index-0 (southernmost row, lat≈4°N)
    appears at the bottom of the panel and latitude increases upward.

    Args:
        ax: Matplotlib Axes.
        data2d (np.ndarray): 2-D field of shape (224, 224) or (H, W).
        title (str): Panel title.
        unit (str): Units label for the colorbar.
        vmin (float): Colorbar lower bound.
        vmax (float): Colorbar upper bound.
        cmap: Matplotlib colormap.

    Returns:
        None: Draws into ax as side effect.

    Example:
        >>> _draw_panel(ax, thetao_224, 'SST', '°C', -5, 5, plt.cm.RdBu_r)
    """
    im = ax.imshow(data2d, extent=EXTENT, origin='lower',
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect='auto', interpolation='bilinear')
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xlabel('Longitude (°E)', fontsize=6, labelpad=2)
    ax.set_ylabel('Latitude (°N)',  fontsize=6, labelpad=2)
    ax.tick_params(labelsize=6)
    ax.set_xticks(np.arange(80, 100, 5))
    ax.set_yticks(np.arange(5, 24, 5))
    ax.grid(color='grey', linewidth=0.3, linestyle='--', alpha=0.5)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.05)
    cb  = plt.colorbar(im, cax=cax)
    cb.set_label(unit, fontsize=6)
    cb.ax.tick_params(labelsize=5)


# ---------------------------------------------------------------------------
# Per-day figure
# ---------------------------------------------------------------------------

def plot_member_day(lead_time: int, pred_date: str, member: int,
                    fields: dict[str, np.ndarray],
                    clims: dict[str, tuple[float, float]],
                    cmaps: dict[str, object],
                    output_path: Path) -> None:
    """Create and save one A4 PNG for a single member and lead day.

    Args:
        lead_time (int): Lead time in days (1–9).
        pred_date (str): Prediction date, e.g. '15-01-2020'.
        member (int): 1-based ensemble member number.
        fields (dict): Variable name → 2-D np.ndarray (224, 224).
        clims (dict): Variable name → (vmin, vmax).
        cmaps (dict): Variable name → matplotlib colormap.
        output_path (Path): File path to write.

    Returns:
        None: Writes PNG as side effect.

    Example:
        >>> plot_member_day(1, '15-01-2020', 1, fields, clims, cmaps, path)
    """
    fig = plt.figure(figsize=(A4_W, A4_H))
    fig.suptitle(
        f'Ensemble member {member:02d}  |  '
        f'Lead +{lead_time}d  |  Valid: {pred_date}  |  '
        f'(model output, normalised space)',
        fontsize=9, fontweight='bold', y=0.995
    )

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.50, wspace=0.38,
                           left=0.08, right=0.95,
                           top=0.96, bottom=0.06)

    positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]

    for (r, c), var in zip(positions, OCEAN_VARS):
        ax    = fig.add_subplot(gs[r, c])
        label, unit = VAR_LABELS[var]
        vmin, vmax  = clims[var]
        _draw_panel(ax, fields[var], label, unit, vmin, vmax, cmaps[var])

    # Sixth slot: data summary text
    ax_txt = fig.add_subplot(gs[2, 1])
    ax_txt.axis('off')
    lines = [f'Member : {member:02d}', f'Lead   : +{lead_time}d ({lead_time*24}h)',
             f'Valid  : {pred_date}', '', 'Value ranges (raw model output):']
    for var in OCEAN_VARS:
        d = fields[var]
        lines.append(f'  {var:7s}: [{d.min():.2f}, {d.max():.2f}]')
    ax_txt.text(0.05, 0.95, '\n'.join(lines), transform=ax_txt.transAxes,
                fontsize=7, verticalalignment='top', fontfamily='monospace')

    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', format='png')
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Load ensemble predictions and plot all variables for one member.

    Args:
        None: Reads --ensemble_dir, --member, --output_dir from sys.argv.

    Returns:
        None: Writes one PNG per lead day to output_dir.

    Example:
        >>> # python src/visualization/plot_ensemble_member.py \\
        >>> #     --ensemble_dir results/AFNO_BoB_Surf_E00/ensemble_Jan_14012020 \\
        >>> #     --member 1
    """
    parser = argparse.ArgumentParser(
        description='Diagnostic plot: all ocean variables for one ensemble member'
    )
    parser.add_argument('--ensemble_dir', required=True,
                        help='Directory with ensemble_predictions.npy and metadata.json')
    parser.add_argument('--member', type=int, default=1,
                        help='1-based ensemble member index (default: 1)')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory (default: ensemble_dir/member_plots)')
    args = parser.parse_args()

    ens_dir  = Path(args.ensemble_dir)
    out_dir  = Path(args.output_dir) if args.output_dir else ens_dir / 'member_plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    member   = args.member

    # --- Metadata ---
    meta       = json.load(open(ens_dir / 'ensemble_metadata.json'))
    init_date  = meta['input_date']
    lead_times = meta['lead_times_h']
    n_steps    = len(lead_times)
    n_members  = meta['n_members']

    if member < 1 or member > n_members:
        print(f"ERROR: --member must be 1–{n_members}")
        sys.exit(1)

    print(f"Init date : {init_date}")
    print(f"Member    : {member}/{n_members}")
    print(f"Output dir: {out_dir}")

    # --- Load predictions for this member: shape (n_steps, 5, 224, 224) ---
    print("\nLoading ensemble_predictions.npy ...", flush=True)
    preds_all = np.load(ens_dir / 'ensemble_predictions.npy')   # (50, 9, 5, 224, 224)
    preds     = preds_all[member - 1]                            # (9, 5, 224, 224)

    print(f"Loaded member {member} predictions — shape: {preds.shape}")
    for vi, var in enumerate(OCEAN_VARS):
        d = preds[:, vi]
        print(f"  {var:7s}: min={d.min():.4f}  max={d.max():.4f}  mean={d.mean():.4f}")

    # --- Colormaps ---
    cmaps = {
        'thetao': plt.cm.RdBu_r,
        'so'    : plt.cm.RdBu_r,
        'uo'    : plt.cm.RdBu_r,
        'vo'    : plt.cm.RdBu_r,
        'zos'   : plt.cm.RdBu_r,
    }

    # --- Compute consistent colormap limits across all 9 days for this member ---
    clims = {}
    for vi, var in enumerate(OCEAN_VARS):
        clims[var] = _clim_symmetric(preds[:, vi])
    print("\nColormap limits (symmetric ±, 99th pct |val|, consistent across days):")
    for var, (lo, hi) in clims.items():
        print(f"  {var:7s}: [{lo:.3f}, {hi:.3f}]")

    # --- Plot one PNG per lead day ---
    print(f"\nPlotting {n_steps} figures ...", flush=True)
    init_dt = datetime.strptime(init_date, '%d-%m-%Y')

    for s in range(n_steps):
        lead_time = s + 1
        pred_dt   = init_dt + timedelta(hours=int(lead_times[s]))
        pred_date = pred_dt.strftime('%d-%m-%Y')
        out_path  = out_dir / f'member{member:02d}_day{lead_time:02d}.png'

        fields = {var: preds[s, vi] for vi, var in enumerate(OCEAN_VARS)}

        plot_member_day(
            lead_time   = lead_time,
            pred_date   = pred_date,
            member      = member,
            fields      = fields,
            clims       = clims,
            cmaps       = cmaps,
            output_path = out_path,
        )

    print(f"\nDone. {n_steps} PNGs written to {out_dir}/")


if __name__ == '__main__':
    main()

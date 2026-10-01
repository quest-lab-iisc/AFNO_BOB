"""Visualise ensemble spread and uncertainty in pre-processed TIGGE atmospheric forcing.

Loads all 50-member .npy files produced by prepare_tigge_ensemble.py and generates
three diagnostic figures that characterise how much atmospheric uncertainty the
TIGGE ensemble contains, and how it grows with forecast lead time.

Figure 1 — spread_profile.png
    Domain-mean ensemble standard deviation vs lead day (+1 d to +9 d) for all six
    atmospheric variables.  When multiple months are specified they are overlaid on
    the same axes with different colours, making seasonal differences immediately
    visible.

Figure 2 — spread_maps.png
    Spatial maps of ensemble spread (std across 50 members) at three lead times
    (+1 d, +5 d, +9 d) for every variable.  Rows = lead time, columns = variable.
    Reveals where in the Bay of Bengal the atmospheric forcing is most uncertain.

Figure 3 — spaghetti.png
    Per-member domain-averaged time series (one thin line per member) with the
    ensemble mean as a bold line and a ±1 σ shaded envelope.  Allows visual
    inspection of outlier members and the shape of the member distribution.

Inputs:
    --tigge_dir (Path): Directory containing Tigge_{month}_2020_ens_m{nn:02d}.npy
        files produced by prepare_tigge_ensemble.py (default: data/tigge_ensemble).
    --months (list[str]): One or more month labels matching the filename stem,
        e.g. Jan Oct (default: Jan).
    --n_members (int): Number of ensemble members per month (default: 50).
    --day_index (int): 0-based day index within the file axis-0 (default: 0).
    --output_dir (Path): Directory for output PNG files (default: results/plots).

Outputs:
    {output_dir}/tigge_spread_profile.png  — 300 dpi
    {output_dir}/tigge_spread_maps_{month}.png  — 300 dpi, one per month
    {output_dir}/tigge_spaghetti_{month}.png    — 300 dpi, one per month

Example:
    conda activate BoB_Surf_2
    python src/visualization/plot_tigge_uncertainty.py \\
        --months Jan Oct \\
        --tigge_dir data/tigge_ensemble \\
        --output_dir results/plots
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    import cmocean
    CMAP_SPREAD = cmocean.cm.amp
except ImportError:
    CMAP_SPREAD = plt.cm.YlOrRd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ATM_VARS = ['ssr', 'tp', 'u10', 'v10', 'msl', 'tcc']

VAR_LABELS = {
    'ssr': 'Solar radiation\n(norm.)',
    'tp' : 'Precipitation\n(norm.)',
    'u10': 'U-wind 10 m\n(m s⁻¹, mean-sub.)',
    'v10': 'V-wind 10 m\n(m s⁻¹, mean-sub.)',
    'msl': 'Sea-level pressure\n(norm.)',
    'tcc': 'Cloud cover\n(fraction)',
}

VAR_SHORT = {v: v for v in ATM_VARS}

LEAD_DAYS = list(range(1, 10))   # +1 d … +9 d

# Palette for months
MONTH_COLORS = {
    'Jan': '#1f77b4',
    'Oct': '#d62728',
    'Feb': '#ff7f0e',
    'Mar': '#2ca02c',
    'Apr': '#9467bd',
    'May': '#8c564b',
    'Jun': '#e377c2',
    'Jul': '#7f7f7f',
    'Aug': '#bcbd22',
    'Sep': '#17becf',
    'Nov': '#aec7e8',
    'Dec': '#ffbb78',
}

# Marker styles to distinguish months on the same line
MONTH_MARKERS = ['o', 's', '^', 'D', 'v', 'P']


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ensemble(tigge_dir: Path, month: str, n_members: int,
                  day_index: int) -> np.ndarray:
    """Load all member .npy files for one month and return a stacked array.

    Args:
        tigge_dir (Path): Directory containing the .npy member files.
        month (str): Month label, e.g. 'Jan'.
        n_members (int): Number of members to load.
        day_index (int): 0-based index along the init-day axis of each file.

    Returns:
        np.ndarray: float32 array of shape (n_members, 9, 6, 224, 224).

    Example:
        >>> data = load_ensemble(Path('data/tigge_ensemble'), 'Jan', 50, 0)
        >>> data.shape
        (50, 9, 6, 224, 224)
    """
    stem = f"Tigge_{month}_2020_ens"
    members = []
    for m in range(1, n_members + 1):
        path = tigge_dir / f"{stem}_m{m:02d}.npy"
        arr = np.load(path)              # (n_days, 9, 6, 224, 224)
        idx = min(day_index, arr.shape[0] - 1)
        members.append(arr[idx])         # (9, 6, 224, 224)
    return np.stack(members, axis=0).astype(np.float32)   # (N, 9, 6, 224, 224)


# ---------------------------------------------------------------------------
# Figure 1 — spread profile
# ---------------------------------------------------------------------------

def plot_spread_profile(month_data: dict, output_path: Path) -> None:
    """Plot domain-mean ensemble spread vs lead day for all atmospheric variables.

    Creates a 2×3 grid of subplots, one per variable.  Each curve shows the
    domain-mean standard deviation across ensemble members at each lead day.
    Multiple months are overlaid with different colours.

    Args:
        month_data (dict): Mapping of month label → (N, 9, 6, 224, 224) array.
        output_path (Path): Where to save the PNG.

    Returns:
        None: Writes a PNG file at output_path.

    Example:
        >>> plot_spread_profile({'Jan': jan_arr, 'Oct': oct_arr}, Path('spread.png'))
    """
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    axes_flat = axes.flatten()

    for vi, var in enumerate(ATM_VARS):
        ax = axes_flat[vi]
        for mi, (month, data) in enumerate(month_data.items()):
            # std across members: (N, 9, 6, H, W) → (9,) domain mean
            std_per_step = data[:, :, vi].std(axis=0).mean(axis=(-1, -2))  # (9,)
            color  = MONTH_COLORS.get(month, f'C{mi}')
            marker = MONTH_MARKERS[mi % len(MONTH_MARKERS)]
            ax.plot(LEAD_DAYS, std_per_step,
                    color=color, marker=marker, linewidth=2,
                    markersize=5, label=month)

        ax.set_title(var, fontsize=11, fontweight='bold')
        ax.set_xlabel('Lead time (days)')
        ax.set_ylabel(VAR_LABELS[var], fontsize=8)
        ax.set_xticks(LEAD_DAYS)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, 9.5)
        ax.set_ylim(bottom=0)
        if vi == 0:
            ax.legend(fontsize=9)

    fig.suptitle('TIGGE Ensemble Atmospheric Forcing — Domain-Mean Spread vs Lead Day',
                 fontsize=13, fontweight='bold')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 2 — spatial spread maps
# ---------------------------------------------------------------------------

def plot_spread_maps(data: np.ndarray, month: str, output_path: Path) -> None:
    """Plot spatial maps of ensemble spread at three representative lead times.

    Produces a grid of (3 lead times) × (6 variables) subplots, each showing
    the per-pixel standard deviation of the ensemble at that step.  Colour
    scale is consistent within each variable column.

    Args:
        data (np.ndarray): Ensemble array of shape (N, 9, 6, 224, 224).
        month (str): Month label for the figure title.
        output_path (Path): Where to save the PNG.

    Returns:
        None: Writes a PNG file at output_path.

    Example:
        >>> plot_spread_maps(data, 'Jan', Path('spread_maps_Jan.png'))
    """
    lead_indices = [0, 4, 8]    # +1d, +5d, +9d (0-based step indices)
    lead_labels  = ['+1 d', '+5 d', '+9 d']
    n_leads = len(lead_indices)
    n_vars  = len(ATM_VARS)

    fig, axes = plt.subplots(n_leads, n_vars,
                             figsize=(3.0 * n_vars, 2.8 * n_leads),
                             constrained_layout=True)

    # Compute std across members for every step: (9, 6, 224, 224)
    spread = data.std(axis=0)   # (9, 6, 224, 224)

    # Per-variable colour scale range (consistent across lead times)
    vmax_per_var = [spread[:, vi].max() for vi in range(n_vars)]

    for ri, (li, ll) in enumerate(zip(lead_indices, lead_labels)):
        for vi, var in enumerate(ATM_VARS):
            ax = axes[ri, vi]
            im = ax.imshow(
                spread[li, vi],
                origin='lower', cmap=CMAP_SPREAD,
                vmin=0, vmax=vmax_per_var[vi],
                aspect='auto',
            )
            ax.set_xticks([])
            ax.set_yticks([])

            if ri == 0:
                ax.set_title(var, fontsize=10, fontweight='bold')
            if vi == 0:
                ax.set_ylabel(ll, fontsize=9)

            # Colourbar on the right of the last column only
            if vi == n_vars - 1:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax)

    fig.suptitle(f'TIGGE Ensemble Spread — {month} 2020 (std across 50 members)',
                 fontsize=12, fontweight='bold')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 3 — spaghetti plots
# ---------------------------------------------------------------------------

def plot_spaghetti(data: np.ndarray, month: str, output_path: Path) -> None:
    """Plot per-member domain-averaged time series with ensemble mean and ±1σ envelope.

    Each member is shown as a thin semi-transparent line; the ensemble mean as
    a bold solid line; and the ±1σ band as a shaded area.  Provides an intuitive
    view of the spread distribution and any outlier members.

    Args:
        data (np.ndarray): Ensemble array of shape (N, 9, 6, 224, 224).
        month (str): Month label for the figure title.
        output_path (Path): Where to save the PNG.

    Returns:
        None: Writes a PNG file at output_path.

    Example:
        >>> plot_spaghetti(data, 'Jan', Path('spaghetti_Jan.png'))
    """
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    axes_flat = axes.flatten()

    # Domain-mean per member: (N, 9, 6, H, W) → (N, 9, 6)
    member_means = data.mean(axis=(-1, -2))   # (N, 9, 6)
    ens_mean = member_means.mean(axis=0)       # (9, 6)
    ens_std  = member_means.std(axis=0)        # (9, 6)
    n_members = data.shape[0]

    base_color = MONTH_COLORS.get(month, '#1f77b4')

    for vi, var in enumerate(ATM_VARS):
        ax = axes_flat[vi]

        # Individual members — thin grey lines
        for m in range(n_members):
            ax.plot(LEAD_DAYS, member_means[m, :, vi],
                    color='grey', alpha=0.15, linewidth=0.7)

        # ±1σ shading
        ax.fill_between(
            LEAD_DAYS,
            ens_mean[:, vi] - ens_std[:, vi],
            ens_mean[:, vi] + ens_std[:, vi],
            color=base_color, alpha=0.25, label='±1σ',
        )

        # Ensemble mean
        ax.plot(LEAD_DAYS, ens_mean[:, vi],
                color=base_color, linewidth=2.5, label='Ens. mean', zorder=3)

        ax.set_title(var, fontsize=11, fontweight='bold')
        ax.set_xlabel('Lead time (days)')
        ax.set_ylabel(VAR_LABELS[var], fontsize=8)
        ax.set_xticks(LEAD_DAYS)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, 9.5)
        if vi == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        f'TIGGE Ensemble — {month} 2020  |  '
        f'{n_members} members, domain-averaged  (grey = individual, bold = mean)',
        fontsize=11, fontweight='bold',
    )
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Parse CLI arguments and generate all three uncertainty figures.

    Args:
        None: All parameters from sys.argv via argparse.

    Returns:
        None: Writes PNG files to output_dir as side effects.

    Example:
        >>> # python src/visualization/plot_tigge_uncertainty.py \\
        >>> #     --months Jan Oct \\
        >>> #     --tigge_dir data/tigge_ensemble \\
        >>> #     --output_dir results/plots
    """
    parser = argparse.ArgumentParser(
        description="Visualise uncertainty in TIGGE atmospheric ensemble forcing"
    )
    parser.add_argument('--tigge_dir', type=str, default='data/tigge_ensemble',
                        help='Directory with Tigge_{month}_2020_ens_m*.npy files')
    parser.add_argument('--months', nargs='+', default=['Jan'],
                        help='Month labels to include, e.g. Jan Oct (default: Jan)')
    parser.add_argument('--n_members', type=int, default=50,
                        help='Number of ensemble members per month (default: 50)')
    parser.add_argument('--day_index', type=int, default=0,
                        help='0-based day index within each file (default: 0)')
    parser.add_argument('--output_dir', type=str, default='results/plots',
                        help='Directory for output PNG files (default: results/plots)')
    args = parser.parse_args()

    tigge_dir  = Path(args.tigge_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load data for all requested months
    # -----------------------------------------------------------------------
    month_data = {}
    for month in args.months:
        print(f"Loading {month} 2020 ({args.n_members} members) ...", flush=True)
        month_data[month] = load_ensemble(
            tigge_dir, month, args.n_members, args.day_index,
        )
        arr = month_data[month]
        print(f"  shape={arr.shape}  "
              f"mean={arr.mean():.3f}  std={arr.std():.3f}")

    # -----------------------------------------------------------------------
    # Figure 1 — spread profile (all months on same axes)
    # -----------------------------------------------------------------------
    month_tag = '_'.join(args.months)
    print("\nGenerating Figure 1: spread profile ...")
    plot_spread_profile(
        month_data,
        output_dir / f'tigge_spread_profile_{month_tag}.png',
    )

    # -----------------------------------------------------------------------
    # Figures 2 & 3 — one set per month
    # -----------------------------------------------------------------------
    for month, data in month_data.items():
        print(f"\nGenerating Figure 2: spatial spread maps ({month}) ...")
        plot_spread_maps(
            data, month,
            output_dir / f'tigge_spread_maps_{month}.png',
        )

        print(f"Generating Figure 3: spaghetti plots ({month}) ...")
        plot_spaghetti(
            data, month,
            output_dir / f'tigge_spaghetti_{month}.png',
        )

    print(f"\nDone. All figures written to {output_dir}/")


if __name__ == '__main__':
    main()

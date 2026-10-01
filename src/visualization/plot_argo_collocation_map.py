"""Plot a geographic map of Argo collocation positions from a collocations CSV.

Reads the ``collocations.csv`` produced by ``run_argo_det_comparison.py`` and
draws a scatter map over a cartopy PlateCarree projection with a land mask,
coastlines, and borders.  Each point is coloured by lead day (1–9).

The land colour (`#d3d3d3`) and coastline style match the project-wide
visualization convention used in ``plot_ensemble_season_comparison.py`` and
``plot_seasonal_forecast.py``.

Inputs:
    --csv (str): Path to collocations.csv (default: results/argo_det_test/collocations.csv).
    --output (str): Path to write the PNG (default: <csv_dir>/map.png).
    --dpi (int): Output resolution in DPI (default: 300).

Outputs:
    map.png (PNG): Scatter map saved to --output path.

Example:
    python src/visualization/plot_argo_collocation_map.py \\
        --csv results/argo_det_test/collocations.csv \\
        --output results/argo_det_test/map.png
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def load_pairs(csv_path: Path) -> list[dict]:
    """Read collocation records from CSV.

    Args:
        csv_path (Path): Path to collocations.csv.

    Returns:
        list[dict]: List of dicts with keys 'lat', 'lon', 'lead_day'.

    Example:
        >>> pairs = load_pairs(Path('results/argo_det_test/collocations.csv'))
    """
    pairs = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            pairs.append({
                'lat':      float(row['lat']),
                'lon':      float(row['lon']),
                'lead_day': int(row['lead_day']),
            })
    return pairs


def plot_map(pairs: list[dict], output_path: Path, dpi: int = 300) -> None:
    """Scatter map of Argo collocation positions coloured by lead day.

    Uses cartopy PlateCarree with land mask, coastlines, and borders styled
    to match project visualization conventions.

    Args:
        pairs (list[dict]): Collocation records with 'lat', 'lon', 'lead_day'.
        output_path (Path): Destination PNG path.
        dpi (int): Output resolution (default: 300).

    Returns:
        None: Writes PNG to output_path.

    Example:
        >>> plot_map(pairs, Path('results/argo_det_test/map.png'))
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
                    zorder=5)
    plt.colorbar(sc, ax=ax, label='Lead day', fraction=0.04, pad=0.04)
    ax.set_title(f'Argo collocation positions (n={len(pairs):,})', fontsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output_path}')


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.

    Example:
        >>> args = parse_args()
    """
    p = argparse.ArgumentParser(
        description='Plot Argo collocation map from collocations.csv.')
    p.add_argument('--csv',    default='results/argo_det_test/collocations.csv',
                   help='Path to collocations.csv')
    p.add_argument('--output', default=None,
                   help='Output PNG path (default: <csv_dir>/map.png)')
    p.add_argument('--dpi',    type=int, default=300,
                   help='Output resolution in DPI (default: 300)')
    return p.parse_args()


def main() -> None:
    """Entry point: load collocations and write map PNG.

    Example:
        python src/visualization/plot_argo_collocation_map.py \\
            --csv results/argo_det_test/collocations.csv
    """
    args = parse_args()
    csv_path = Path(args.csv)
    out_path = Path(args.output) if args.output else csv_path.parent / 'map.png'

    print(f'Reading collocations from {csv_path} …')
    pairs = load_pairs(csv_path)
    print(f'  {len(pairs):,} records loaded')

    plot_map(pairs, out_path, dpi=args.dpi)


if __name__ == '__main__':
    sys.exit(main())

"""Validate AFNO ocean forecasts against Argo float near-surface observations.

Loads pre-processed ARGO profile files from NCEI
(https://www.ncei.noaa.gov/data/oceans/argo/gadr/data/indian/YYYY/MM/)
for the 9-day lead window following a specified initialisation date, extracts
near-surface temperature and salinity observations within the Bay of Bengal
domain, collocates them with the corresponding model ensemble-mean prediction,
and computes bias and MAE as a function of lead day.

The model predictions are the ensemble_mean.npy files produced by
run_ensemble_inference.py (shape 9×5×224×224, normalised space).  They are
denormalised using the same pipeline as run_ensemble_verification.py.

Collocation method:
    Each Argo profile is matched to the nearest GLORYS 1/12° grid cell
    (lat 4–23°N, lon 77–99°E, 229×265).  The ensemble-mean prediction at
    that grid cell on the matching lead day is compared with the deepest
    near-surface observation available within 0–20 dbar.

Inputs:
    --ensemble_dir (Path): Directory containing ensemble_mean.npy produced by
        run_ensemble_inference.py (e.g.
        results/AFNO_BoB_Surf_E00/ensemble_Jan_14012020).
    --argo_dir (Path): Directory containing YYYYMMDD_prof.nc files
        (e.g. data/argo/jan2020).
    --init_date (str): Forecast initialisation date in dd-mm-yyyy format.
    --mean_dir (Path): Directory with normalisation mean .npy files
        (default: data/1993_2020/mean).
    --max_depth_dbar (float): Maximum pressure (dbar) to accept as
        "near-surface" (default: 20).
    --output_dir (Path): Where output PNG and CSV are written
        (default: same as ensemble_dir).

Outputs:
    argo_validation_map.png      — spatial map of Argo float positions and
                                   model thetao on the final lead day.
    argo_validation_scatter.png  — model vs Argo scatter for thetao and so.
    argo_validation_skill.png    — bias and MAE vs lead day for thetao and so.
    argo_collocations.csv        — collocated pairs: date, lat, lon, lead,
                                   model_thetao, argo_temp, model_so, argo_psal.

Example:
    conda activate BoB_Surf_2
    python src/inference/run_argo_validation.py \\
        --ensemble_dir results/AFNO_BoB_Surf_E00/ensemble_Jan_14012020 \\
        --argo_dir data/argo/jan2020 \\
        --init_date 14-01-2020 \\
        --output_dir results/AFNO_BoB_Surf_E00/ensemble_Jan_14012020/argo_validation
"""

import argparse
import csv
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr

sys.path.append(str(Path(__file__).parent.parent))
from inference.run_ensemble_verification import (
    load_norm_stats,
    interp_to_glorys,
    denormalize,
)

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OCEAN_VARS  = ['thetao', 'so', 'uo', 'vo', 'zos']
BOB_LAT_MIN, BOB_LAT_MAX = 4.0, 23.0
BOB_LON_MIN, BOB_LON_MAX = 77.0, 99.0

# GLORYS 1/12° grid (229 × 265)
GLORYS_LATS = np.linspace(4.0, 23.0, 229)
GLORYS_LONS = np.linspace(77.0, 99.0, 265)

ARGO_FILL = 99999.0
ARGO_QC_GOOD = b'1'


# ---------------------------------------------------------------------------
# ARGO loading
# ---------------------------------------------------------------------------

def load_argo_profiles(nc_path: Path, max_depth_dbar: float = 20.0) -> list[dict]:
    """Extract near-surface temperature and salinity profiles from one ARGO file.

    Filters to the Bay of Bengal domain, retains only profiles whose shallowest
    good observation is within max_depth_dbar, and records the near-surface
    TEMP and PSAL with QC flag == '1'.

    Args:
        nc_path (Path): Path to a YYYYMMDD_prof.nc file from NCEI ARGO GADR.
        max_depth_dbar (float): Maximum pressure (dbar) to accept as surface.

    Returns:
        list[dict]: One dict per valid profile with keys:
            lat, lon, date (str YYYYMMDD), pres (dbar), temp (°C), psal (psu).

    Example:
        >>> profiles = load_argo_profiles(Path('data/argo/jan2020/20200114_prof.nc'))
        >>> profiles[0].keys()
        dict_keys(['lat', 'lon', 'date', 'pres', 'temp', 'psal'])
    """
    date_str = nc_path.stem[:8]   # YYYYMMDD
    ds = xr.open_dataset(nc_path, mask_and_scale=False)

    lat = ds['LATITUDE'].values.astype(float)
    lon = ds['LONGITUDE'].values.astype(float)
    in_bob = (
        (lat >= BOB_LAT_MIN) & (lat <= BOB_LAT_MAX) &
        (lon >= BOB_LON_MIN) & (lon <= BOB_LON_MAX)
    )
    idx = np.where(in_bob)[0]

    pres     = ds['PRES'].values
    temp     = ds['TEMP'].values
    temp_qc  = ds['TEMP_QC'].values
    psal     = ds['PSAL'].values
    psal_qc  = ds['PSAL_QC'].values
    ds.close()

    records = []
    for i in idx:
        valid = np.where(
            (pres[i] != ARGO_FILL) & (pres[i] > 0) &
            (temp[i] != ARGO_FILL) & (temp_qc[i] == ARGO_QC_GOOD) &
            (psal[i] != ARGO_FILL) & (psal_qc[i] == ARGO_QC_GOOD)
        )[0]
        if len(valid) == 0:
            continue
        j0 = valid[0]    # shallowest good level
        if pres[i, j0] > max_depth_dbar:
            continue
        records.append({
            'lat'  : float(lat[i]),
            'lon'  : float(lon[i]),
            'date' : date_str,
            'pres' : float(pres[i, j0]),
            'temp' : float(temp[i, j0]),
            'psal' : float(psal[i, j0]),
        })
    return records


# ---------------------------------------------------------------------------
# Collocation
# ---------------------------------------------------------------------------

def glorys_index(lat: float, lon: float) -> tuple[int, int]:
    """Return the nearest GLORYS grid indices for a given (lat, lon).

    Args:
        lat (float): Latitude in decimal degrees.
        lon (float): Longitude in decimal degrees.

    Returns:
        tuple[int, int]: (lat_idx, lon_idx) 0-based indices into the
            229 × 265 GLORYS grid.

    Example:
        >>> glorys_index(14.1, 85.0)
        (121, 96)
    """
    lat_idx = int(np.round((lat - GLORYS_LATS[0]) / (GLORYS_LATS[1] - GLORYS_LATS[0])))
    lon_idx = int(np.round((lon - GLORYS_LONS[0]) / (GLORYS_LONS[1] - GLORYS_LONS[0])))
    lat_idx = np.clip(lat_idx, 0, len(GLORYS_LATS) - 1)
    lon_idx = np.clip(lon_idx, 0, len(GLORYS_LONS) - 1)
    return lat_idx, lon_idx


def collocate(argo_profiles: list[dict], ensemble_mean: np.ndarray,
              init_date: datetime, mean_dict: dict) -> list[dict]:
    """Match each Argo profile to the model prediction at the same location and lead day.

    Args:
        argo_profiles (list[dict]): Output of load_argo_profiles().
        ensemble_mean (np.ndarray): Shape (9, 5, 224, 224) normalised predictions.
        init_date (datetime): Forecast initialisation date.
        mean_dict (dict): Normalisation means from load_norm_stats().

    Returns:
        list[dict]: Collocated pairs, each with keys: lat, lon, date, lead_day,
            pres, argo_temp, argo_psal, model_thetao, model_so.

    Example:
        >>> pairs = collocate(profiles, em, datetime(2020, 1, 14), mean_dict)
        >>> pairs[0]['model_thetao']
        27.43
    """
    # Pre-compute denormalised thetao and so on the GLORYS grid for all 9 lead days
    # Shape: (9, 229, 265) for thetao and so
    pred_thetao = np.stack([
        denormalize(interp_to_glorys(ensemble_mean[ld, 0]), 'thetao', mean_dict)
        for ld in range(9)
    ])   # (9, 229, 265)
    pred_so = np.stack([
        denormalize(interp_to_glorys(ensemble_mean[ld, 1]), 'so', mean_dict)
        for ld in range(9)
    ])   # (9, 229, 265)

    pairs = []
    for rec in argo_profiles:
        obs_date = datetime.strptime(rec['date'], '%Y%m%d')
        lead = (obs_date - init_date).days   # 0 = init day, 1 = +1d, ...
        if lead < 1 or lead > 9:
            continue
        step = lead - 1    # 0-indexed into ensemble_mean axis 0

        li, lj = glorys_index(rec['lat'], rec['lon'])
        m_thetao = float(pred_thetao[step, li, lj])
        m_so     = float(pred_so[step, li, lj])

        if np.isnan(m_thetao) or np.isnan(m_so):
            continue  # land pixel in GLORYS

        pairs.append({
            'lat'          : rec['lat'],
            'lon'          : rec['lon'],
            'date'         : rec['date'],
            'lead_day'     : lead,
            'pres'         : rec['pres'],
            'argo_temp'    : rec['temp'],
            'argo_psal'    : rec['psal'],
            'model_thetao' : m_thetao,
            'model_so'     : m_so,
        })
    return pairs


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def lead_day_stats(pairs: list[dict]) -> dict:
    """Compute bias and MAE for thetao and so at each lead day.

    Args:
        pairs (list[dict]): Output of collocate().

    Returns:
        dict: Keys 'thetao' and 'so', each mapping to a dict with keys
            'lead', 'bias', 'mae', 'n' (all lists ordered by lead day 1–9).

    Example:
        >>> stats = lead_day_stats(pairs)
        >>> stats['thetao']['mae']
        [0.45, 0.82, 1.21, ...]
    """
    result = {}
    for var, obs_key in [('thetao', 'argo_temp'), ('so', 'argo_psal')]:
        model_key = f'model_{var}'
        by_lead = {}
        for p in pairs:
            ld = p['lead_day']
            by_lead.setdefault(ld, []).append(p[model_key] - p[obs_key])

        leads = sorted(by_lead)
        result[var] = {
            'lead': leads,
            'bias': [float(np.mean(by_lead[ld])) for ld in leads],
            'mae' : [float(np.mean(np.abs(by_lead[ld]))) for ld in leads],
            'n'   : [len(by_lead[ld]) for ld in leads],
        }
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_map(pairs: list[dict], month_label: str, output_path: Path) -> None:
    """Plot Argo float positions on a plain background.

    Args:
        pairs (list[dict]): Collocated profile pairs.
        month_label (str): e.g. 'January 2020', used in the title.
        output_path (Path): Where to save the PNG.

    Returns:
        None: Writes a PNG file at output_path.

    Example:
        >>> plot_map(pairs, 'January 2020', Path('map.png'))
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    lats = [p['lat'] for p in pairs]
    lons = [p['lon'] for p in pairs]
    leads = [p['lead_day'] for p in pairs]
    sc = ax.scatter(lons, lats, c=leads, cmap='viridis', s=60,
                    edgecolors='k', linewidths=0.5, zorder=5,
                    vmin=1, vmax=9, label='Argo floats')
    plt.colorbar(sc, ax=ax, label='Lead day', fraction=0.046, pad=0.04)

    ax.set_xlim(BOB_LON_MIN, BOB_LON_MAX)
    ax.set_ylim(BOB_LAT_MIN, BOB_LAT_MAX)
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title(f'Argo float positions — {month_label}\nDots coloured by lead day',
                 fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_scatter(pairs: list[dict], month_label: str, output_path: Path) -> None:
    """Plot model vs Argo scatter for thetao and so.

    Args:
        pairs (list[dict]): Collocated profile pairs.
        month_label (str): Used in the title.
        output_path (Path): Where to save the PNG.

    Returns:
        None: Writes a PNG file at output_path.

    Example:
        >>> plot_scatter(pairs, 'January 2020', Path('scatter.png'))
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    for ax, (var, obs_key, model_key, title, xlabel, ylabel, xlim, ylim) in zip(
        axes,
        [
            ('thetao', 'argo_temp', 'model_thetao', 'SST',
             'Argo SST (°C)', 'Model SST (°C)', (23, 31), (23, 31)),
            ('so', 'argo_psal', 'model_so', 'SSS',
             'Argo SSS (psu)', 'Model SSS (psu)', (27, 36), (27, 36)),
        ]
    ):
        obs    = np.array([p[obs_key]   for p in pairs])
        model  = np.array([p[model_key] for p in pairs])
        leads  = np.array([p['lead_day'] for p in pairs])

        sc = ax.scatter(obs, model, c=leads, cmap='viridis',
                        s=50, edgecolors='k', linewidths=0.5,
                        vmin=1, vmax=9, alpha=0.85)
        plt.colorbar(sc, ax=ax, label='Lead day')

        lo = min(xlim[0], ylim[0])
        hi = max(xlim[1], ylim[1])
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='1:1 line')

        bias = float(np.mean(model - obs))
        mae  = float(np.mean(np.abs(model - obs)))
        rmse = float(np.sqrt(np.mean((model - obs) ** 2)))
        r    = float(np.corrcoef(obs, model)[0, 1])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f'{title}  (n={len(obs)})\n'
            f'Bias={bias:+.2f}  MAE={mae:.2f}  RMSE={rmse:.2f}  r={r:.3f}',
            fontsize=10
        )

    fig.suptitle(f'Model vs Argo — {month_label}', fontsize=12, fontweight='bold')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_skill(stats: dict, month_label: str, output_path: Path) -> None:
    """Plot bias and MAE vs lead day for thetao and so.

    Args:
        stats (dict): Output of lead_day_stats().
        month_label (str): Used in the title.
        output_path (Path): Where to save the PNG.

    Returns:
        None: Writes a PNG file at output_path.

    Example:
        >>> plot_skill(stats, 'January 2020', Path('skill.png'))
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = {'thetao': '#1f77b4', 'so': '#d62728'}
    labels = {'thetao': 'SST (°C)', 'so': 'SSS (psu)'}

    for ax_idx, metric in enumerate(['bias', 'mae']):
        ax = axes[ax_idx]
        for var in ['thetao', 'so']:
            s = stats[var]
            ns = s['n']
            vals = s[metric]
            ax.plot(s['lead'], vals, marker='o', linewidth=2, markersize=7,
                    color=colors[var], label=labels[var])
            # Annotate with profile count
            for x, y, n in zip(s['lead'], vals, ns):
                ax.annotate(f'n={n}', (x, y), textcoords='offset points',
                            xytext=(4, 4), fontsize=6, color=colors[var])

        ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
        ax.set_xlabel('Lead day')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{"Bias (model − Argo)" if metric == "bias" else "MAE"}',
                     fontsize=10)
        ax.set_xticks(range(1, 10))
        ax.set_xlim(0.5, 9.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(f'Model skill vs Argo floats — {month_label}',
                 fontsize=12, fontweight='bold')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Parse CLI arguments and run the full Argo collocation and validation pipeline.

    Args:
        None: All parameters read from sys.argv via argparse.

    Returns:
        None: Writes PNG plots and a CSV of collocated pairs as side effects.

    Example:
        >>> # python src/inference/run_argo_validation.py \\
        >>> #     --ensemble_dir results/AFNO_BoB_Surf_E00/ensemble_Jan_14012020 \\
        >>> #     --argo_dir data/argo/jan2020 \\
        >>> #     --init_date 14-01-2020
    """
    parser = argparse.ArgumentParser(
        description='Validate AFNO ensemble-mean forecast against Argo near-surface observations'
    )
    parser.add_argument('--ensemble_dir', required=True,
                        help='Directory with ensemble_mean.npy (from run_ensemble_inference.py)')
    parser.add_argument('--argo_dir', required=True,
                        help='Directory with YYYYMMDD_prof.nc Argo files')
    parser.add_argument('--init_date', required=True,
                        help='Forecast init date in dd-mm-yyyy format')
    parser.add_argument('--mean_dir', default='data/1993_2020/mean',
                        help='Directory with normalisation mean .npy files')
    parser.add_argument('--max_depth_dbar', type=float, default=20.0,
                        help='Max pressure (dbar) for near-surface obs (default: 20)')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory for plots and CSV (default: ensemble_dir/argo_validation)')
    args = parser.parse_args()

    ensemble_dir = Path(args.ensemble_dir)
    argo_dir     = Path(args.argo_dir)
    output_dir   = Path(args.output_dir) if args.output_dir else ensemble_dir / 'argo_validation'
    output_dir.mkdir(parents=True, exist_ok=True)

    init_date = datetime.strptime(args.init_date, '%d-%m-%Y')
    month_label = init_date.strftime('%B %Y')

    # --- Load model predictions ---
    print('Loading ensemble mean ...')
    em_path = ensemble_dir / 'ensemble_mean.npy'
    if not em_path.exists():
        print(f'ERROR: {em_path} not found.  Run run_ensemble_inference.py first.')
        sys.exit(1)
    ensemble_mean = np.load(em_path)   # (9, 5, 224, 224)

    # --- Load normalisation stats ---
    print('Loading normalisation stats ...')
    mean_dict = load_norm_stats(Path(args.mean_dir))

    # --- Load Argo profiles ---
    print(f'Loading Argo profiles from {argo_dir} ...')
    all_profiles = []
    for nc_file in sorted(argo_dir.glob('*_prof.nc')):
        recs = load_argo_profiles(nc_file, max_depth_dbar=args.max_depth_dbar)
        all_profiles.extend(recs)
    print(f'  Total BoB near-surface profiles: {len(all_profiles)}')

    if not all_profiles:
        print('No Argo profiles found. Check --argo_dir and domain bounds.')
        sys.exit(1)

    # --- Collocate ---
    print('Collocating with model predictions ...')
    pairs = collocate(all_profiles, ensemble_mean, init_date, mean_dict)
    print(f'  Collocated pairs (ocean grid point, not land): {len(pairs)}')

    if not pairs:
        print('No valid collocations. Check lead-day range and land mask.')
        sys.exit(1)

    # --- Save CSV ---
    csv_path = output_dir / 'argo_collocations.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'date', 'lead_day', 'lat', 'lon', 'pres',
            'argo_temp', 'model_thetao',
            'argo_psal', 'model_so',
        ])
        writer.writeheader()
        for p in pairs:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v
                             for k, v in p.items()})
    print(f'  Saved: {csv_path}')

    # --- Statistics ---
    stats = lead_day_stats(pairs)
    print(f'\n=== Argo validation summary — {month_label} ===')
    print(f'{"Lead":>5}  {"n":>4}  {"T_bias":>8}  {"T_MAE":>7}  {"S_bias":>8}  {"S_MAE":>7}')
    for ld, tb, tm, tn, sb, sm, sn in zip(
        stats['thetao']['lead'], stats['thetao']['bias'], stats['thetao']['mae'],
        stats['thetao']['n'],
        stats['so']['bias'], stats['so']['mae'], stats['so']['n'],
    ):
        print(f'+{ld:>3}d  {tn:>4}  {tb:>+8.3f}  {tm:>7.3f}  {sb:>+8.3f}  {sm:>7.3f}')

    # --- Plots ---
    print('\nGenerating plots ...')

    plot_map(pairs, month_label,
             output_dir / 'argo_validation_map.png')

    plot_scatter(pairs, month_label,
                 output_dir / 'argo_validation_scatter.png')

    plot_skill(stats, month_label,
               output_dir / 'argo_validation_skill.png')

    print(f'\nAll outputs written to {output_dir}/')


if __name__ == '__main__':
    main()

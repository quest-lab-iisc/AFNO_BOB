"""Ensemble spread-skill and CRPS validation against GLORYS, OSTIA, and Argo.

Produces a 2-row × 2-column publication figure comparing ensemble reliability
for two Bay-of-Bengal seasons (Winter = January IC, Post-monsoon = October IC):

  Row 1 — Spread-skill ratio (spread / RMSE) vs lead day.
           Ideal value is 1.0 (horizontal dashed line).
           Values < 1 indicate an overconfident ensemble.
  Row 2 — CRPS vs lead day.

Each panel overlays two truth sources:
  Solid line   — GLORYS reanalysis (the training truth)
  Dashed line  — OSTIA L4 analysis (independent operational SST product)

If Argo CSV files are provided (from run_argo_det_comparison.py), per-season
Argo-validated RMSE is overlaid as filled circles scaled by number of Argo
collocations.

Inputs:
    --glorys_jan_dir (str): verification dir for Winter GLORYS ensemble verification.
    --glorys_oct_dir (str): verification dir for Post-monsoon GLORYS ensemble verification.
    --ostia_jan_dir  (str): verification dir for Winter OSTIA ensemble verification.
    --ostia_oct_dir  (str): verification dir for Post-monsoon OSTIA ensemble verification.
    --argo_csv       (str): Optional path to Argo summary_by_lead.csv from
                            run_argo_det_comparison.py.
    --var            (str): Ocean variable to plot (default: thetao = SST).
    --output         (str): Output base path without extension.
    --pdf            (flag): Also write a vector PDF alongside the PNG.
    --dpi            (int): Output raster DPI (default 300).

Outputs:
    {output}.png  — 300 dpi raster figure.
    {output}.pdf  — vector PDF (only with --pdf).

Example:
    conda activate BoB_Surf_2
    python src/visualization/plot_ensemble_skill.py \\
        --glorys_jan_dir results/AFNO_BoB_Surf_E14/ensemble_verification_jan \\
        --glorys_oct_dir results/AFNO_BoB_Surf_E14/ensemble_verification_oct \\
        --ostia_jan_dir  results/AFNO_BoB_Surf_E14/ensemble_ic_era5_15-01-2020/verification_ostia \\
        --ostia_oct_dir  results/AFNO_BoB_Surf_E14/ensemble_ic_era5_15-10-2020/verification_ostia \\
        --argo_csv       results/AFNO_BoB_Surf_E14/argo_comparison/summary_by_lead.csv \\
        --output results/figures/fig9_ensemble_skill
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

_COL_WIN     = '#1565C0'   # deep blue — Winter
_COL_OCT     = '#E65100'   # deep orange — Post-monsoon
_LW          = 2.0
_LS_GLORYS   = '-'
_LS_OSTIA    = '--'

_VAR_LABELS  = {'thetao': 'SST', 'so': 'SSS', 'uo': 'U', 'vo': 'V', 'zos': 'SSH'}
_VAR_UNITS   = {'thetao': '°C',  'so': 'psu', 'uo': 'm s⁻¹', 'vo': 'm s⁻¹', 'zos': 'm'}

GLORYS_CSV   = 'verification_metrics.csv'
OSTIA_CSV    = 'verification_metrics_ostia.csv'


def _load_csv(csv_path: Path, var: str) -> pd.DataFrame:
    """Load a verification_metrics CSV and filter to one variable.

    Args:
        csv_path (Path): Path to the CSV file.
        var (str): Ocean variable name (e.g. 'thetao').

    Returns:
        pd.DataFrame: Rows for the requested variable, sorted by lead_day.
            Returns an empty DataFrame if the file does not exist.

    Example:
        >>> df = _load_csv(Path('verification_metrics.csv'), 'thetao')
        >>> df['spread_skill'].values
    """
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    return df[df['variable'] == var].sort_values('lead_day').reset_index(drop=True)


def _safe(df: pd.DataFrame, col: str) -> np.ndarray:
    """Extract a column from a DataFrame, returning NaN array if missing.

    Args:
        df (pd.DataFrame): Source dataframe.
        col (str): Column name.

    Returns:
        np.ndarray: Array of values, or array of NaN if df is empty.

    Example:
        >>> vals = _safe(df, 'spread_skill')
    """
    if df.empty or col not in df.columns:
        return np.full(9, np.nan)
    return df[col].values.astype(float)


def _parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.

    Example:
        >>> args = _parse_args()
    """
    p = argparse.ArgumentParser(
        description='Ensemble spread-skill and CRPS: GLORYS vs OSTIA vs Argo'
    )
    p.add_argument('--glorys_jan_dir', required=True)
    p.add_argument('--glorys_oct_dir', required=True)
    p.add_argument('--ostia_jan_dir',  required=True)
    p.add_argument('--ostia_oct_dir',  required=True)
    p.add_argument('--glorys_apr_dir', default=None)
    p.add_argument('--glorys_jun_dir', default=None)
    p.add_argument('--ostia_apr_dir',  default=None)
    p.add_argument('--ostia_jun_dir',  default=None)
    p.add_argument('--argo_csv',       default=None,
                   help='summary_by_lead.csv from run_argo_det_comparison.py')
    p.add_argument('--var',            default='thetao')
    p.add_argument('--output',         default='results/figures/fig9_ensemble_skill')
    p.add_argument('--pdf',            action='store_true')
    p.add_argument('--dpi',            type=int, default=300)
    return p.parse_args()


def main():
    """Load verification CSVs, build and save the ensemble skill figure.

    Args:
        None: All settings read from sys.argv (see module docstring).

    Returns:
        None: Writes PNG (and optionally PDF) to the output path.

    Example:
        >>> # python src/visualization/plot_ensemble_skill.py \\
        >>> #     --glorys_jan_dir ... --glorys_oct_dir ... \\
        >>> #     --ostia_jan_dir  ... --ostia_oct_dir  ...
    """
    args = _parse_args()
    var  = args.var
    vlab = _VAR_LABELS.get(var, var)
    vunit = _VAR_UNITS.get(var, '')

    # --- Load verification CSVs ---
    g_jan = _load_csv(Path(args.glorys_jan_dir) / GLORYS_CSV, var)
    g_oct = _load_csv(Path(args.glorys_oct_dir) / GLORYS_CSV, var)
    o_jan = _load_csv(Path(args.ostia_jan_dir)  / OSTIA_CSV,  var)
    o_oct = _load_csv(Path(args.ostia_oct_dir)  / OSTIA_CSV,  var)

    leads = np.arange(1, 10)   # lead days 1–9

    # --- Optional Argo overlay ---
    argo_df = None
    if args.argo_csv and Path(args.argo_csv).exists():
        argo_df = pd.read_csv(args.argo_csv)
        # Expected columns: lead_day, sst_rmse, n_collocations (or similar)
        if 'lead_day' not in argo_df.columns:
            argo_df = None

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    plt.rcParams.update({'font.size': 8})

    row_labels  = [f'Spread-Skill Ratio  (spread / RMSE)',
                   f'RMSE  ({vunit})']
    col_labels  = ['Winter (IC: 15 Jan 2020)',
                   'Post-monsoon (IC: 15 Oct 2020)']
    # Row 0: spread-skill (both GLORYS and OSTIA have this).
    # Row 1: RMSE (both have this; CRPS is NaN for OSTIA so not used).
    row_keys    = [('spread_skill', 'spread_skill'),
                   ('rmse',         'rmse')]
    data_pairs  = [(g_jan, o_jan), (g_oct, o_oct)]

    for ci, (g_df, o_df) in enumerate(data_pairs):
        for ri, (gkey, okey) in enumerate(row_keys):
            ax = axes[ri, ci]
            color = _COL_WIN if ci == 0 else _COL_OCT

            g_vals = _safe(g_df, gkey)
            o_vals = _safe(o_df, okey)

            ax.plot(leads, g_vals, color=color, lw=_LW, ls=_LS_GLORYS,
                    label='vs GLORYS' if (ri == 0 and ci == 0) else None)
            ax.plot(leads, o_vals, color=color, lw=_LW, ls=_LS_OSTIA,
                    label='vs OSTIA'  if (ri == 0 and ci == 0) else None)

            # Spread-skill: ideal = 1 reference line
            if ri == 0:
                ax.axhline(1.0, color='k', lw=0.8, ls=':', alpha=0.6,
                           label='Ideal (= 1)' if ci == 0 else None)
                ss_max = max(np.nanmax(g_vals), np.nanmax(o_vals), 1.0)
                ax.set_ylim(0, np.ceil(ss_max * 1.1))
                ax.axhspan(0, 1.0, color='#fee8d6', alpha=0.35, zorder=0,
                           label='Overconfident' if ci == 0 else None)

            # RMSE row: optional Argo overlay
            if ri == 1 and argo_df is not None:
                season_col = 'winter_rmse' if ci == 0 else 'oct_rmse'
                n_col      = 'winter_n'    if ci == 0 else 'oct_n'
                if season_col in argo_df.columns:
                    argo_leads = argo_df['lead_day'].values
                    argo_rmse  = argo_df[season_col].values
                    argo_n     = argo_df[n_col].values if n_col in argo_df.columns else None
                    ms = 20 + 200 * (argo_n / argo_n.max()) if argo_n is not None else 50
                    ax.scatter(argo_leads, argo_rmse, s=ms, color=color,
                               marker='o', zorder=5,
                               label='vs Argo' if ci == 0 else None)

            ax.set_ylabel(row_labels[ri], fontsize=7)
            ax.set_xlim(1, 9)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(labelsize=7)

            if ri == 0:
                ax.set_title(col_labels[ci], fontsize=8, fontweight='bold')
            if ri == 1:
                ax.set_xlabel('Lead day', fontsize=7)

    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
               fontsize=7, framealpha=0.8,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'Ensemble Reliability: {vlab} — GLORYS vs OSTIA vs Argo',
                 fontsize=9, fontweight='bold')
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])

    out = args.output
    for ext in (['png', 'pdf'] if args.pdf else ['png']):
        p = f'{out}.{ext}'
        fig.savefig(p, dpi=args.dpi if ext == 'png' else None, bbox_inches='tight')
        print(f'Saved: {p}')
    plt.close(fig)

    # --- Optional Apr/Jun figure ---
    if args.glorys_apr_dir and args.glorys_jun_dir and args.ostia_apr_dir and args.ostia_jun_dir:
        g_apr = _load_csv(Path(args.glorys_apr_dir) / GLORYS_CSV, var)
        g_jun = _load_csv(Path(args.glorys_jun_dir) / GLORYS_CSV, var)
        o_apr = _load_csv(Path(args.ostia_apr_dir)  / OSTIA_CSV,  var)
        o_jun = _load_csv(Path(args.ostia_jun_dir)  / OSTIA_CSV,  var)

        _COL_APR = '#1B5E20'
        _COL_JUN = '#B71C1C'

        fig2, axes2 = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
        plt.rcParams.update({'font.size': 8})

        col_labels2 = ['Pre-monsoon (IC: 14 Apr 2020)',
                       'Monsoon (IC: 14 Jun 2020)']
        data_pairs2 = [(g_apr, o_apr), (g_jun, o_jun)]
        cols2       = [_COL_APR, _COL_JUN]

        for ci, (g_df, o_df) in enumerate(data_pairs2):
            color = cols2[ci]
            for ri, (gkey, okey) in enumerate(row_keys):
                ax = axes2[ri, ci]
                g_vals = _safe(g_df, gkey)
                o_vals = _safe(o_df, okey)
                ax.plot(leads, g_vals, color=color, lw=_LW, ls=_LS_GLORYS,
                        label='vs GLORYS' if (ri == 0 and ci == 0) else None)
                ax.plot(leads, o_vals, color=color, lw=_LW, ls=_LS_OSTIA,
                        label='vs OSTIA'  if (ri == 0 and ci == 0) else None)
                if ri == 0:
                    ax.axhline(1.0, color='k', lw=0.8, ls=':', alpha=0.6,
                               label='Ideal (= 1)' if ci == 0 else None)
                    ss_max = max(np.nanmax(g_vals), np.nanmax(o_vals), 1.0)
                    ax.set_ylim(0, np.ceil(ss_max * 1.1))
                    ax.axhspan(0, 1.0, color='#fee8d6', alpha=0.35, zorder=0,
                               label='Overconfident' if ci == 0 else None)
                ax.set_ylabel(row_labels[ri], fontsize=7)
                ax.set_xlim(1, 9)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.grid(axis='y', alpha=0.3)
                ax.tick_params(labelsize=7)
                if ri == 0:
                    ax.set_title(col_labels2[ci], fontsize=8, fontweight='bold')
                if ri == 1:
                    ax.set_xlabel('Lead day', fontsize=7)

        handles2, labels2 = axes2[0, 0].get_legend_handles_labels()
        fig2.legend(handles2, labels2, loc='lower center', ncol=4,
                    fontsize=7, framealpha=0.8,
                    bbox_to_anchor=(0.5, -0.02))
        fig2.suptitle(f'Ensemble Reliability: {vlab} — GLORYS vs OSTIA vs Argo',
                      fontsize=9, fontweight='bold')
        fig2.tight_layout(rect=[0, 0.06, 1, 0.96])

        out2 = args.output.replace('fig9_ensemble_skill', 'fig9_ensemble_skill_apr_jun')
        if out2 == args.output:
            out2 = args.output + '_apr_jun'
        for ext in (['png', 'pdf'] if args.pdf else ['png']):
            p2 = f'{out2}.{ext}'
            fig2.savefig(p2, dpi=args.dpi if ext == 'png' else None, bbox_inches='tight')
            print(f'Saved: {p2}')
        plt.close(fig2)


if __name__ == '__main__':
    main()

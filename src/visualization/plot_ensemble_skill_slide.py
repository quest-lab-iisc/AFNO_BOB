"""Slide-quality figure: ensemble spread-skill ratio, CRPS, and ARGO validation.

Compares two ensemble configurations for two Bay of Bengal seasons:
  - TIGGE-only : atmospheric perturbation, no IC perturbation
  - IC + TIGGE : EOF-based ocean IC perturbation + atmospheric perturbation

Produces a wide three-panel figure:
  Left   — Spread-Skill Ratio (spread / RMSE) vs lead day for SST.
            Dashed horizontal line marks the ideal ratio = 1.
  Middle — CRPS vs lead day for SST; RMSE shown as a reference.
  Right  — ARGO-validated SST RMSE vs lead day (independent obs verification).
            GLORYS-verified RMSE plotted as reference lines; ARGO RMSE as
            large markers sized by the number of collocations at each lead.

ARGO collocations come from the TIGGE-only run (the ensemble mean forecast
is essentially identical to IC+TIGGE at the mean level; IC perturbations
affect spread, not the mean trajectory).

Inputs:
    --ic_jan_dir (str): Verification dir for IC+TIGGE Winter ensemble.
    --ic_oct_dir (str): Verification dir for IC+TIGGE PostMonsoon ensemble.
    --tigge_jan_dir (str): Verification dir for TIGGE-only Winter ensemble.
    --tigge_oct_dir (str): Verification dir for TIGGE-only PostMonsoon ensemble.
    --argo_jan_csv (str): argo_collocations.csv for Winter ensemble.
    --argo_oct_csv (str): argo_collocations.csv for PostMonsoon ensemble.
    --var (str): Ocean variable to plot (default: thetao).
    --output (str): Output base path without extension.
    --pdf (flag): Also save vector PDF alongside the PNG.

Outputs:
    {output}.png   — 300 dpi raster
    {output}.pdf   — vector PDF, only with --pdf

Example:
    conda activate BoB_Surf_2
    python src/visualization/plot_ensemble_skill_slide.py \\
        --ic_jan_dir    results/AFNO_BoB_Surf_E11p1/ensemble_ic_Jan_14012020/verification \\
        --ic_oct_dir    results/AFNO_BoB_Surf_E11p1/ensemble_ic_Oct_14102020/verification \\
        --tigge_jan_dir results/AFNO_BoB_Surf_E11p1/ensemble_Jan_14012020/verification \\
        --tigge_oct_dir results/AFNO_BoB_Surf_E11p1/ensemble_Oct_14102020/verification \\
        --argo_jan_csv  results/AFNO_BoB_Surf_E11p1/ensemble_Jan_14012020/argo_validation/argo_collocations.csv \\
        --argo_oct_csv  results/AFNO_BoB_Surf_E11p1/ensemble_Oct_14102020/argo_validation/argo_collocations.csv \\
        --output results/figures/ensemble_skill_slide
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_COL_WIN     = '#1565C0'   # deep blue  — Winter
_COL_OCT     = '#E65100'   # deep orange — PostMonsoon
_ALPHA_TIGGE = 0.45        # opacity for TIGGE-only lines
_LW          = 2.5         # line width
_MS          = 7           # marker size

_VAR_LABELS = {
    'thetao': 'SST',
    'so':     'SSS',
    'uo':     'U-current',
    'vo':     'V-current',
    'zos':    'SSH',
}

_VAR_UNITS = {
    'thetao': '°C',
    'so':     'PSU',
    'uo':     'm s⁻¹',
    'vo':     'm s⁻¹',
    'zos':    'm',
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_var(csv_path: Path, var: str) -> pd.DataFrame:
    """Load verification metrics for a single variable from a CSV file.

    Args:
        csv_path (Path): Path to verification_metrics.csv.
        var (str): Variable name to filter, e.g. 'thetao'.

    Returns:
        pd.DataFrame: Rows for the requested variable, sorted by lead_day.

    Example:
        >>> df = load_var(Path('verification/verification_metrics.csv'), 'thetao')
    """
    df = pd.read_csv(csv_path)
    return df[df['variable'] == var].sort_values('lead_day').reset_index(drop=True)


def load_argo_rmse(csv_path: Path) -> pd.DataFrame:
    """Compute SST RMSE and bias vs ARGO collocations by lead day.

    Args:
        csv_path (Path): Path to argo_collocations.csv with columns
            lead_day, argo_temp, model_thetao.

    Returns:
        pd.DataFrame: Columns lead_day, n, rmse, bias; one row per lead day.

    Example:
        >>> argo = load_argo_rmse(Path('argo_validation/argo_collocations.csv'))
    """
    df = pd.read_csv(csv_path)
    rows = []
    for lead, grp in df.groupby('lead_day'):
        err  = grp['model_thetao'] - grp['argo_temp']
        rows.append({
            'lead_day': int(lead),
            'n':        len(grp),
            'rmse':     float((err ** 2).mean() ** 0.5),
            'bias':     float(err.mean()),
        })
    return pd.DataFrame(rows).sort_values('lead_day').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def build_slide(
    ic_jan:    pd.DataFrame,
    ic_oct:    pd.DataFrame,
    tig_jan:   pd.DataFrame,
    tig_oct:   pd.DataFrame,
    var:       str,
    output:    str,
    save_pdf:  bool = False,
    argo_jan:  pd.DataFrame | None = None,
    argo_oct:  pd.DataFrame | None = None,
) -> None:
    """Build and save the slide-quality figure (2 or 3 panels).

    Left   : Spread-Skill Ratio vs lead day.
    Middle : CRPS vs lead day (RMSE as reference).
    Right  : ARGO-validated SST RMSE vs lead day (only if argo_jan/argo_oct given).

    Args:
        ic_jan (pd.DataFrame): IC+TIGGE Winter verification, one variable.
        ic_oct (pd.DataFrame): IC+TIGGE PostMonsoon verification, one variable.
        tig_jan (pd.DataFrame): TIGGE-only Winter verification, one variable.
        tig_oct (pd.DataFrame): TIGGE-only PostMonsoon verification, one variable.
        var (str): Variable name (for axis labels).
        output (str): Base path without extension.
        save_pdf (bool): Also save vector PDF.
        argo_jan (pd.DataFrame or None): ARGO RMSE by lead day for Winter.
        argo_oct (pd.DataFrame or None): ARGO RMSE by lead day for PostMonsoon.

    Returns:
        None: Saves PNG (and optionally PDF).

    Example:
        >>> build_slide(ic_jan, ic_oct, tig_jan, tig_oct, 'thetao',
        ...             'results/figures/ensemble_skill_slide',
        ...             argo_jan=argo_jan_df, argo_oct=argo_oct_df)
    """
    var_label  = _VAR_LABELS.get(var, var.upper())
    unit       = _VAR_UNITS.get(var, '')
    show_argo  = (argo_jan is not None) and (argo_oct is not None)
    n_panels   = 3 if show_argo else 2

    leads = ic_jan['lead_day'].values

    plt.rcParams.update({
        'font.size':         13,
        'axes.titlesize':    14,
        'axes.labelsize':    13,
        'xtick.labelsize':   11,
        'ytick.labelsize':   11,
        'legend.fontsize':   10,
        'axes.spines.top':   False,
        'axes.spines.right': False,
    })

    fig_w = 14 if n_panels == 2 else 20
    fig, axes = plt.subplots(
        1, n_panels, figsize=(fig_w, 5.5),
        gridspec_kw={'wspace': 0.32}
    )
    ax_ss, ax_crps = axes[0], axes[1]
    ax_argo = axes[2] if show_argo else None

    # -----------------------------------------------------------------------
    # Panel 1 — Spread-Skill Ratio
    # -----------------------------------------------------------------------
    ax = ax_ss

    # IC+TIGGE (solid)
    ax.plot(leads, ic_jan['spread_skill'],
            color=_COL_WIN, lw=_LW, marker='o', ms=_MS,
            label='IC+TIGGE  Winter')
    ax.plot(leads, ic_oct['spread_skill'],
            color=_COL_OCT, lw=_LW, marker='s', ms=_MS,
            label='IC+TIGGE  Post-monsoon')

    # TIGGE-only (dashed, faded)
    ax.plot(leads, tig_jan['spread_skill'],
            color=_COL_WIN, lw=_LW, ls='--', marker='o', ms=_MS - 2,
            alpha=_ALPHA_TIGGE, label='TIGGE-only  Winter')
    ax.plot(leads, tig_oct['spread_skill'],
            color=_COL_OCT, lw=_LW, ls='--', marker='s', ms=_MS - 2,
            alpha=_ALPHA_TIGGE, label='TIGGE-only  Post-monsoon')

    # Ideal calibration line
    ax.axhline(1.0, color='#555555', lw=1.2, ls=':', label='Ideal  (SS = 1)')

    ax.set_xlabel('Forecast lead (days)')
    ax.set_ylabel('Spread-Skill Ratio')
    ax.set_title(f'{var_label} — Spread-Skill Ratio', fontweight='bold')
    ax.set_xticks(leads)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.legend(frameon=False, ncol=1, loc='upper right')
    ax.grid(axis='y', lw=0.5, alpha=0.4)

    # -----------------------------------------------------------------------
    # Panel 2 — CRPS
    # -----------------------------------------------------------------------
    ax = ax_crps

    # IC+TIGGE CRPS (solid)
    ax.plot(leads, ic_jan['crps'],
            color=_COL_WIN, lw=_LW, marker='o', ms=_MS,
            label='IC+TIGGE  Winter')
    ax.plot(leads, ic_oct['crps'],
            color=_COL_OCT, lw=_LW, marker='s', ms=_MS,
            label='IC+TIGGE  Post-monsoon')

    # TIGGE-only CRPS (dashed, faded)
    ax.plot(leads, tig_jan['crps'],
            color=_COL_WIN, lw=_LW, ls='--', marker='o', ms=_MS - 2,
            alpha=_ALPHA_TIGGE, label='TIGGE-only  Winter')
    ax.plot(leads, tig_oct['crps'],
            color=_COL_OCT, lw=_LW, ls='--', marker='s', ms=_MS - 2,
            alpha=_ALPHA_TIGGE, label='TIGGE-only  Post-monsoon')

    # RMSE reference (IC+TIGGE ensemble mean)
    ax.plot(leads, ic_jan['rmse'],
            color=_COL_WIN, lw=1.2, ls=':', ms=0,
            alpha=0.7, label='RMSE  Winter (ref)')
    ax.plot(leads, ic_oct['rmse'],
            color=_COL_OCT, lw=1.2, ls=':', ms=0,
            alpha=0.7, label='RMSE  Post-monsoon (ref)')

    ax.set_xlabel('Forecast lead (days)')
    ax.set_ylabel(f'CRPS  ({unit})')
    ax.set_title(f'{var_label} — CRPS', fontweight='bold')
    ax.set_xticks(leads)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.legend(frameon=False, ncol=1, loc='upper left')
    ax.grid(axis='y', lw=0.5, alpha=0.4)

    # -----------------------------------------------------------------------
    # Panel 3 — ARGO validation (if provided)
    # -----------------------------------------------------------------------
    if show_argo:
        ax = ax_argo

        # GLORYS RMSE reference lines (IC+TIGGE, same model)
        ax.plot(leads, ic_jan['rmse'],
                color=_COL_WIN, lw=1.5, ls='--', alpha=0.5,
                label='GLORYS RMSE  Winter')
        ax.plot(leads, ic_oct['rmse'],
                color=_COL_OCT, lw=1.5, ls='--', alpha=0.5,
                label='GLORYS RMSE  Post-monsoon')

        # ARGO RMSE — markers sized by number of collocations
        for argo_df, col, marker, season in [
            (argo_jan, _COL_WIN, 'o', 'Winter'),
            (argo_oct, _COL_OCT, 's', 'Post-monsoon'),
        ]:
            sizes = (argo_df['n'] * 18).clip(lower=30)   # min visible size
            sc = ax.scatter(argo_df['lead_day'], argo_df['rmse'],
                            c=col, marker=marker, s=sizes, zorder=5,
                            edgecolors='white', linewidths=0.6,
                            label=f'ARGO RMSE  {season}')
            ax.plot(argo_df['lead_day'], argo_df['rmse'],
                    color=col, lw=_LW, alpha=0.85)

        ax.set_xlabel('Forecast lead (days)')
        ax.set_ylabel(f'RMSE  ({unit})')
        ax.set_title(f'{var_label} — ARGO Validation', fontweight='bold')
        ax.set_xticks(leads)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.legend(frameon=False, ncol=1, loc='upper left', fontsize=9)
        ax.grid(axis='y', lw=0.5, alpha=0.4)
        ax.text(0.97, 0.05,
                'Marker size ∝ N floats',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8, color='#555555')

    # -----------------------------------------------------------------------
    # Shared annotation strip
    # -----------------------------------------------------------------------
    fig.text(0.5, 0.01,
             'Solid = IC + TIGGE  |  Dashed = TIGGE-only  |  '
             'Init: 14 Jan 2020 (Winter)  /  14 Oct 2020 (Post-monsoon)  |  '
             'Model: AFNO E11p1  |  50 members',
             ha='center', va='bottom', fontsize=9, color='#444444')

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    png_path = f'{output}.png'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {png_path}')
    if save_pdf:
        pdf_path = f'{output}.pdf'
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f'Saved: {pdf_path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and produce the ensemble skill slide figure.

    Args:
        None: All parameters read from sys.argv via argparse.

    Returns:
        None: Saves PNG (and optionally PDF) to --output.

    Example:
        >>> # python src/visualization/plot_ensemble_skill_slide.py \\
        >>> #     --ic_jan_dir    results/AFNO_BoB_Surf_E11p1/ensemble_ic_Jan_14012020/verification \\
        >>> #     --ic_oct_dir    results/AFNO_BoB_Surf_E11p1/ensemble_ic_Oct_14102020/verification \\
        >>> #     --tigge_jan_dir results/AFNO_BoB_Surf_E11p1/ensemble_Jan_14012020/verification \\
        >>> #     --tigge_oct_dir results/AFNO_BoB_Surf_E11p1/ensemble_Oct_14102020/verification \\
        >>> #     --output results/figures/ensemble_skill_slide
    """
    parser = argparse.ArgumentParser(
        description='Slide figure: ensemble spread-skill and CRPS vs lead time'
    )
    parser.add_argument(
        '--ic_jan_dir',
        default='results/AFNO_BoB_Surf_E11p1/ensemble_ic_Jan_14012020/verification',
        help='Verification dir for IC+TIGGE Winter ensemble'
    )
    parser.add_argument(
        '--ic_oct_dir',
        default='results/AFNO_BoB_Surf_E11p1/ensemble_ic_Oct_14102020/verification',
        help='Verification dir for IC+TIGGE PostMonsoon ensemble'
    )
    parser.add_argument(
        '--tigge_jan_dir',
        default='results/AFNO_BoB_Surf_E11p1/ensemble_Jan_14012020/verification',
        help='Verification dir for TIGGE-only Winter ensemble'
    )
    parser.add_argument(
        '--tigge_oct_dir',
        default='results/AFNO_BoB_Surf_E11p1/ensemble_Oct_14102020/verification',
        help='Verification dir for TIGGE-only PostMonsoon ensemble'
    )
    parser.add_argument('--argo_jan_csv', default=None,
                        help='argo_collocations.csv for Winter ensemble (optional)')
    parser.add_argument('--argo_oct_csv', default=None,
                        help='argo_collocations.csv for PostMonsoon ensemble (optional)')
    parser.add_argument('--var',    default='thetao',
                        help='Variable to plot (default: thetao)')
    parser.add_argument('--output', default='results/figures/ensemble_skill_slide',
                        help='Output base path without extension')
    parser.add_argument('--pdf',    action='store_true',
                        help='Also save a vector PDF alongside the PNG')
    args = parser.parse_args()

    def csv(d):
        return Path(d) / 'verification_metrics.csv'

    print('Loading verification metrics ...')
    ic_jan  = load_var(csv(args.ic_jan_dir),    args.var)
    ic_oct  = load_var(csv(args.ic_oct_dir),    args.var)
    tig_jan = load_var(csv(args.tigge_jan_dir), args.var)
    tig_oct = load_var(csv(args.tigge_oct_dir), args.var)

    argo_jan = argo_oct = None
    if args.argo_jan_csv and args.argo_oct_csv:
        print('Loading ARGO collocations ...')
        argo_jan = load_argo_rmse(Path(args.argo_jan_csv))
        argo_oct = load_argo_rmse(Path(args.argo_oct_csv))

    print('Building slide figure ...')
    build_slide(ic_jan, ic_oct, tig_jan, tig_oct, args.var, args.output, args.pdf,
                argo_jan=argo_jan, argo_oct=argo_oct)


if __name__ == '__main__':
    main()

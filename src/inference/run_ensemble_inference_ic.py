"""Run ocean ensemble forecasts with perturbed initial conditions (IC).

Generates an ensemble of ocean IC perturbations by sampling from the leading
seasonal EOFs computed by compute_ocean_eofs.py, then runs the AFNO ocean model
autoregressively for each member.

Atmospheric forcing for each member can be:
  (a) ERA5 deterministic forcing — same for all members (--tigge_dir omitted).
  (b) TIGGE ensemble forcing — member i uses TIGGE member i (--tigge_dir given).
      When --tigge_dir is supplied the ensemble size is limited to the number of
      available TIGGE members.

Season is inferred automatically from --input_date using the Bay of Bengal
season definitions:
    Winter:      December, January, February
    PreMonsoon:  March, April, May
    Monsoon:     June, July, August, September
    PostMonsoon: October, November

Each IC perturbation is drawn from N(0, I) in the PC space of the retained
seasonal EOFs, then projected back to the 224×224 model input space.  The
perturbation amplitude is scaled by the per-mode PC standard deviation so that
the ensemble spread matches historical within-season variability.

Inputs:
    --config_file (str): YAML config filename in config/ (default: afno_bob_config.yaml).
    --model_path (str): Path to .pth weights (default: from config).
    --input_date (str): Initialisation date in dd-mm-yyyy format.
    --eof_dir (str): Directory with seasonal EOF .npy files from
        compute_ocean_eofs.py (default: data/ocean_eofs).
    --n_members (int): Ensemble size — number of IC perturbations (default: 50).
    --ic_scale (float): Multiplicative scale applied to EOF perturbations
        (default: 1.0 — realistic historical spread).
    --tigge_dir (str): Optional. Directory with TIGGE per-member .npy files.
        If supplied, TIGGE member i is paired with IC perturbation i.
    --month (str): Month label for TIGGE filenames, e.g. 'Jan' (required if
        --tigge_dir is given).
    --device (str): Torch device string (default: from config).
    --output_dir (str): Override output directory.
    --seed (int): Random seed for IC perturbation draws (default: 42).

Outputs:
    {results_dir}/{name}/ensemble_ic_{month}_{date}/   (or ensemble_ic_era5_{date}/)
        ensemble_predictions.npy : float32, shape (N, 9, 5, 224, 224)
        ensemble_mean.npy        : float32, shape (9, 5, 224, 224)
        ensemble_std.npy         : float32, shape (9, 5, 224, 224)
        ensemble_metadata.json   : run parameters including EOF info.

Example:
    conda activate BoB_Surf_2
    python src/inference/run_ensemble_inference_ic.py \\
        --config_file afno_bob_surf_e11p1.yaml \\
        --input_date 14-01-2020 \\
        --eof_dir data/ocean_eofs \\
        --n_members 50 \\
        --tigge_dir data/tigge_ensemble \\
        --month Jan
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))

from configmypy import ConfigPipeline, YamlConfig
from models.architectures.afno.afnonet import AFNONet
from data_pipeline.preprocessing.transformers.normalize_transform import PreprocessTransform
from inference.utils import date_to_day_index
from inference.run_inference import (
    load_normalization_stats,
    load_initial_ocean_state,
    load_atmospheric_forcing,
)


# ---------------------------------------------------------------------------
# Season helpers
# ---------------------------------------------------------------------------

SEASONS = {
    'Winter':      [12, 1, 2],
    'PreMonsoon':  [3, 4, 5],
    'Monsoon':     [6, 7, 8, 9],
    'PostMonsoon': [10, 11],
}

OCEAN_VARS = ['thetao', 'so', 'uo', 'vo', 'zos']


def get_season(month: int) -> str:
    """Return the season name for a given calendar month number.

    Args:
        month (int): Calendar month (1–12).

    Returns:
        str: Season name, one of 'Winter', 'PreMonsoon', 'Monsoon', 'PostMonsoon'.

    Example:
        >>> get_season(1)
        'Winter'
        >>> get_season(10)
        'PostMonsoon'
    """
    for name, months in SEASONS.items():
        if month in months:
            return name
    raise ValueError(f'Month {month} not mapped to any season')


# ---------------------------------------------------------------------------
# EOF loading and perturbation generation
# ---------------------------------------------------------------------------

def load_season_eofs(eof_dir: Path, season: str) -> dict:
    """Load precomputed seasonal EOFs for all ocean variables.

    Args:
        eof_dir (Path): Directory produced by compute_ocean_eofs.py.
        season (str): Season name, e.g. 'Winter'.

    Returns:
        dict: Maps variable name → {'eofs': (k, H, W), 'pcstd': (k,),
            'mean': (H, W)}.

    Example:
        >>> eofs = load_season_eofs(Path('data/ocean_eofs'), 'Winter')
        >>> eofs['thetao']['eofs'].shape
        (23, 224, 224)
    """
    eofs = {}
    for var in OCEAN_VARS:
        tag = f'{season}_{var}'
        eofs[var] = {
            'eofs':  np.load(eof_dir / f'{tag}_eofs.npy'),   # (k, H, W)
            'pcstd': np.load(eof_dir / f'{tag}_pcstd.npy'),  # (k,)
            'mean':  np.load(eof_dir / f'{tag}_mean.npy'),   # (H, W)
        }
    return eofs


def generate_ic_perturbation(
    season_eofs: dict,
    rng: np.random.Generator,
    scale: float = 1.0,
) -> dict:
    """Draw one set of IC perturbations from the seasonal EOF distribution.

    For each variable, samples random PC coefficients α ~ N(0, 1), scales them
    by the per-mode standard deviation, and projects back to the 224×224 grid.

    Args:
        season_eofs (dict): Output of load_season_eofs — maps var to EOF arrays.
        rng (np.random.Generator): Seeded random number generator.
        scale (float): Multiplicative scale on perturbation amplitude (default 1.0).

    Returns:
        dict: Maps variable name → perturbation array of shape (224, 224), float32.

    Example:
        >>> rng = np.random.default_rng(42)
        >>> pert = generate_ic_perturbation(eofs, rng)
        >>> pert['thetao'].shape
        (224, 224)
    """
    perturbations = {}
    for var, data in season_eofs.items():
        eofs  = data['eofs']   # (k, H, W)
        pcstd = data['pcstd']  # (k,)
        k = len(pcstd)

        alpha = rng.standard_normal(k).astype(np.float32)   # (k,)
        coeffs = alpha * pcstd * scale                       # (k,) scaled
        # Project: sum_i coeffs_i * eofs_i  →  (H, W)
        pert = (coeffs[:, None, None] * eofs).sum(axis=0)
        perturbations[var] = pert.astype(np.float32)
    return perturbations


def apply_perturbation_to_ocean_state(
    ocean_state: dict,
    perturbation: dict,
) -> dict:
    """Add an IC perturbation to the current ocean state dictionary.

    Args:
        ocean_state (dict): Maps variable → torch.Tensor of shape (1, 224, 224)
            (the normalised IC as returned by load_initial_ocean_state).
        perturbation (dict): Maps variable → np.ndarray of shape (224, 224).

    Returns:
        dict: New ocean state dict with perturbation added, same tensor shapes.

    Example:
        >>> perturbed = apply_perturbation_to_ocean_state(ocean_state, pert)
    """
    perturbed = {}
    for var, tensor in ocean_state.items():
        delta = torch.tensor(
            perturbation[var], dtype=torch.float32
        ).unsqueeze(0)                                  # (1, 224, 224)
        perturbed[var] = tensor + delta
    return perturbed


# ---------------------------------------------------------------------------
# Single-member forecast
# ---------------------------------------------------------------------------

def run_forecast(
    config,
    model: torch.nn.Module,
    ocean_state: dict,
    atm_forcings: np.ndarray | None,
    init_day_index: int,
    transform,
    mean_dict: dict,
    variance_dict: dict,
    device: torch.device,
) -> np.ndarray:
    """Run a 9-step autoregressive forecast for one ensemble member.

    Args:
        config: Configuration object.
        model (torch.nn.Module): AFNO model in eval mode.
        ocean_state (dict): Normalised IC, maps var → tensor (1, 224, 224).
        atm_forcings (np.ndarray or None): Pre-loaded TIGGE atm array of shape
            (9, 6, 224, 224) in normalised model space, or None to use ERA5.
        init_day_index (int): Day index of the IC in the ERA5 dataset (used when
            atm_forcings is None).
        transform: PreprocessTransform instance.
        mean_dict (dict): Normalisation means.
        variance_dict (dict): Normalisation variances.
        device (torch.device): Target device.

    Returns:
        np.ndarray: Float32 array of shape (9, 5, 224, 224) — normalised
            predictions for lead days +1 through +9.

    Example:
        >>> preds = run_forecast(config, model, ocean_state, None, day_idx, ...)
        >>> preds.shape
        (9, 5, 224, 224)
    """
    model.eval()
    n_atm = len(config.data.atm_variable)   # 6
    predictions = []

    current_ocean = ocean_state

    with torch.no_grad():
        for step in range(9):
            # --- Atmospheric forcing ---
            if atm_forcings is not None:
                atm = torch.tensor(
                    atm_forcings[step], dtype=torch.float32
                ).to(device)                                    # (6, 224, 224)
            else:
                atm = load_atmospheric_forcing(
                    config, init_day_index + step + 1,
                    transform, mean_dict, variance_dict,
                ).to(device)                                    # (6, 224, 224)

            # --- Build model input: [atm | ocean] ---
            ocean_tensor = torch.cat(
                [current_ocean[v].to(device) for v in OCEAN_VARS], dim=0
            )                                                   # (5, 224, 224)
            x = torch.cat([atm, ocean_tensor], dim=0).unsqueeze(0)  # (1, 11, 224, 224)

            pred = model(x).squeeze(0)                          # (5, 224, 224)

            # Apply northern boundary mask to ocean output
            rows = config.data.north_mask_rows
            if rows > 0:
                pred[:, -rows:, :] = 0.0

            predictions.append(pred.cpu().numpy())

            # Update ocean state for next step: pred[vi] is (H, W), need (1, H, W)
            current_ocean = {
                var: pred[vi].unsqueeze(0).cpu()   # (1, 224, 224)
                for vi, var in enumerate(OCEAN_VARS)
            }

    return np.stack(predictions, axis=0).astype(np.float32)  # (9, 5, 224, 224)


# ---------------------------------------------------------------------------
# North-row fill (model forces pred[:, -20:, :] = 0 during inference)
# ---------------------------------------------------------------------------

_N_NORTH_ROWS = 20


def fill_north_rows_224(preds: np.ndarray) -> np.ndarray:
    """Fill the zeroed northernmost rows of each member's prediction before aggregation.

    The model applies a hard mask that zeros the last 20 rows of every 224×224
    output during inference.  Computing ensemble mean and std from these zeroed
    rows would produce artefacts in the statistics.  This function replaces those
    rows with the value from the last valid row (index -21) via constant
    extrapolation, which is physically reasonable for the narrow Bangladesh/Myanmar
    delta boundary strip.

    The fill is applied before mean/std aggregation so that both statistics are
    computed from consistent, artefact-free values.  In 224×224 normalized space
    there is no explicit land mask (NaN was filled to per-sample mean during
    preprocessing), so the fill covers all channels uniformly.

    Args:
        preds (np.ndarray): shape (N, L, C, 224, 224) — stacked member predictions
            in normalized space, with the last 20 rows zeroed by the model mask.

    Returns:
        np.ndarray: Same shape as preds; rows [-20:] replaced by the value at
            row [-21] for every member, lead, and channel.

    Example:
        >>> all_preds = np.stack(member_list, axis=0)   # (50, 9, 5, 224, 224)
        >>> all_preds = fill_north_rows_224(all_preds)
        >>> ens_mean = all_preds.mean(axis=0)
    """
    out = preds.copy()
    # row index -(_N_NORTH_ROWS+1) = the last row with valid model output
    boundary = out[..., -(_N_NORTH_ROWS + 1) : -_N_NORTH_ROWS, :]   # (..., 1, 224)
    out[..., -_N_NORTH_ROWS:, :] = boundary
    return out


# ---------------------------------------------------------------------------
# TIGGE loader (mirrors run_ensemble_inference.py)
# ---------------------------------------------------------------------------

def load_tigge_member(
    tigge_dir: Path,
    month: str,
    member_idx: int,
    day_in_month: int,
) -> np.ndarray:
    """Load TIGGE pre-processed forcing for one ensemble member.

    Args:
        tigge_dir (Path): Directory with Tigge_{month}_2020_ens_m{nn:02d}.npy.
        month (str): Month label, e.g. 'Jan'.
        member_idx (int): 1-based member index.
        day_in_month (int): 0-based day index within the file's first axis.

    Returns:
        np.ndarray: Float32 array of shape (9, 6, 224, 224) — normalised atm
            forcing for lead days +1 through +9.

    Example:
        >>> atm = load_tigge_member(Path('data/tigge_ensemble'), 'Jan', 1, 13)
        >>> atm.shape
        (9, 6, 224, 224)
    """
    path = tigge_dir / f'Tigge_{month}_2020_ens_m{member_idx:02d}.npy'
    arr = np.load(path)                      # (n_days, 9, 6, 224, 224)
    idx = min(day_in_month, arr.shape[0] - 1)
    return arr[idx].astype(np.float32)       # (9, 6, 224, 224)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Parse CLI arguments and run the IC-perturbed ocean ensemble forecast.

    Args:
        None: All parameters sourced from sys.argv via argparse.

    Returns:
        None: Writes ensemble .npy files and metadata JSON to the output directory.

    Example:
        >>> # python src/inference/run_ensemble_inference_ic.py \\
        >>> #     --config_file afno_bob_surf_e11p1.yaml \\
        >>> #     --input_date 14-01-2020 \\
        >>> #     --eof_dir data/ocean_eofs \\
        >>> #     --n_members 50 \\
        >>> #     --tigge_dir data/tigge_ensemble \\
        >>> #     --month Jan
    """
    parser = argparse.ArgumentParser(
        description='Ocean ensemble forecast with EOF-based IC perturbations'
    )
    parser.add_argument('--config_file', default='afno_bob_config.yaml')
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--input_date', required=True,
                        help='IC date in dd-mm-yyyy format')
    parser.add_argument('--eof_dir', default='data/ocean_eofs',
                        help='Directory with seasonal EOF files')
    parser.add_argument('--n_members', type=int, default=50,
                        help='Number of IC ensemble members (default: 50)')
    parser.add_argument('--ic_scale', type=float, default=1.0,
                        help='Multiplicative scale on IC perturbation amplitude')
    parser.add_argument('--tigge_dir', default=None,
                        help='Optional TIGGE ensemble directory')
    parser.add_argument('--month', default=None,
                        help='Month label for TIGGE files, e.g. Jan (required if --tigge_dir given)')
    parser.add_argument('--device', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for IC perturbation draws')
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------------
    config_dir = Path('config')
    pipe = ConfigPipeline([
        YamlConfig(args.config_file, config_name='default', config_folder=str(config_dir)),
        YamlConfig(config_folder=str(config_dir)),
    ])
    config = pipe.read_conf()

    device = torch.device(
        args.device or (config.device if torch.cuda.is_available() else 'cpu')
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model_path = Path(
        args.model_path or
        f"{config.results.model_dir}/{config.name}.pth"
    )
    print(f'Loading model from {model_path}', flush=True)
    model = AFNONet(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # -----------------------------------------------------------------------
    # Normalization and transform
    # -----------------------------------------------------------------------
    transform = PreprocessTransform(config)
    mean_dict, variance_dict = load_normalization_stats(config)

    # -----------------------------------------------------------------------
    # IC date → day index
    # -----------------------------------------------------------------------
    from datetime import datetime as dt
    init_dt = dt.strptime(args.input_date, '%d-%m-%Y')
    reference_date = config.evaluation.reference_date   # e.g. '01-01-1993'
    day_index = date_to_day_index(args.input_date, reference_date)
    season = get_season(init_dt.month)
    date_str = args.input_date.replace('-', '')

    print(f'Init date : {args.input_date}  (day index {day_index})', flush=True)
    print(f'Season    : {season}', flush=True)

    # -----------------------------------------------------------------------
    # Load reference (unperturbed) IC
    # -----------------------------------------------------------------------
    ref_ocean_state = load_initial_ocean_state(
        config, day_index, transform, mean_dict
    )

    # -----------------------------------------------------------------------
    # Load seasonal EOFs
    # -----------------------------------------------------------------------
    eof_dir = Path(args.eof_dir)
    print(f'Loading EOFs from {eof_dir} ...', flush=True)
    season_eofs = load_season_eofs(eof_dir, season)
    for var in OCEAN_VARS:
        k = season_eofs[var]['eofs'].shape[0]
        print(f'  {var}: {k} modes', flush=True)

    # -----------------------------------------------------------------------
    # TIGGE: validate and get day-in-month index
    # -----------------------------------------------------------------------
    use_tigge = args.tigge_dir is not None
    if use_tigge:
        if args.month is None:
            parser.error('--month is required when --tigge_dir is given')
        tigge_dir = Path(args.tigge_dir)
        day_in_month = init_dt.day - 1    # 0-based
        n_tigge = args.n_members
    else:
        tigge_dir = None
        n_tigge = 0

    # -----------------------------------------------------------------------
    # Output directory
    # -----------------------------------------------------------------------
    atm_tag = f'ic_{args.month}' if use_tigge else 'ic_era5'
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = (
            Path(config.results.save_dir) / config.name /
            f'ensemble_{atm_tag}_{date_str}'
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output    : {out_dir}', flush=True)

    # -----------------------------------------------------------------------
    # Ensemble loop
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    all_preds = []

    for m in range(args.n_members):
        print(f'  Member {m+1:02d}/{args.n_members} ... ', end='', flush=True)

        # IC perturbation
        pert = generate_ic_perturbation(season_eofs, rng, scale=args.ic_scale)
        perturbed_ocean = apply_perturbation_to_ocean_state(ref_ocean_state, pert)

        # Atmospheric forcing
        if use_tigge:
            atm_forcings = load_tigge_member(
                tigge_dir, args.month, m + 1, day_in_month
            )
        else:
            atm_forcings = None

        preds = run_forecast(
            config, model, perturbed_ocean,
            atm_forcings, day_index,
            transform, mean_dict, variance_dict, device,
        )
        all_preds.append(preds)
        print(f'thetao std={preds[:, 0].std():.4f}', flush=True)

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    all_preds = np.stack(all_preds, axis=0)          # (N, 9, 5, 224, 224)
    # Only fill north rows when the rollout actually zeroed them; for models with
    # north_mask_rows=0 those rows contain valid predictions and must not be overwritten.
    if config.data.north_mask_rows > 0:
        all_preds = fill_north_rows_224(all_preds)
    ens_mean  = all_preds.mean(axis=0)               # (9, 5, 224, 224)
    ens_std   = all_preds.std(axis=0)                # (9, 5, 224, 224)

    np.save(out_dir / 'ensemble_predictions.npy', all_preds)
    np.save(out_dir / 'ensemble_mean.npy',         ens_mean)
    np.save(out_dir / 'ensemble_std.npy',           ens_std)

    # EOF mode counts for metadata
    eof_modes = {var: int(season_eofs[var]['eofs'].shape[0]) for var in OCEAN_VARS}

    metadata = {
        'model':            config.name,
        'input_date':       args.input_date,
        'season':           season,
        'n_members':        args.n_members,
        'ic_scale':         args.ic_scale,
        'seed':             args.seed,
        'eof_dir':          str(eof_dir),
        'eof_modes':        eof_modes,
        'atm_forcing':      f'TIGGE {args.month}' if use_tigge else 'ERA5',
        'variables':        OCEAN_VARS,
        'lead_days':        list(range(1, 10)),
        'lead_times_h':     list(range(24, 217, 24)),
        'north_mask_rows':  int(config.data.north_mask_rows),
    }
    (out_dir / 'ensemble_metadata.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8'
    )

    print(f'\nEnsemble spread summary (mean std across all steps):')
    for vi, var in enumerate(OCEAN_VARS):
        print(f'  {var:<10}: mean std = {ens_std[:, vi].mean():.4f}')

    print(f'\nSaved to {out_dir}/')
    for f in ['ensemble_predictions.npy', 'ensemble_mean.npy',
              'ensemble_std.npy', 'ensemble_metadata.json']:
        arr = np.load(out_dir / f) if f.endswith('.npy') else None
        shape_str = f'  shape={arr.shape}' if arr is not None else ''
        print(f'  {f}{shape_str}')


if __name__ == '__main__':
    main()

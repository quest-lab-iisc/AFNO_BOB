"""Run ensemble ocean forecasts driven by TIGGE atmospheric forcing.

For each of the 50 TIGGE perturbed-forecast ensemble members, loads the
pre-processed atmospheric forcing (output of prepare_tigge_ensemble.py) and
runs the ocean-only AFNO model autoregressively for 9 lead-time steps.
Results are saved as per-member predictions, ensemble mean, and ensemble std.

The atmospheric forcing per member replaces the ERA5 forcing used during
normal inference (run_inference.py). The ocean model and normalisation are
identical to the standard inference pipeline.

Inputs:
    --config_file (str): YAML config in config/ (default: afno_bob_config.yaml).
    --model_path (str): Path to .pth weights (default: results/models/{name}.pth).
    --input_date (str): Initialisation date in dd-mm-yyyy format.
    --tigge_dir (Path): Directory containing pre-processed .npy files from
        prepare_tigge_ensemble.py.
    --month (str): Month label matching the .npy filename stem prefix
        (e.g. 'Jan' → looks for Tigge_Jan_2020_ens_m{nn:02d}.npy).
    --n_members (int): Number of ensemble members to use (default: 50).
    --device (str): Torch device (default: from config, e.g. 'cuda:0').
    --output_dir (str): Override output directory.

Outputs:
    {results_dir}/{name}/ensemble_{month}_{date}/
        ensemble_predictions.npy : float32, shape (N_members, 9, 5, 224, 224)
                                   Normalised-space predictions for all members.
        ensemble_mean.npy        : float32, shape (9, 5, 224, 224)
        ensemble_std.npy         : float32, shape (9, 5, 224, 224)
        ensemble_metadata.json   : run parameters, variable names, lead times.

Example:
    conda activate BoB_Surf_2
    python src/inference/run_ensemble_inference.py \\
        --config_file afno_bob_surf_e06p1.yaml \\
        --input_date 14-01-2020 \\
        --tigge_dir data/tigge_ensemble \\
        --month Jan \\
        --device cuda:0
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Resolve src/ on the path regardless of invocation directory
sys.path.append(str(Path(__file__).parent.parent))

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from models.architectures.afno.afnonet import AFNONet
from data_pipeline.preprocessing.transformers.normalize_transform import PreprocessTransform
from inference.utils import date_to_day_index
from inference.run_inference import (
    load_normalization_stats,
    load_initial_ocean_state,
)


# ---------------------------------------------------------------------------
# Ocean forecast with pre-loaded TIGGE atmospheric forcing
# ---------------------------------------------------------------------------

def run_forecast_with_tigge_forcing(
    config,
    model: torch.nn.Module,
    initial_ocean_state: dict,
    atm_forcings: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Run 9-step autoregressive ocean forecast using TIGGE atmospheric forcing.

    Mirrors run_autoregressive_forecast() in run_inference.py but substitutes
    the per-step ERA5 disk read with a slice from the pre-normalised TIGGE
    atm tensor.

    Args:
        config: Configuration object (same as used for training).
        model (torch.nn.Module): Loaded, eval-mode AFNO model.
        initial_ocean_state (dict): Preprocessed ocean variables at t=0,
            each a torch.Tensor of shape (1, 224, 224).
            Produced by load_initial_ocean_state().
        atm_forcings (np.ndarray): Pre-normalised TIGGE atmospheric forcing
            of shape (9, 6, 224, 224). Axis 0 = lead step (0 → +24 h,
            8 → +216 h), axis 1 = atm variable in ATM_VARS order.
        device (torch.device): Torch compute device.

    Returns:
        np.ndarray: Predictions in preprocessed space, shape (9, 5, 224, 224).
            Axis 0 = lead time (1-indexed in physical terms, i.e. +24 h first),
            axis 1 = ocean variable in config.data.out_variable order.

    Example:
        >>> preds = run_forecast_with_tigge_forcing(
        ...     config, model, ocean_state, atm_forcings[day_in_month], device
        ... )
        >>> preds.shape
        (9, 5, 224, 224)
    """
    n_steps = atm_forcings.shape[0]
    n_ocean_vars = len(config.data.out_variable)

    # Work with a mutable copy of the ocean state dict
    ocean_state = {k: v.clone() for k, v in initial_ocean_state.items()}

    predictions = np.empty((n_steps, n_ocean_vars, 224, 224), dtype=np.float32)

    for lead_time in range(1, n_steps + 1):
        step_idx = lead_time - 1

        # TIGGE atmospheric forcing at this lead time: (6, 224, 224)
        atm_tensor = torch.tensor(
            atm_forcings[step_idx], dtype=torch.float32
        )  # (6, 224, 224)

        # Assemble ocean tensor from current state
        ocean_tensor = torch.cat(
            [ocean_state[v] for v in config.data.variable], dim=0
        )  # (5, 224, 224)

        # Model input: [atm (6), ocean (5)] = 11 channels
        input_tensor = torch.cat([atm_tensor, ocean_tensor], dim=0)  # (11, 224, 224)

        with torch.no_grad():
            inp = input_tensor.unsqueeze(0).to(device)   # (1, 11, 224, 224)
            out = model(inp).squeeze(0).cpu()             # (5, 224, 224)

        # Store prediction and update ocean state for next step
        for i, var in enumerate(config.data.out_variable):
            ch = out[i]                                   # (224, 224)
            predictions[step_idx, i] = ch.numpy()
            ocean_state[var] = ch.unsqueeze(0)            # (1, 224, 224)

    return predictions


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_model(config, model_path: str, device: torch.device) -> torch.nn.Module:
    """Load a trained AFNO model from a .pth checkpoint.

    Args:
        config: Configuration object.
        model_path (str): Path to the .pth weights file.
        device (torch.device): Device to map the model onto.

    Returns:
        torch.nn.Module: Model in eval mode.

    Example:
        >>> model = load_model(config, 'results/models/AFNO_BoB_Surf_E06P1.pth', device)
    """
    model = AFNONet(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from: {model_path}")
    return model


def find_member_files(tigge_dir: Path, month: str, n_members: int) -> list[Path]:
    """Locate pre-processed per-member .npy files for the given month.

    Searches tigge_dir for files matching the pattern
    Tigge_{month}_2020_ens_m{nn:02d}.npy.

    Args:
        tigge_dir (Path): Directory produced by prepare_tigge_ensemble.py.
        month (str): Month label, e.g. 'Jan' or 'Oct'.
        n_members (int): Number of members expected (typically 50).

    Returns:
        list[Path]: Sorted list of .npy file paths, one per member.

    Example:
        >>> files = find_member_files(Path('data/tigge_ensemble'), 'Jan', 50)
        >>> len(files)
        50
    """
    stem = f"Tigge_{month}_2020_ens"
    files = []
    for m in range(1, n_members + 1):
        p = tigge_dir / f"{stem}_m{m:02d}.npy"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing member file: {p}\n"
                f"Run prepare_tigge_ensemble.py first."
            )
        files.append(p)
    return files


def month_label_to_day_in_file(input_date: str) -> int:
    """Return the 0-based day index within the month for the given date.

    TIGGE .npy files have shape (N_days, ...) where axis 0 runs from
    day 1 to day N of the month. This converts a date string to that index.

    Args:
        input_date (str): Date string in dd-mm-yyyy format.

    Returns:
        int: 0-based day index within the month (0 = 1st, 30 = 31st).

    Example:
        >>> month_label_to_day_in_file('14-01-2020')
        13
    """
    dt = datetime.strptime(input_date, "%d-%m-%Y")
    return dt.day - 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Parse CLI arguments and run the 50-member ensemble inference pipeline.

    Args:
        None: All parameters read from sys.argv via argparse.

    Returns:
        None: Writes ensemble_predictions.npy, ensemble_mean.npy,
            ensemble_std.npy, and ensemble_metadata.json to the output directory.

    Example:
        >>> # python src/inference/run_ensemble_inference.py \\
        >>> #     --config_file afno_bob_surf_e06p1.yaml \\
        >>> #     --input_date 14-01-2020 \\
        >>> #     --tigge_dir data/tigge_ensemble \\
        >>> #     --month Jan \\
        >>> #     --device cuda:0
    """
    parser = argparse.ArgumentParser(
        description="Ensemble ocean forecast with TIGGE atmospheric forcing"
    )
    parser.add_argument("--config_file", type=str, default="afno_bob_config.yaml",
                        help="YAML config filename (in config/)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to .pth model weights (default: results/models/{name}.pth)")
    parser.add_argument("--input_date", type=str, required=True,
                        help="Initialisation date dd-mm-yyyy")
    parser.add_argument("--tigge_dir", type=str, default="data/tigge_ensemble",
                        help="Directory with pre-processed TIGGE .npy files")
    parser.add_argument("--month", type=str, required=True,
                        help="Month label used in .npy filenames (e.g. Jan, Oct)")
    parser.add_argument("--n_members", type=int, default=50,
                        help="Number of ensemble members to run (default: 50)")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (default: from config)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    # --- Configuration ---
    print("=== Loading Configuration ===")
    pipe = ConfigPipeline([
        YamlConfig(args.config_file, config_name='default', config_folder='config/'),
        YamlConfig(config_folder='config/'),
    ])
    config = pipe.read_conf()

    device_str = args.device or getattr(config, 'device', 'cpu')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Model ---
    model_path = args.model_path or f"{config.results.model_dir}/{config.name}.pth"
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)
    model = load_model(config, model_path, device)

    # --- Dates / indices ---
    input_date   = args.input_date
    day_index    = date_to_day_index(input_date, "01-01-1993")
    day_in_month = month_label_to_day_in_file(input_date)
    print(f"Init date: {input_date}  (absolute day index: {day_index}, "
          f"day-in-month index: {day_in_month})")

    # --- Initial ocean state (shared across all members) ---
    print("\nLoading initial ocean state ...")
    mean_dict, variance_dict = load_normalization_stats(config)
    transform = PreprocessTransform(config)
    ocean_state = load_initial_ocean_state(config, day_index, transform, mean_dict)

    # --- Locate member files ---
    tigge_dir   = Path(args.tigge_dir)
    member_files = find_member_files(tigge_dir, args.month, args.n_members)
    n_members   = len(member_files)
    n_steps     = 9
    n_ocean_vars = len(config.data.out_variable)

    # --- Output directory ---
    date_tag = input_date.replace("-", "")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(config.results.save_dir) / config.name / \
                  f"ensemble_{args.month}_{date_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_dir}")

    # --- Allocate ensemble array ---
    # Shape: (n_members, n_steps, n_ocean_vars, 224, 224)
    all_preds = np.empty((n_members, n_steps, n_ocean_vars, 224, 224), dtype=np.float32)

    # --- Run inference for each member ---
    print(f"\n=== Running ensemble inference: {n_members} members × {n_steps} steps ===\n")
    t0 = datetime.now()

    for mi, member_file in enumerate(member_files):
        member_num = mi + 1
        print(f"Member {member_num:02d}/{n_members}  ({member_file.name}) ...", flush=True)

        # atm shape: (N_days, 9, 6, 224, 224)
        # Each .npy contains only the downloaded date(s); axis 0 is typically 1.
        atm_all_days = np.load(member_file)
        n_days_in_file = atm_all_days.shape[0]
        day_idx = min(day_in_month, n_days_in_file - 1)
        atm_forcing = atm_all_days[day_idx]  # (9, 6, 224, 224)

        preds = run_forecast_with_tigge_forcing(
            config, model, ocean_state, atm_forcing, device
        )  # (9, 5, 224, 224)

        all_preds[mi] = preds

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\nInference complete in {elapsed:.1f} s  "
          f"({elapsed / n_members:.1f} s/member)")

    # --- Ensemble statistics ---
    ens_mean = all_preds.mean(axis=0)   # (9, 5, 224, 224)
    ens_std  = all_preds.std(axis=0)    # (9, 5, 224, 224)

    # --- Save outputs ---
    preds_path = out_dir / "ensemble_predictions.npy"
    mean_path  = out_dir / "ensemble_mean.npy"
    std_path   = out_dir / "ensemble_std.npy"
    meta_path  = out_dir / "ensemble_metadata.json"

    np.save(preds_path, all_preds)
    np.save(mean_path,  ens_mean)
    np.save(std_path,   ens_std)

    metadata = {
        "input_date"      : input_date,
        "month"           : args.month,
        "day_index"       : int(day_index),
        "day_in_month"    : int(day_in_month),
        "n_members"       : n_members,
        "n_steps"         : n_steps,
        "lead_times_h"    : list(range(24, 24 * n_steps + 1, 24)),
        "ocean_variables" : list(config.data.out_variable),
        "atm_variables"   : ['ssr', 'tp', 'u10', 'v10', 'msl', 'tcc'],
        "model"           : config.name,
        "tigge_dir"       : str(tigge_dir),
        "predictions_shape": list(all_preds.shape),
        "generated_at"    : datetime.now().isoformat(timespec='seconds'),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n=== Saved outputs to {out_dir}/ ===")
    print(f"  ensemble_predictions.npy  shape={all_preds.shape}  "
          f"({all_preds.nbytes / 1e6:.1f} MB)")
    print(f"  ensemble_mean.npy         shape={ens_mean.shape}")
    print(f"  ensemble_std.npy          shape={ens_std.shape}")
    print(f"  ensemble_metadata.json")

    # --- Summary statistics ---
    print("\n=== Ensemble spread summary (mean std across all variables/steps) ===")
    for vi, var in enumerate(config.data.out_variable):
        avg_std = float(ens_std[:, vi].mean())
        print(f"  {var:10s}: mean std = {avg_std:.4f}")


if __name__ == "__main__":
    main()

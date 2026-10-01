"""Rollout-steps ablation trainer for AFNO.

Trains AFNO with a configurable autoregressive rollout horizon to find the
optimal number of steps for E13.  Architecture and data are identical to
E11p1 (hidden_size=768); only ``rollout_steps`` varies across runs.

Each run saves its best weights and a val-loss CSV so results can be compared
after all runs complete.

Inputs:
    --rollout_steps (int): Autoregressive steps per training iteration (default 3).
    --config_file (str): Base YAML config in config/ (default: afno_bob_surf_e11p1.yaml).
    --name (str): Experiment name override (default: AFNO_Ablation_rs<rollout_steps>).
    --opt.epochs (int): Number of training epochs (default 15).
    --device (str): Torch device string (default from config).

Outputs:
    results/models/<name>.pth                     Best model weights.
    results/models/checkpoint_<name>.pt           Full checkpoint.
    results/<name>/losses.csv                     Per-epoch train/val losses.
    results/experiments/logs/<name>.log           Verbatim console log.

Example:
    conda activate BoB_Surf_2
    python src/training/scripts/train_afno_rollout_ablation.py --rollout_steps 1
    python src/training/scripts/train_afno_rollout_ablation.py --rollout_steps 3
    python src/training/scripts/train_afno_rollout_ablation.py --rollout_steps 5
"""

import argparse
import sys
from pathlib import Path  # used by sys.path.append only

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent.parent.parent))

from configmypy import ArgparseConfig, ConfigPipeline, YamlConfig
from data_pipeline.loaders.data_loader import load_and_prepare_data
from models.architectures.afno.afnonet import AFNONet
from training.trainer import Trainer
from training.utils.experiment_logger import (
    cleanup_logging, log_experiment, setup_logging,
)


def main() -> None:
    """Parse args, load data, train AFNO with given rollout_steps, save results.

    Example:
        python src/training/scripts/train_afno_rollout_ablation.py --rollout_steps 3
    """
    # --- pre-parse our custom args, then strip them from sys.argv so
    #     ArgparseConfig (which is strict) never sees them ---
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--rollout_steps', type=int, default=3)
    pre.add_argument('--name', type=str, default=None)
    pre.add_argument('--config_file', type=str, default='afno_bob_surf_e11p1.yaml')
    known, remaining = pre.parse_known_args()

    rollout_steps = known.rollout_steps
    config_file   = known.config_file
    default_name  = f'AFNO_Ablation_rs{rollout_steps}'

    # Replace sys.argv so ConfigPipeline only sees remaining args
    sys.argv = [sys.argv[0]] + remaining

    pipe = ConfigPipeline([
        YamlConfig(f'./{config_file}', config_name='default', config_folder='config/'),
        YamlConfig(config_folder='config/'),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
    ])
    config = pipe.read_conf()

    # Apply name override: CLI --name > default ablation name
    if known.name:
        config.name = known.name
    elif config.name in ('AFNO_BoB_Surf_E11p1', 'AFNO_BoB_Surf_E11'):
        config.name = default_name

    tee_logger = setup_logging(config)
    try:
        print(f'=== Rollout ablation: rollout_steps={rollout_steps}  '
              f'name={config.name}  epochs={config.opt.epochs} ===')

        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(config.seed)

        log_experiment(config)

        print(f'\n=== Loading Data (k_steps={rollout_steps}) ===')
        train_loader, val_loader, _, mask = load_and_prepare_data(
            config, k_steps=rollout_steps)
        print(f'  Train batches: {len(train_loader)}  Val batches: {len(val_loader)}')

        print('\n=== Initialising Model ===')
        device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        model  = AFNONet(config).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'  Trainable parameters: {n_params:,}')

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.opt.lr,
            weight_decay=config.opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
        loss_fn = nn.L1Loss()

        trainer = Trainer(
            model=model, mask=mask, config=config,
            rollout_steps=rollout_steps,
        )

        print(f'\n=== Training (rollout_steps={rollout_steps}, '
              f'epochs={config.opt.epochs}) ===\n')
        trainer.train(train_loader, val_loader, loss_fn, optimizer, scheduler)

    except Exception as exc:
        import traceback
        print(f'\n[FATAL] {exc}')
        traceback.print_exc()
        raise
    finally:
        cleanup_logging(tee_logger)


if __name__ == '__main__':
    main()

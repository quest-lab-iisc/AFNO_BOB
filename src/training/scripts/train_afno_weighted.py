"""Train AFNO ocean model with per-variable weighted loss (experiment E12).

Applies loss weights that up-weight sea surface temperature (thetao) and
salinity (so) relative to currents and sea surface height.  All other training
settings match E00.

Variable loss weights: thetao=2.0, so=1.5, uo=1.0, vo=1.0, zos=0.5.
The loss is a weighted L1: weight[v] * |pred[v] - truth[v]|, summed over
all variables.

Inputs:
    --config_file (str): YAML config filename (default: afno_bob_surf_e12.yaml).
    --opt.epochs (int): Total training epochs (default 150).
    --opt.lr (float): Learning rate (default 0.001 from config).
    --device (str): Torch device string (default from config).
    --results.resume_checkpoint (str): Path to .pt checkpoint for resumption.
    --name (str): Override experiment name.

Outputs:
    results/models/AFNO_BoB_Surf_E12.pth          — best model weights.
    results/models/checkpoint_AFNO_BoB_Surf_E12.pt — full checkpoint.
    results/experiments/logs/AFNO_BoB_Surf_E12_train_val_losses.csv

Example:
    conda activate BoB_Surf_2
    python src/training/scripts/train_afno_weighted.py \\
        --name AFNO_BoB_Surf_E12 \\
        --device cuda:0 \\
        --opt.epochs 150
"""
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from models.architectures.afno.afnonet import AFNONet
from data_pipeline.loaders.data_loader import load_and_prepare_data
from training.trainer import Trainer
from training.utils.experiment_logger import log_experiment, setup_logging, cleanup_logging

# Loss weights for [thetao, so, uo, vo, zos]
VAR_WEIGHTS = [2.0, 1.5, 1.0, 1.0, 0.5]


def main():
    """Load config and data, initialise weighted-loss trainer, and run training.

    Args:
        None: All settings read from config YAML and CLI overrides.

    Returns:
        None: Writes model checkpoints and loss CSV as side effects.

    Example:
        >>> # python src/training/scripts/train_afno_weighted.py --device cuda:0
    """
    print("=== Variable-weighted Loss Training (E12) ===")
    print(f"  Loss weights: {dict(zip(['thetao','so','uo','vo','zos'], VAR_WEIGHTS))}")

    pipe = ConfigPipeline([
        YamlConfig('./afno_bob_surf_e12.yaml', config_name='default', config_folder='config/'),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder='config/'),
    ])
    config = pipe.read_conf()

    tee_logger = setup_logging(config)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(config.seed)

    log_experiment(config)

    try:
        if config.wb:
            import wandb
            wandb.init(project='AFNO for Bay of Bengal',
                       name=f"{config.name}_weighted", config=config)

        print("\n=== Loading Data ===")
        train_data_loader, val_data_loader, _, mask = load_and_prepare_data(config)
        print(f"  Train batches: {len(train_data_loader)}  "
              f"Val batches: {len(val_data_loader)}")

        print("\n=== Initialising Model ===")
        model = AFNONet(config).to(config.device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {total_params:,}")

        start_epoch   = 0
        best_val_loss = float('inf')
        checkpoint_data = None

        try:
            _cp = config.results.resume_checkpoint
        except (KeyError, AttributeError):
            _cp = None
        if _cp:
            cp_path = _cp
            if Path(cp_path).exists():
                print(f"\n=== Loading Checkpoint: {cp_path} ===")
                checkpoint_data = torch.load(cp_path, map_location=config.device,
                                             weights_only=False)
                start_epoch   = checkpoint_data['epoch'] + 1
                best_val_loss = checkpoint_data.get('best_val_loss', float('inf'))
                additional    = config.opt.epochs
                config.opt.epochs = start_epoch + additional
                print(f"  Resuming from epoch {start_epoch}")

        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=config.opt.lr,
                                      weight_decay=config.opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

        if checkpoint_data is not None:
            model.load_state_dict(checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])

        loss_fn = nn.L1Loss()

        trainer = Trainer(
            model=model, mask=mask, config=config,
            start_epoch=start_epoch, best_val_loss=best_val_loss,
            var_weights=VAR_WEIGHTS,
        )

        print(f"\n=== Training (var_weights={VAR_WEIGHTS}, "
              f"epochs={config.opt.epochs}) ===\n")
        trainer.train(train_data_loader, val_data_loader, loss_fn, optimizer, scheduler)

        if config.wb:
            import wandb
            wandb.finish()

    finally:
        cleanup_logging(tee_logger)


if __name__ == '__main__':
    main()

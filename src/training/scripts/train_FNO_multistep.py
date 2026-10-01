"""Train FNO ocean model with multi-step autoregressive rollout loss.

At each training iteration the model is unrolled autoregressively for
ROLLOUT_STEPS lead days: the ocean prediction from step t is detached and
concatenated with the ERA5 atmospheric forcing for t+1 before the next
model call. The training loss is the mean L1 loss over all rollout steps.

This directly penalises cumulative drift during autoregressive rollout —
unlike standard 1-step training (E03) — mirroring the AFNO RT (E14)
strategy but applied to the FNO architecture.

Inputs:
    --config_file (str): YAML config filename (default: fno_bob_surf_e04.yaml).
    --opt.epochs (int): Total training epochs (default from config).
    --opt.lr (float): Learning rate (default from config).
    --device (str): Torch device string (default from config).
    --results.resume_checkpoint (str): Path to .pt checkpoint for resumption.
    --name (str): Override experiment name.

Outputs:
    results/models/FNO_BoB_Surf_E04.pth          — best model weights.
    results/models/checkpoint_FNO_BoB_Surf_E04.pt — full checkpoint.
    results/experiments/logs/FNO_BoB_Surf_E04.log
    results/experiments/logs/FNO_BoB_Surf_E04_train_val_losses.csv

Example:
    conda activate BoB_Surf_2
    python src/training/scripts/train_FNO_multistep.py \\
        --config_file fno_bob_surf_e04.yaml \\
        --device cuda:0 \\
        --opt.epochs 250
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from neuralop.models import FNO
from data_pipeline.loaders.data_loader import load_and_prepare_data
from training.trainer import Trainer
from training.utils.experiment_logger import log_experiment, setup_logging, cleanup_logging

ROLLOUT_STEPS = 3


class FNOWrapper(nn.Module):
    """Wrapper around neuralop FNO to match the training pipeline interface.

    Handles input/output channel mapping and ensures compatibility with
    the existing data pipeline and Trainer.

    Attributes:
        fno (FNO): Underlying neuralop FNO model.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Args:
        config: Configuration object with fno and data sub-configs.

    Example:
        >>> model = FNOWrapper(config)
        >>> out = model(torch.randn(2, 11, 224, 224))
        >>> out.shape
        torch.Size([2, 5, 224, 224])
    """

    def __init__(self, config):
        """Initialise FNO from config."""
        super().__init__()
        non_linearity_map = {'gelu': F.gelu, 'relu': F.relu, 'tanh': torch.tanh}
        non_linearity = non_linearity_map.get(config.fno.non_linearity, F.gelu)

        self.fno = FNO(
            n_modes=tuple(config.fno.n_modes),
            in_channels=config.data.in_chs,
            out_channels=config.data.out_chs,
            hidden_channels=config.fno.hidden_channels,
            lifting_channel_ratio=config.fno.lifting_channel_ratio,
            projection_channel_ratio=config.fno.projection_channel_ratio,
            n_layers=config.fno.n_layers,
            non_linearity=non_linearity,
            use_channel_mlp=config.fno.use_channel_mlp,
            channel_mlp_expansion=config.fno.channel_mlp_expansion,
            channel_mlp_dropout=config.fno.channel_mlp_dropout,
            channel_mlp_skip=config.fno.channel_mlp_skip,
            fno_skip=config.fno.fno_skip,
            positional_embedding=config.fno.positional_embedding,
            norm=config.fno.norm,
        )
        self.in_channels = config.data.in_chs
        self.out_channels = config.data.out_chs

    def forward(self, x):
        """Forward pass through FNO.

        Args:
            x (torch.Tensor): Input tensor (batch, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor (batch, out_channels, H, W).

        Example:
            >>> out = model(torch.randn(2, 11, 224, 224).cuda())
        """
        return self.fno(x)


def main():
    """Load config and data, initialise multi-step FNO trainer, and run training.

    Args:
        None: All settings read from config YAML and CLI overrides.

    Returns:
        None: Writes model checkpoints and loss CSV as side effects.

    Example:
        >>> # python src/training/scripts/train_FNO_multistep.py --device cuda:0
    """
    print(f"=== Multi-step Rollout Training — FNO (rollout_steps={ROLLOUT_STEPS}) ===")

    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument('--config_file', default=None)
    _known, _ = _pre.parse_known_args()
    _exp_yaml = _known.config_file or 'fno_bob_surf_e04.yaml'

    pipe = ConfigPipeline([
        YamlConfig('./fno_bob_config.yaml', config_name='default', config_folder='config/'),
        YamlConfig(f'./{_exp_yaml}', config_name='default', config_folder='config/'),
        ArgparseConfig(infer_types=True, config_name='default', config_file=None),
    ])
    config = pipe.read_conf()

    tee_logger = setup_logging(config)

    print("\n=== Setting Random Seeds ===")
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(config.seed)

    print("\n=== Logging Experiment Configuration ===")
    log_experiment(config, csv_filename="runtable_FNO.csv")

    try:
        if config.wb:
            import wandb
            wandb.init(project='FNO for Bay of Bengal',
                       name=f"{config.name}_rollout{ROLLOUT_STEPS}", config=config)

        print(f"\n=== Loading Data (k_steps={ROLLOUT_STEPS}) ===")
        train_data_loader, val_data_loader, _, mask = load_and_prepare_data(
            config, k_steps=ROLLOUT_STEPS,
        )
        print(f"  Train batches: {len(train_data_loader)}  "
              f"Val batches: {len(val_data_loader)}")

        print("\n=== Initialising FNO Model ===")
        model = FNOWrapper(config).to(config.device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {total_params:,}")

        start_epoch = 0
        best_val_loss = float('inf')
        checkpoint_data = None

        try:
            _cp = config.results.resume_checkpoint
        except (KeyError, AttributeError):
            _cp = None
        if _cp and Path(_cp).exists():
            print(f"\n=== Loading Checkpoint: {_cp} ===")
            checkpoint_data = torch.load(_cp, map_location=config.device, weights_only=False)
            start_epoch = checkpoint_data['epoch'] + 1
            best_val_loss = checkpoint_data.get('best_val_loss', float('inf'))
            additional = config.opt.epochs
            config.opt.epochs = start_epoch + additional
            print(f"  Resuming from epoch {start_epoch}, training {additional} more epochs")

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
            rollout_steps=ROLLOUT_STEPS,
        )

        print(f"\n=== Training (rollout_steps={ROLLOUT_STEPS}, "
              f"epochs={config.opt.epochs}) ===\n")
        trainer.train(train_data_loader, val_data_loader, loss_fn, optimizer, scheduler)

        if config.wb:
            import wandb
            wandb.finish()

    finally:
        cleanup_logging(tee_logger)


if __name__ == '__main__':
    main()

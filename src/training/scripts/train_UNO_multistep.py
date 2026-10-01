"""Train UNO ocean model with multi-step autoregressive rollout loss.

At each training iteration the model is unrolled autoregressively for
ROLLOUT_STEPS lead days: the ocean prediction from step t is detached and
concatenated with the ERA5 atmospheric forcing for t+1 before the next
model call. The training loss is the mean L1 loss over all rollout steps.

This directly penalises cumulative drift during autoregressive rollout —
unlike standard 1-step training (E03) — mirroring the AFNO RT (E14)
strategy but applied to the UNO architecture.

Inputs:
    --config_file (str): YAML config filename (default: uno_bob_surf_e04.yaml).
    --opt.epochs (int): Total training epochs (default from config).
    --opt.lr (float): Learning rate (default from config).
    --device (str): Torch device string (default from config).
    --results.resume_checkpoint (str): Path to .pt checkpoint for resumption.
    --name (str): Override experiment name.

Outputs:
    results/models/UNO_BoB_Surf_E04.pth          — best model weights.
    results/models/checkpoint_UNO_BoB_Surf_E04.pt — full checkpoint.
    results/experiments/logs/UNO_BoB_Surf_E04.log
    results/experiments/logs/UNO_BoB_Surf_E04_train_val_losses.csv

Example:
    conda activate BoB_Surf_2
    python src/training/scripts/train_UNO_multistep.py \\
        --config_file uno_bob_surf_e04.yaml \\
        --device cuda:1 \\
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
from neuralop.models import UNO
from data_pipeline.loaders.data_loader import load_and_prepare_data
from training.trainer import Trainer
from training.utils.experiment_logger import log_experiment, setup_logging, cleanup_logging

ROLLOUT_STEPS = 3


class UNOWrapper(nn.Module):
    """Wrapper around neuralop UNO to match the training pipeline interface.

    Handles input/output channel mapping and ensures compatibility with
    the existing data pipeline and Trainer.

    Attributes:
        uno (UNO): Underlying neuralop UNO model.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Args:
        config: Configuration object with uno and data sub-configs.

    Example:
        >>> model = UNOWrapper(config)
        >>> out = model(torch.randn(2, 11, 224, 224))
        >>> out.shape
        torch.Size([2, 5, 224, 224])
    """

    def __init__(self, config):
        """Initialise UNO from config."""
        super().__init__()
        non_linearity_map = {'gelu': F.gelu, 'relu': F.relu, 'tanh': torch.tanh}
        non_linearity = non_linearity_map.get(config.uno.non_linearity, F.gelu)

        horizontal_skips_map = (dict(config.uno.horizontal_skips_map)
                                if config.uno.horizontal_skips_map else None)

        self.uno = UNO(
            in_channels=config.data.in_chs,
            out_channels=config.data.out_chs,
            hidden_channels=config.uno.hidden_channels,
            lifting_channels=config.uno.lifting_channels,
            projection_channels=config.uno.projection_channels,
            positional_embedding=config.uno.positional_embedding,
            n_layers=config.uno.n_layers,
            uno_out_channels=config.uno.uno_out_channels,
            uno_n_modes=[tuple(m) for m in config.uno.uno_n_modes],
            uno_scalings=[list(s) for s in config.uno.uno_scalings],
            horizontal_skips_map=horizontal_skips_map,
            channel_mlp_dropout=config.uno.channel_mlp_dropout,
            channel_mlp_expansion=config.uno.channel_mlp_expansion,
            non_linearity=non_linearity,
            norm=config.uno.norm,
            preactivation=config.uno.preactivation,
            fno_skip=config.uno.fno_skip,
            horizontal_skip=config.uno.horizontal_skip,
            channel_mlp_skip=config.uno.channel_mlp_skip,
            separable=config.uno.separable,
            factorization=config.uno.factorization,
            rank=config.uno.rank,
            domain_padding=config.uno.domain_padding,
        )
        self.in_channels = config.data.in_chs
        self.out_channels = config.data.out_chs

    def forward(self, x):
        """Forward pass through UNO.

        Args:
            x (torch.Tensor): Input tensor (batch, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor (batch, out_channels, H, W).

        Example:
            >>> out = model(torch.randn(2, 11, 224, 224).cuda())
        """
        return self.uno(x)


def main():
    """Load config and data, initialise multi-step UNO trainer, and run training.

    Args:
        None: All settings read from config YAML and CLI overrides.

    Returns:
        None: Writes model checkpoints and loss CSV as side effects.

    Example:
        >>> # python src/training/scripts/train_UNO_multistep.py --device cuda:1
    """
    print(f"=== Multi-step Rollout Training — UNO (rollout_steps={ROLLOUT_STEPS}) ===")

    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument('--config_file', default=None)
    _known, _ = _pre.parse_known_args()
    _exp_yaml = _known.config_file or 'uno_bob_surf_e04.yaml'

    pipe = ConfigPipeline([
        YamlConfig('./uno_bob_config.yaml', config_name='default', config_folder='config/'),
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
    log_experiment(config, csv_filename="runtable_UNO.csv")

    try:
        if config.wb:
            import wandb
            wandb.init(project='UNO for Bay of Bengal',
                       name=f"{config.name}_rollout{ROLLOUT_STEPS}", config=config)

        print(f"\n=== Loading Data (k_steps={ROLLOUT_STEPS}) ===")
        train_data_loader, val_data_loader, _, mask = load_and_prepare_data(
            config, k_steps=ROLLOUT_STEPS,
        )
        print(f"  Train batches: {len(train_data_loader)}  "
              f"Val batches: {len(val_data_loader)}")

        print("\n=== Initialising UNO Model ===")
        model = UNOWrapper(config).to(config.device)
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

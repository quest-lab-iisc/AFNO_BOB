"""Training script for the ConvLSTM ocean dynamics baseline.

Trains a stateful multi-layer ConvLSTM as a one-step-ahead predictor on the
same BoB ocean/atmosphere data as the AFNO experiments.  Training is identical
to AFNO: each iteration receives a single (input, target) pair; the ConvLSTM
hidden state is initialised to zeros at every step (stateless training).  At
inference time the hidden state is carried forward across the 9-day rollout.

Inputs:
    --config_file (str): YAML config filename in config/ (default: convlstm_bob_config.yaml).
    --name (str): Experiment name override.
    --device (str): PyTorch device string override (e.g. 'cuda:1').
    --opt.epochs (int): Number of training epochs.
    --opt.lr (float): Learning rate.
    --results.resume_checkpoint (str): Path to checkpoint .pt file to resume from.

Outputs:
    results/models/<name>.pth            Best model weights (inference use).
    results/models/checkpoint_<name>.pt  Full training checkpoint (resume use).
    results/experiments/logs/<name>.log  Verbatim console log with timestamps.
    results/experiments/runtable.csv     Appended run summary row.

Example:
    conda activate BoB_Surf_2
    python src/training/scripts/train_convlstm.py
    python src/training/scripts/train_convlstm.py --device cuda:1 --opt.epochs 200
    python src/training/scripts/train_convlstm.py \\
        --results.resume_checkpoint results/models/checkpoint_ConvLSTM_BoB_Surf_E01.pt
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from configmypy import ArgparseConfig, ConfigPipeline, YamlConfig
from data_pipeline.loaders.data_loader import load_and_prepare_data
from models.architectures.convlstm import ConvLSTMNet
from training.utils.experiment_logger import (
    TeeLogger, cleanup_logging, get_model_filename,
    log_experiment, setup_logging,
)


def build_model(config) -> ConvLSTMNet:
    """Instantiate ConvLSTMNet from config.

    Args:
        config: Configuration namespace with a ``convlstm`` sub-namespace.

    Returns:
        ConvLSTMNet: Model ready for training.

    Example:
        >>> model = build_model(config)
        >>> sum(p.numel() for p in model.parameters())
        8300165
    """
    c = config.convlstm
    return ConvLSTMNet(
        in_channels=c.in_channels,
        hidden_channels=c.hidden_channels,
        num_layers=c.num_layers,
        out_channels=c.out_channels,
        kernel_size=c.kernel_size,
    )


def save_checkpoint(model, optimizer, scheduler, epoch: int, best_val_loss: float,
                    path: str) -> None:
    """Save a full training checkpoint.

    Args:
        model (ConvLSTMNet): Model to save.
        optimizer: Optimizer state.
        scheduler: LR scheduler state.
        epoch (int): Current epoch number.
        best_val_loss (float): Best validation loss so far.
        path (str): Destination file path.

    Returns:
        None: Writes checkpoint to ``path``.

    Example:
        >>> save_checkpoint(model, opt, sched, 10, 0.05, 'results/models/ckpt.pt')
    """
    torch.save({
        'epoch':          epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss':  best_val_loss,
    }, path)


def train(config) -> None:
    """Run the ConvLSTM one-step-ahead training loop.

    Identical data loading to AFNO: each batch is a (input, target) 2-tuple.
    The hidden state is initialised to zeros for every batch (stateless
    training), so the model learns a one-step mapping just like AFNO.
    At inference time the hidden state is carried forward across leads.

    The land-sea mask is applied to ocean channels of both prediction and
    target before computing the loss, matching the AFNO training convention.

    Args:
        config: Parsed configuration namespace.

    Returns:
        None: Saves model weights and checkpoints to disk.

    Example:
        >>> train(config)
    """
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    # ----- Data (k_steps=1: standard 2-tuple batches, same as AFNO) -----------
    train_loader, val_loader, _, mask = load_and_prepare_data(config)
    mask = mask.to(device)

    # ----- Model ---------------------------------------------------------------
    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'ConvLSTMNet parameters: {n_params:,}')

    optimizer = optim.AdamW(model.parameters(),
                            lr=config.opt.lr,
                            weight_decay=config.opt.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=config.opt.patience,
                                  factor=config.opt.gamma, verbose=True)
    loss_fn = nn.MSELoss()

    # ----- Resume from checkpoint ----------------------------------------------
    start_epoch   = 0
    best_val_loss = float('inf')
    resume_path   = getattr(config.results, 'resume_checkpoint', None)

    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f'Resumed from {resume_path} (epoch {ckpt["epoch"]})')

    model_dir = Path(config.results.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / get_model_filename(config)
    ckpt_path = model_dir / f'checkpoint_{config.name}.pt'

    train_losses: list[float] = []
    val_losses:   list[float] = []

    # ----- Training loop -------------------------------------------------------
    total_epochs = start_epoch + config.opt.epochs
    for epoch in range(start_epoch, total_epochs):
        model.train()
        train_loss    = 0.0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]', file=sys.stdout,
                    dynamic_ncols=True)

        for inputs, targets in pbar:
            # inputs  : (B, 11, H, W)
            # targets : (B, 5, H, W)
            inputs  = inputs.to(device)
            targets = targets.to(device)

            # Fresh hidden state per batch — stateless one-step training
            pred, _ = model(inputs, hidden_state=None)

            pred_m   = pred    * mask
            target_m = targets * mask
            loss     = loss_fn(pred_m, target_m)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            B = inputs.shape[0]
            train_loss    += loss.item() * B
            total_samples += B
            pbar.set_postfix({'loss': f'{loss.item():.6f}',
                              'lr':   f'{optimizer.param_groups[0]["lr"]:.2e}'})

        train_loss /= total_samples
        train_losses.append(train_loss)

        # ----- Validation ------------------------------------------------------
        model.eval()
        val_loss  = 0.0
        total_val = 0

        with torch.no_grad():
            for val_input, val_target in tqdm(val_loader, desc=f'Epoch {epoch} [Val]',
                                              file=sys.stdout, dynamic_ncols=True,
                                              leave=False):
                val_input  = val_input.to(device)
                val_target = val_target.to(device)

                pred, _ = model(val_input, hidden_state=None)
                loss     = loss_fn(pred * mask, val_target * mask)

                val_loss  += loss.item() * val_input.shape[0]
                total_val += val_input.shape[0]

        val_loss /= total_val
        val_losses.append(val_loss)

        tqdm.write(f'Epoch {epoch:4d} | train {train_loss:.6f} | val {val_loss:.6f} '
                   f'| lr {optimizer.param_groups[0]["lr"]:.2e}')

        scheduler.step(val_loss)

        # ----- Save best model -------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict()}, str(best_path))
            tqdm.write(f'  ✓ Best model saved (val={best_val_loss:.6f})')

        # ----- Periodic checkpoint ---------------------------------------------
        if (epoch + 1) % config.results.checkpoint_frequency == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss,
                            str(ckpt_path))

    # Final checkpoint
    save_checkpoint(model, optimizer, scheduler, total_epochs - 1, best_val_loss,
                    str(ckpt_path))
    print(f'\nTraining complete. Best val loss: {best_val_loss:.6f}')
    print(f'Best model: {best_path}')
    print(f'Checkpoint: {ckpt_path}')

    # ----- Loss CSV -----------------------------------------------------------
    loss_csv = Path(config.results.save_dir) / config.name / 'losses.csv'
    loss_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(loss_csv, 'w') as f:
        f.write('epoch,train_loss,val_loss\n')
        for i, (tl, vl) in enumerate(zip(train_losses, val_losses)):
            f.write(f'{start_epoch + i},{tl:.8f},{vl:.8f}\n')
    print(f'Loss history: {loss_csv}')


def main() -> None:
    """Entry point: parse config, set up logging, run training.

    Example:
        python src/training/scripts/train_convlstm.py
        python src/training/scripts/train_convlstm.py --device cuda:1 --opt.epochs 200
    """
    pipe = ConfigPipeline([
        YamlConfig('./convlstm_bob_config.yaml', config_name='default',
                   config_folder='config/'),
        YamlConfig(config_folder='config/'),
        ArgparseConfig(),
    ])
    config = pipe.read_conf()

    tee_logger = setup_logging(config)
    try:
        print(f'=== ConvLSTM Training: {config.name} ===')
        print(f'Device: {config.device}  |  Epochs: {config.opt.epochs}  '
              f'|  Mode: one-step-ahead')
        log_experiment(config)
        train(config)

    except Exception as exc:
        import traceback
        print(f'\n[FATAL] {exc}')
        traceback.print_exc()
        raise
    finally:
        cleanup_logging(tee_logger)


if __name__ == '__main__':
    main()

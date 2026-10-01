"""
AFNO Coupled Atmospheric-Ocean Model - Training Script
=======================================================

Trains the Adaptive Fourier Neural Operator (AFNO) for coupled atmospheric
and ocean forecasting in the Bay of Bengal.

Input:  11 channels at time t  (6 atm + 5 ocean)
Output: 11 channels at time t+1 (6 atm + 5 ocean)

QUICK START
-----------
Training from scratch (uses config/afno_atmocean_config.yaml by default):
    python src/training/scripts/train_afno_atmocean.py

DEVICE SELECTION
----------------
Select a specific GPU or CPU at runtime:
    python src/training/scripts/train_afno_atmocean.py --device cuda:0
    python src/training/scripts/train_afno_atmocean.py --device cuda:1
    python src/training/scripts/train_afno_atmocean.py --device cpu

CONFIG FILE OVERRIDE
--------------------
Point to a different YAML config (must be in the config/ directory).
Values in the override file are layered on top of afno_atmocean_config.yaml;
anything not specified in the override falls back to the base config:
    python src/training/scripts/train_afno_atmocean.py --config_file my_experiment.yaml

Example override file (config/my_experiment.yaml):
    default:
      name: 'AFNO_AtmOcean_E02'
      opt:
        lr: 5e-4
        epochs: 200

PARAMETER OVERRIDES
-------------------
Any config key can be overridden directly on the command line:
    python src/training/scripts/train_afno_atmocean.py --opt.lr 5e-4
    python src/training/scripts/train_afno_atmocean.py --opt.epochs 200
    python src/training/scripts/train_afno_atmocean.py --name AFNO_AtmOcean_E02
    python src/training/scripts/train_afno_atmocean.py --wb True

RESUME FROM CHECKPOINT
-----------------------
opt.epochs means ADDITIONAL epochs when resuming (e.g. resume from epoch 50
with --opt.epochs 100 trains until epoch 150):
    python src/training/scripts/train_afno_atmocean.py \
        --results.resume_checkpoint results/models/checkpoint_AFNO_AtmOcean_E01.pt \
        --opt.epochs 100

EXAMPLE WORKFLOWS
-----------------
1. Fresh training on GPU 1 for 150 epochs with custom LR:
    python src/training/scripts/train_afno_atmocean.py \
        --device cuda:1 --opt.lr 5e-4 --opt.epochs 150 --name AFNO_AtmOcean_E02

2. Training with a custom config file on GPU 0:
    python src/training/scripts/train_afno_atmocean.py \
        --config_file my_experiment.yaml --device cuda:0

3. Resume from checkpoint and train 50 more epochs with W&B logging:
    python src/training/scripts/train_afno_atmocean.py \
        --results.resume_checkpoint results/models/checkpoint_AFNO_AtmOcean_E01.pt \
        --opt.epochs 50 --wb True

4. Resume on a different GPU with a different scheduler:
    python src/training/scripts/train_afno_atmocean.py \
        --results.resume_checkpoint results/models/checkpoint_AFNO_AtmOcean_E01.pt \
        --device cuda:1 --opt.scheduler CosineAnnealingLR

OUTPUTS
-------
    results/models/{name}.pth                        # best model weights (for inference)
    results/models/checkpoint_{name}.pt              # full checkpoint (for resuming)
    results/experiments/logs/{name}.log              # full training log
    results/experiments/runtable.csv                 # hyperparameter run table
    results/experiments/logs/{name}_train_val_losses.csv  # per-epoch losses
"""
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from models.architectures.afno.afnonet import AFNONet
from data_pipeline.loaders.data_loader_atmocean import load_and_prepare_data_atmocean
from training.trainer import Trainer
from training.utils.experiment_logger import log_experiment, setup_logging, cleanup_logging


def main():
    """Main training function"""

    print("=== Loading Configuration ===")
    import argparse as _ap, sys as _sys
    _pre = _ap.ArgumentParser(add_help=False)
    _pre.add_argument('--config_file', default=None)
    _known, _ = _pre.parse_known_args(_sys.argv[1:])
    _exp_yaml = _known.config_file or 'afno_atmocean_config.yaml'
    pipe = ConfigPipeline([
        YamlConfig("./afno_atmocean_config.yaml", config_name='default', config_folder='config/'),
        YamlConfig(f'./{_exp_yaml}', config_name='default', config_folder='config/'),
        ArgparseConfig(infer_types=True, config_name='default', config_file=None),
    ])
    config = pipe.read_conf()

    print("\n=== Setting Up Logging ===")
    tee_logger = setup_logging(config)

    print("\n=== Setting Random Seeds ===")
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(config.seed)

    print("\n=== Logging Experiment Configuration ===")
    log_experiment(config)

    try:
        if config.wb:
            import wandb
            name = (f"AFNO_AtmOcean_train_{config.data.train_years[0]}-"
                    f"{config.data.train_years[1]}_val_{config.data.validate_year}_"
                    f"lr_{config.opt.lr}")
            wandb.init(project='AFNO AtmOcean Bay of Bengal', name=name, config=config)

        # ---- Data ----
        print("\n=== Loading Data ===")
        train_loader, val_loader, test_loader, mask = load_and_prepare_data_atmocean(config)

        # ---- Model ----
        print("\n=== Initializing Model ===")
        model = AFNONet(config)
        model = model.to(config.device)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")

        try:
            from fvcore.nn import FlopCountAnalysis
            import time

            model.eval()
            dummy = torch.randn(1, config.data.in_chs, 224, 224).to(config.device)
            flops = FlopCountAnalysis(model, dummy)
            total_flops = flops.total() / 1e9
            print(f"Total GFLOPs: {total_flops:.2f}")

            H, W = 224, 224
            for _ in range(10):
                _ = model(dummy)

            num_runs = 100
            if torch.cuda.is_available() and "cuda" in str(config.device):
                starter = torch.cuda.Event(enable_timing=True)
                ender   = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                times = []
                for _ in range(num_runs):
                    starter.record()
                    _ = model(dummy)
                    ender.record()
                    torch.cuda.synchronize()
                    times.append(starter.elapsed_time(ender))
            else:
                import time as _time
                times = []
                for _ in range(num_runs):
                    t0 = _time.time()
                    _ = model(dummy)
                    times.append((_time.time() - t0) * 1000)

            avg_ms = sum(times) / len(times)
            print(f"Average inference time: {avg_ms:.3f} ms")
        except ImportError:
            print("FLOPs computation skipped (fvcore not installed)")

        # ---- Checkpoint resumption ----
        start_epoch   = 0
        best_val_loss = float('inf')
        checkpoint_data = None

        if config.results.resume_checkpoint:
            cp_path = config.results.resume_checkpoint
            if Path(cp_path).exists():
                print(f"\n=== Loading Checkpoint: {cp_path} ===")
                checkpoint_data = torch.load(cp_path, map_location=config.device,
                                             weights_only=False)
                start_epoch   = checkpoint_data['epoch'] + 1
                best_val_loss = checkpoint_data.get('best_val_loss', float('inf'))

                additional = config.opt.epochs
                config.opt.epochs = start_epoch + additional
                print(f"Resuming from epoch {start_epoch}, "
                      f"will train until epoch {config.opt.epochs - 1}")
            else:
                print(f"\n⚠ Warning: Checkpoint not found at {cp_path}. Starting fresh.")

        # ---- Optimizer & scheduler ----
        print("\n=== Setting up Optimizer and Scheduler ===")
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=config.opt.lr,
                                      weight_decay=config.opt.weight_decay)

        if config.opt.scheduler == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=config.opt.lr,
                steps_per_epoch=len(train_loader), epochs=config.opt.epochs)
        elif config.opt.scheduler == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        elif config.opt.scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.opt.scheduler_T_max)
        elif config.opt.scheduler == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma)
        else:
            raise ValueError(f"Unknown scheduler: {config.opt.scheduler}")

        if checkpoint_data is not None:
            print("\n=== Restoring Model and Optimizer States ===")
            model.load_state_dict(checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])

        loss_fn = nn.L1Loss()

        # ---- Train ----
        trainer = Trainer(model=model, mask=mask, config=config,
                          start_epoch=start_epoch, best_val_loss=best_val_loss)

        if config.verbose:
            print("\n### MODEL ###\n", model)
            print("\n### OPTIMIZER ###\n", optimizer)
            print("\n### SCHEDULER ###\n", scheduler)
            print("\n### Beginning Training ###\n")

        trainer.train(train_data_loader=train_loader, val_data_loader=val_loader,
                      loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler)

        if config.wb:
            wandb.finish()

        print("\n=== Training Complete ===")

    finally:
        cleanup_logging(tee_logger)


if __name__ == "__main__":
    main()

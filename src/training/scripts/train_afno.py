"""
AFNO Ocean Dynamics Model - Training Script
============================================

This script trains the Adaptive Fourier Neural Operator (AFNO) model for ocean dynamics
forecasting in the Bay of Bengal region. It supports both training from scratch and
resuming from checkpoints.

QUICK START
-----------
Training from scratch:
    python src/training/scripts/train_afno.py

Resume from checkpoint:
    python src/training/scripts/train_afno.py

The script uses configuration files to control all training parameters.

CONFIGURATION
-------------
Training is controlled via YAML configuration files located in the config/ directory.
The default configuration is loaded from config/afno_bob_config.yaml.

Key configuration sections:
    - General Settings: name, device, seed, verbose, wandb logging
    - Model Architecture (afno2d): hidden_size, num_blocks, attention heads
    - Optimizer (opt): learning rate, epochs, scheduler, weight decay
    - Data (data): data paths, variables, year splits, batch size
    - Results (results): save directories, checkpoint settings

Override config file:
    python src/training/scripts/train_afno.py --config_file my_config.yaml

Override individual parameters:
    python src/training/scripts/train_afno.py --opt.lr 5e-4 --opt.epochs 200

TRAINING FROM SCRATCH
---------------------
1. Configure your experiment:
   - Edit config/afno_bob_config.yaml
   - Set experiment name: `name: 'AFNO_MyExperiment'`
   - Set device: `device: 'cuda:0'` or `'cpu'`
   - Configure data paths: `data.data_dir` and `data.mean_dir`
   - Set training years: `data.train_years: [1993, 2018]`
   - Set validation year: `data.validate_year: 2019`
   - Set test years: `data.test_years: [2020]`
   - Configure optimizer: `opt.lr`, `opt.epochs`, `opt.scheduler`

2. Ensure checkpoint resumption is disabled:
   - Set `results.resume_checkpoint: null` or comment it out

3. Run training:
   python src/training/scripts/train_afno.py

4. Monitor training:
   - Console output shows epoch progress, losses, and metrics
   - Logs saved to: results/logs/{experiment_name}_train.log
   - Checkpoints saved to: results/models/
   - Enable Weights & Biases tracking: set `wb: True`

RESUMING FROM CHECKPOINT
-------------------------
1. Locate your checkpoint file:
   - Checkpoints are saved in results/models/
   - Format: checkpoint_{experiment_name}.pt or checkpoint_epoch_{N}.pt
   - Contains: model weights, optimizer state, scheduler state, epoch number

2. Configure checkpoint path:
   - Edit config/afno_bob_config.yaml
   - Set `results.resume_checkpoint: 'results/models/checkpoint_AFNO_MyExp.pt'`

   OR override via command line:
   python src/training/scripts/train_afno.py --results.resume_checkpoint results/models/checkpoint_epoch_50.pt

3. Set number of ADDITIONAL epochs to train:
   - The `opt.epochs` parameter specifies ADDITIONAL epochs when resuming
   - Example: Resuming from epoch 50 with `opt.epochs: 100` will train until epoch 150
   - For fresh training, `opt.epochs` is the total number of epochs

4. Run training:
   python src/training/scripts/train_afno.py

5. The script will:
   - Load model weights from checkpoint
   - Restore optimizer and scheduler states
   - Resume from the next epoch after the checkpoint
   - Train for the specified number of additional epochs
   - Preserve the best validation loss for model selection
   - Continue saving checkpoints as configured

CHECKPOINT CONFIGURATION
------------------------
Control checkpoint behavior in the config file:

    results:
        save_dir: "results/"
        model_dir: "results/models/"
        checkpoint_frequency: 5          # Save every N epochs
        save_best_only: True             # Only save if validation loss improves
        resume_checkpoint: null          # Path to checkpoint or null

- checkpoint_frequency: How often to save checkpoints (in epochs)
- save_best_only: If True, only saves when validation loss is the best so far
- resume_checkpoint: Path to checkpoint file for resuming (null for fresh start)

CHECKPOINT STRUCTURE
--------------------
Each checkpoint (.pt file) contains:
    - model_state_dict: Model weights and biases
    - optimizer_state_dict: Optimizer state (momentum, learning rates, etc.)
    - scheduler_state_dict: Learning rate scheduler state
    - epoch: Last completed epoch number
    - best_val_loss: Best validation loss achieved so far
    - config: Full configuration used for training

DATA REQUIREMENTS
-----------------
Required directory structure:
    data/
        ocean_1993.nc, ocean_1994.nc, ...  # Ocean variable files
        atm_1993.nc, atm_1994.nc, ...      # Atmospheric forcing files
        mean/
            thetao_mean.nc                  # Normalization statistics
            so_mean.nc
            ... (for each variable)

Variables:
    Ocean: thetao (temperature), so (salinity), uo, vo (currents), zos (sea level)
    Atmospheric: ssr, tp, u10, v10, msl, tcc

EXAMPLE WORKFLOWS
-----------------
1. Fresh training with custom learning rate for 150 epochs:
   python src/training/scripts/train_afno.py --opt.lr 5e-4 --opt.epochs 150 --name AFNO_LR5e4

2. Resume from epoch 50 and train for 100 additional epochs (until epoch 150):
   python src/training/scripts/train_afno.py --results.resume_checkpoint results/models/checkpoint_epoch_50.pt --opt.epochs 100

3. Resume and train for 50 more epochs with modified scheduler:
   python src/training/scripts/train_afno.py --results.resume_checkpoint results/models/checkpoint.pt --opt.epochs 50 --opt.scheduler CosineAnnealingLR

4. Resume training on different GPU:
   python src/training/scripts/train_afno.py --results.resume_checkpoint results/models/checkpoint.pt --device cuda:1

5. Resume with Weights & Biases logging:
   python src/training/scripts/train_afno.py --results.resume_checkpoint results/models/checkpoint.pt --wb True

OUTPUTS
-------
During and after training, the following outputs are generated:

    results/
        models/
            checkpoint_{name}.pt           # Latest checkpoint
            best_model_{name}.pt           # Best model by validation loss
        logs/
            {name}_train.log               # Training log file
        plots/                             # Training visualizations (if enabled)
        tables/                            # Metrics tables (if enabled)

TROUBLESHOOTING
---------------
1. "Checkpoint file not found":
   - Verify the path in results.resume_checkpoint
   - Use absolute path or path relative to project root

2. "CUDA out of memory":
   - Reduce batch_size in config: data.batch_size
   - Use gradient accumulation (modify trainer if needed)
   - Use smaller model: reduce afno2d.hidden_size or afno2d.num_blocks

3. "Configuration mismatch" when resuming:
   - Ensure model architecture params match checkpoint
   - You can change optimizer/scheduler params when resuming
   - Cannot change model structure (hidden_size, num_blocks, etc.)

4. Training diverges or NaN losses:
   - Lower learning rate: opt.lr
   - Increase weight_decay: opt.weight_decay
   - Check data normalization statistics
   - Verify data quality (no NaN or inf values)

NOTES
-----
- IMPORTANT: opt.epochs means ADDITIONAL epochs when resuming from checkpoint
  - Fresh training: opt.epochs = total epochs to train
  - Resuming: opt.epochs = additional epochs to train from checkpoint
  - Example: Resume from epoch 50 with opt.epochs=100 trains until epoch 150
- Set deterministic behavior with seed for reproducibility
- Model checkpoints can be large (>100MB) depending on architecture
- Training logs are automatically tee'd to both console and log file
- Press Ctrl+C to gracefully interrupt training (checkpoint saved)
- The trainer automatically handles learning rate scheduling
- Validation is performed after each epoch
- Best model is tracked by validation loss

For more information, see:
    - config/afno_bob_config.yaml: Full configuration reference
    - src/training/trainer.py: Trainer implementation details
    - src/models/architectures/afno/: AFNO model architecture
"""
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from models.architectures.afno.afnonet import AFNONet
from data_pipeline.loaders.data_loader import load_and_prepare_data
from training.trainer import Trainer
from training.utils.experiment_logger import log_experiment, setup_logging, cleanup_logging


def main():
    """Main training function"""

    # Load configuration
    print("=== Loading Configuration ===")
    config_name = "default"
    import argparse as _ap, sys as _sys
    _pre = _ap.ArgumentParser(add_help=False)
    _pre.add_argument('--config_file', default=None)
    _known, _ = _pre.parse_known_args(_sys.argv[1:])
    _exp_yaml = _known.config_file or 'afno_bob_config.yaml'
    pipe = ConfigPipeline([
        YamlConfig("./afno_bob_config.yaml", config_name='default', config_folder='config/'),
        YamlConfig(f'./{_exp_yaml}', config_name='default', config_folder='config/'),
        ArgparseConfig(infer_types=True, config_name='default', config_file=None)
    ])
    config = pipe.read_conf()

    # Set up logging to file
    print("\n=== Setting Up Logging ===")
    tee_logger = setup_logging(config)

    # Set random seeds for reproducibility
    print("\n=== Setting Random Seeds ===")
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(config.seed)

    # Log experiment configuration
    print("\n=== Logging Experiment Configuration ===")
    log_experiment(config)

    try:
        # Wandb initialization
        if config.wb:
            import wandb
            name = f"AFNO_BOB_train_{config.data.train_years[0]}-{config.data.train_years[1]}_" \
                   f"val_{config.data.validate_year}_test_{config.data.test_years}_lr_{config.opt.lr}"
            wandb.init(
                project='AFNO for Bay of Bengal',
                name=name,
                config=config
            )

        # Load data
        print("\n=== Loading Data ===")
        train_data_loader, val_data_loader, test_dataloader, mask = load_and_prepare_data(config)

        # Initialize model
        print("\n=== Initializing Model ===")
        model = AFNONet(config)
        model = model.to(config.device)

        # Count parameters
        total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")

        # FLOPs and inference time computation (optional)
        try:
            from fvcore.nn import FlopCountAnalysis
            import time

            input_tensor = torch.randn(1, config.data.in_chs, 224, 224).to(config.device)
            model.eval()

            flops = FlopCountAnalysis(model, input_tensor)
            total_flops = flops.total() / 1e9
            print(f"Total GFLOPs: {total_flops:.2f}")

            _, _, H, W = input_tensor.shape
            num_grid_points = H * W
            flops_per_grid_point = total_flops / num_grid_points
            print(f"GFLOPs per grid point: {flops_per_grid_point:.6f}")

            # Inference time
            for _ in range(10):
                _ = model(input_tensor)

            num_runs = 100
            if torch.cuda.is_available() and "cuda" in str(config.device):
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                times = []
                for _ in range(num_runs):
                    starter.record()
                    _ = model(input_tensor)
                    ender.record()
                    torch.cuda.synchronize()
                    times.append(starter.elapsed_time(ender))
                avg_inference_time = sum(times) / len(times)
            else:
                times = []
                for _ in range(num_runs):
                    start = time.time()
                    _ = model(input_tensor)
                    end = time.time()
                    times.append((end - start) * 1000)
                avg_inference_time = sum(times) / len(times)

            print(f"Average inference time: {avg_inference_time:.3f} ms")
            print(f"Inference time per grid point: {avg_inference_time / num_grid_points:.6f} ms")
        except ImportError:
            print("FLOPs computation skipped (fvcore not installed)")

        # Check if resuming from checkpoint and adjust epochs accordingly
        start_epoch = 0
        best_val_loss = float('inf')
        checkpoint_data = None

        if config.results.resume_checkpoint:
            checkpoint_path = config.results.resume_checkpoint
            if Path(checkpoint_path).exists():
                print(f"\n=== Loading Checkpoint: {checkpoint_path} ===")
                checkpoint_data = torch.load(checkpoint_path, map_location=config.device, weights_only=False)

                start_epoch = checkpoint_data['epoch'] + 1
                best_val_loss = checkpoint_data.get('best_val_loss', float('inf'))

                print(f"Checkpoint loaded from epoch {checkpoint_data['epoch']}")
                print(f"Previous best validation loss: {best_val_loss:.6f}")

                # Adjust total epochs: start_epoch + additional epochs from config
                additional_epochs = config.opt.epochs
                config.opt.epochs = start_epoch + additional_epochs
                print(f"Will train for {additional_epochs} additional epochs (epochs {start_epoch} to {config.opt.epochs-1})")
            else:
                print(f"\n⚠ Warning: Checkpoint file not found at {checkpoint_path}")
                print("Starting training from scratch")

        # Create optimizer
        print("\n=== Setting up Optimizer and Scheduler ===")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.opt.lr,
            weight_decay=config.opt.weight_decay,
        )

        # Create scheduler
        if config.opt.scheduler == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config.opt.lr,
                steps_per_epoch=len(train_data_loader),
                epochs=config.opt.epochs,
            )
        elif config.opt.scheduler == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
            )
        elif config.opt.scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.opt.scheduler_T_max
            )
        elif config.opt.scheduler == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
            )
        else:
            raise ValueError(f"Unknown scheduler: {config.opt.scheduler}")

        # Load checkpoint states if resuming
        if checkpoint_data is not None:
            print(f"\n=== Restoring Model and Optimizer States ===")
            model.load_state_dict(checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            print(f"Resuming training from epoch {start_epoch}")

        # Loss function
        loss_fn = nn.L1Loss()  # or nn.MSELoss()

        # Initialize trainer
        trainer = Trainer(model=model, mask=mask, config=config, start_epoch=start_epoch, best_val_loss=best_val_loss)

        if config.verbose:
            print("\n### MODEL ###\n", model)
            print("\n### OPTIMIZER ###\n", optimizer)
            print("\n### SCHEDULER ###\n", scheduler)
            print("\n### Beginning Training ###\n")

        # Train
        trainer.train(
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler
        )

        # Finish wandb
        if config.wb:
            wandb.finish()

        print("\n=== Training Complete ===")

    finally:
        # Clean up logging
        cleanup_logging(tee_logger)


if __name__ == "__main__":
    main()

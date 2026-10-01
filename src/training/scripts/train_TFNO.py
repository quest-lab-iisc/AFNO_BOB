"""
TFNO Ocean Dynamics Model - Training Script
============================================

This script trains the Tensorized Fourier Neural Operator (TFNO) model for ocean dynamics
forecasting in the Bay of Bengal region. TFNO uses Tucker factorization to reduce parameters
while maintaining performance. It supports both training from scratch and resuming from checkpoints.

QUICK START
-----------
Training from scratch:
    python src/training/scripts/train_TFNO.py

Resume from checkpoint:
    python src/training/scripts/train_TFNO.py

The script uses configuration files to control all training parameters.

CONFIGURATION
-------------
Training is controlled via YAML configuration files located in the config/ directory.
The default configuration is loaded from config/tfno_bob_config.yaml.

Key configuration sections:
    - General Settings: name, device, seed, verbose, wandb logging
    - Model Architecture (tfno): n_modes, hidden_channels, n_layers, factorization, rank
    - Optimizer (opt): learning rate, epochs, scheduler, weight decay
    - Data (data): data paths, variables, year splits, batch size
    - Results (results): save directories, checkpoint settings

Override config file:
    python src/training/scripts/train_TFNO.py --config_file my_config.yaml

Override individual parameters:
    python src/training/scripts/train_TFNO.py --opt.lr 5e-4 --opt.epochs 200

TRAINING FROM SCRATCH
---------------------
1. Configure your experiment:
   - Edit config/tfno_bob_config.yaml
   - Set experiment name: `name: 'TFNO_MyExperiment'`
   - Set device: `device: 'cuda:0'` or `'cpu'`
   - Configure data paths: `data.data_dir` and `data.mean_dir`
   - Set training years: `data.train_years: [1993, 2018]`
   - Set validation year: `data.validate_year: 2019`
   - Set test years: `data.test_years: [2020]`
   - Configure optimizer: `opt.lr`, `opt.epochs`, `opt.scheduler`

2. Ensure checkpoint resumption is disabled:
   - Set `results.resume_checkpoint: null` or comment it out

3. Run training:
   python src/training/scripts/train_TFNO.py

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
   - Edit config/tfno_bob_config.yaml
   - Set `results.resume_checkpoint: 'results/models/checkpoint_TFNO_MyExp.pt'`

   OR override via command line:
   python src/training/scripts/train_TFNO.py --results.resume_checkpoint results/models/checkpoint_epoch_50.pt

3. Set number of ADDITIONAL epochs to train:
   - The `opt.epochs` parameter specifies ADDITIONAL epochs when resuming
   - Example: Resuming from epoch 50 with `opt.epochs: 100` will train until epoch 150
   - For fresh training, `opt.epochs` is the total number of epochs

4. Run training:
   python src/training/scripts/train_TFNO.py

5. The script will:
   - Load model weights from checkpoint
   - Restore optimizer and scheduler states
   - Resume from the next epoch after the checkpoint
   - Train for the specified number of additional epochs
   - Preserve the best validation loss for model selection
   - Continue saving checkpoints as configured

TFNO MODEL ARCHITECTURE
------------------------
The TFNO model uses Tucker factorization of FNO weights for parameter efficiency:

Key parameters:
    - n_modes: Number of Fourier modes to keep per dimension (e.g., [16, 16] for 2D)
    - hidden_channels: Width of the hidden layers
    - n_layers: Number of TFNO layers (depth of the network)
    - factorization: Tensor factorization method (default: 'Tucker')
    - rank: Factorization rank (0.1 = 10% of dense FNO parameters, 0.5 = 50%)
    - non_linearity: Activation function (gelu, relu, tanh)

The model takes input shape (batch, in_channels, height, width) and outputs
(batch, out_channels, height, width).

TFNO vs FNO:
    - TFNO with rank=0.1 has ~10% of FNO parameters
    - TFNO with rank=0.5 has ~50% of FNO parameters
    - Lower rank = fewer parameters but may reduce accuracy
    - Tucker factorization enables efficient forward pass

EXAMPLE WORKFLOWS
-----------------
1. Fresh training with custom learning rate for 150 epochs:
   python src/training/scripts/train_TFNO.py --opt.lr 5e-4 --opt.epochs 150 --name TFNO_LR5e4

2. Resume from epoch 50 and train for 100 additional epochs (until epoch 150):
   python src/training/scripts/train_TFNO.py --results.resume_checkpoint results/models/checkpoint_epoch_50.pt --opt.epochs 100

3. Resume and train for 50 more epochs with modified scheduler:
   python src/training/scripts/train_TFNO.py --results.resume_checkpoint results/models/checkpoint.pt --opt.epochs 50 --opt.scheduler CosineAnnealingLR

4. Resume training on different GPU:
   python src/training/scripts/train_TFNO.py --results.resume_checkpoint results/models/checkpoint.pt --device cuda:1

5. Train with different factorization rank (more parameters):
   python src/training/scripts/train_TFNO.py --tfno.rank 0.5 --tfno.hidden_channels 128

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
   - Reduce model size: tfno.hidden_channels or tfno.n_modes
   - Reduce rank (already should be small with Tucker factorization)

3. "Configuration mismatch" when resuming:
   - Ensure model architecture params match checkpoint
   - You can change optimizer/scheduler params when resuming
   - Cannot change model structure (n_modes, hidden_channels, rank, etc.)

4. Training diverges or NaN losses:
   - Lower learning rate: opt.lr
   - Increase weight_decay: opt.weight_decay
   - Try increasing rank: tfno.rank (may be too compressed)
   - Check data normalization statistics
   - Verify data quality (no NaN or inf values)

NOTES
-----
- IMPORTANT: opt.epochs means ADDITIONAL epochs when resuming from checkpoint
  - Fresh training: opt.epochs = total epochs to train
  - Resuming: opt.epochs = additional epochs to train from checkpoint
  - Example: Resume from epoch 50 with opt.epochs=100 trains until epoch 150
- Set deterministic behavior with seed for reproducibility
- Model checkpoints are smaller than FNO due to factorization
- Training logs are automatically tee'd to both console and log file
- Press Ctrl+C to gracefully interrupt training (checkpoint saved)
- The trainer automatically handles learning rate scheduling
- Validation is performed after each epoch
- Best model is tracked by validation loss

For more information, see:
    - config/tfno_bob_config.yaml: Full configuration reference
    - src/training/trainer.py: Trainer implementation details
    - NeuralOperator documentation: https://neuraloperator.github.io/
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from neuralop.models import TFNO
from data_pipeline.loaders.data_loader import load_and_prepare_data
from training.trainer import Trainer
from training.utils.experiment_logger import log_experiment, setup_logging, cleanup_logging



class TFNOWrapper(nn.Module):
    """
    Wrapper around neuralop TFNO to match the interface expected by the training pipeline.

    TFNO (Tensorized FNO) uses Tucker factorization to reduce the number of parameters
    while maintaining performance. It's essentially an FNO with factorization='Tucker'
    and a specified rank (default 0.1 = 10% of dense FNO parameters).

    The wrapper handles input/output channel mapping and ensures compatibility with
    the existing data pipeline and trainer.
    """
    def __init__(self, config):
        super().__init__()

        # Extract TFNO configuration
        n_modes = tuple(config.tfno.n_modes)
        hidden_channels = config.tfno.hidden_channels
        lifting_channel_ratio = config.tfno.lifting_channel_ratio
        projection_channel_ratio = config.tfno.projection_channel_ratio
        n_layers = config.tfno.n_layers
        factorization = config.tfno.factorization
        rank = config.tfno.rank

        # Determine non-linearity function
        if config.tfno.non_linearity == 'gelu':
            non_linearity = F.gelu
        elif config.tfno.non_linearity == 'relu':
            non_linearity = F.relu
        elif config.tfno.non_linearity == 'tanh':
            non_linearity = torch.tanh
        else:
            non_linearity = F.gelu  # default

        # Initialize TFNO model from neuralop
        # TFNO is FNO with Tucker factorization by default
        self.tfno = TFNO(
            n_modes=n_modes,
            in_channels=config.data.in_chs,
            out_channels=config.data.out_chs,
            hidden_channels=hidden_channels,
            lifting_channel_ratio=lifting_channel_ratio,
            projection_channel_ratio=projection_channel_ratio,
            n_layers=n_layers,
            factorization=factorization,
            rank=rank,
            non_linearity=non_linearity,
            use_channel_mlp=config.tfno.use_channel_mlp,
            channel_mlp_expansion=config.tfno.channel_mlp_expansion,
            channel_mlp_dropout=config.tfno.channel_mlp_dropout,
            channel_mlp_skip=config.tfno.channel_mlp_skip,
            fno_skip=config.tfno.fno_skip,
            positional_embedding=config.tfno.positional_embedding,
            norm=config.tfno.norm,
        )

        self.in_channels = config.data.in_chs
        self.out_channels = config.data.out_chs
        self.rank = rank
        self.factorization = factorization

    def forward(self, x):
        """
        Forward pass through TFNO.

        Args:
            x: Input tensor of shape (batch, in_channels, height, width)

        Returns:
            Output tensor of shape (batch, out_channels, height, width)
        """
        return self.tfno(x)


def main():
    """Main training function"""

    # Load configuration
    print("=== Loading Configuration ===")
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument('--config_file', default=None)
    _known, _ = _pre.parse_known_args()
    _exp_yaml = _known.config_file or 'tfno_bob_config.yaml'
    pipe = ConfigPipeline([
        YamlConfig("./tfno_bob_config.yaml", config_name='default', config_folder='config/'),
        YamlConfig(f'./{_exp_yaml}', config_name='default', config_folder='config/'),
        ArgparseConfig(infer_types=True, config_name='default', config_file=None),
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
    log_experiment(config, csv_filename="runtable_TFNO.csv")

    try:
        # Wandb initialization
        if config.wb:
            import wandb
            name = f"TFNO_BOB_train_{config.data.train_years[0]}-{config.data.train_years[1]}_" \
                   f"val_{config.data.validate_year}_test_{config.data.test_years}_lr_{config.opt.lr}_rank_{config.tfno.rank}"
            wandb.init(
                project='TFNO for Bay of Bengal',
                name=name,
                config=config
            )

        # Load data
        print("\n=== Loading Data ===")
        train_data_loader, val_data_loader, test_dataloader, mask = load_and_prepare_data(config)

        # Initialize model
        print("\n=== Initializing TFNO Model ===")
        model = TFNOWrapper(config)
        model = model.to(config.device)

        # Count parameters
        total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")
        print(f"Factorization method: {config.tfno.factorization}")
        print(f"Factorization rank: {config.tfno.rank} (approx. {config.tfno.rank*100:.0f}% of dense FNO)")

        # Combined FLOPs and timing analysis
        try:
            import time
            print("\n=== Model Complexity Analysis ===")

            H, W = config.data.img_size[0], config.data.img_size[1]
            num_grid_points = H * W

            # 1. Theoretical FLOPs (TFNO-specific with factorization)
            n_modes = config.tfno.n_modes[0]
            hidden_channels = config.tfno.hidden_channels
            n_layers = config.tfno.n_layers
            in_channels = config.data.in_chs
            out_channels = config.data.out_chs
            rank = config.tfno.rank

            # Lifting layer: pointwise convolution (1x1 conv)
            lifting_flops = H * W * in_channels * hidden_channels * 2

            # TFNO layers with Tucker factorization
            # Tucker factorization reduces weight matrix operations
            # Approximate reduction factor based on rank
            factorization_efficiency = rank

            fno_layer_flops = n_layers * (
                10 * H * W * np.log2(H * W) * hidden_channels +  # FFT + IFFT (approximation)
                n_modes * n_modes * hidden_channels * hidden_channels * 2 * factorization_efficiency +  # Factorized spectral conv
                H * W * hidden_channels * hidden_channels * 2 +  # Skip connection
                H * W * hidden_channels  # Activation function
            )

            # Projection layer: pointwise convolution (1x1 conv)
            projection_flops = H * W * hidden_channels * out_channels * 2

            # Total FLOPs
            total_flops = lifting_flops + fno_layer_flops + projection_flops
            gflops = total_flops / 1e9

            print(f"\n--- Theoretical FLOPs Breakdown (with Tucker factorization) ---")
            print(f"Lifting layer:     {lifting_flops / 1e9:>8.2f} GFLOPs")
            print(f"TFNO layers (x{n_layers}):  {fno_layer_flops / 1e9:>8.2f} GFLOPs (rank={rank})")
            print(f"Projection layer:  {projection_flops / 1e9:>8.2f} GFLOPs")
            print(f"{'=' * 40}")
            print(f"Total:             {gflops:>8.2f} GFLOPs")
            print(f"Per grid point:    {total_flops / num_grid_points:>8.2f} FLOPs")
            print(f"\nNote: Tucker factorization reduces computational cost by ~{(1-rank)*100:.0f}%")

            # 2. Inference timing
            print(f"\n--- Inference Time Measurement ---")
            model.eval()
            input_tensor = torch.randn(1, in_channels, H, W).to(config.device)

            # Warmup
            print("Warming up (10 runs)...")
            with torch.no_grad():
                for _ in range(10):
                    _ = model(input_tensor)

            # Timing
            num_runs = 100
            print(f"Measuring inference time ({num_runs} runs)...")

            if torch.cuda.is_available() and "cuda" in str(config.device):
                torch.cuda.synchronize()
                start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
                end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

                with torch.no_grad():
                    for i in range(num_runs):
                        start_events[i].record()
                        _ = model(input_tensor)
                        end_events[i].record()

                torch.cuda.synchronize()
                times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            else:
                times = []
                with torch.no_grad():
                    for _ in range(num_runs):
                        start = time.perf_counter()
                        _ = model(input_tensor)
                        end = time.perf_counter()
                        times.append((end - start) * 1000)

            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)

            print(f"\nInference time:        {avg_time:>8.3f} ± {std_time:.3f} ms")
            print(f"Min/Max time:          {min_time:>8.3f} / {max_time:.3f} ms")
            print(f"Time per grid point:   {avg_time / num_grid_points * 1000:>8.3f} μs")
            print(f"Throughput:            {1000 / avg_time:>8.2f} FPS")

            if avg_time > 0:
                achieved_gflops = gflops / (avg_time / 1000)
                print(f"Achieved performance:  {achieved_gflops:>8.2f} GFLOP/s")

            print(f"{'=' * 40}\n")

            # Set model back to train mode
            model.train()

        except Exception as e:
            print(f"⚠ Performance analysis failed: {e}")
            import traceback
            traceback.print_exc()
            # Ensure model is in train mode even if analysis fails
            model.train()

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

"""Training orchestration for AFNO ocean dynamics prediction.

Supports single-step and multi-step autoregressive rollout training, as well
as optional per-variable loss weighting.

Inputs:
    model (torch.nn.Module): AFNO model to train.
    mask (torch.Tensor): Land-sea mask, shape (1, 1, 224, 224).
    config: Configuration object.
    rollout_steps (int): Number of autoregressive steps to unroll during
        training (default 1 — standard single-step training).  When > 1 the
        training DataLoader must yield 3-tuples from NetCDFDataset(k_steps>1).
    var_weights (list[float] | None): Per-variable loss multipliers, length
        equal to the number of ocean output channels (default None — equal
        weights).  Applied as elementwise multiplication before the L1 loss.

Outputs:
    None: Writes .pth and .pt checkpoint files; appends loss CSV.

Example:
    >>> trainer = Trainer(model, mask, config, rollout_steps=3,
    ...                   var_weights=[2.0, 1.5, 1.0, 1.0, 0.5])
    >>> trainer.train(train_loader, val_loader, loss_fn, optimizer, scheduler)
"""
import os
import csv
import torch
from tqdm import tqdm
from training.utils.experiment_logger import get_model_filename


class Trainer:
    """Trainer for the AFNO ocean dynamics model.

    Attributes:
        rollout_steps (int): Autoregressive steps per training iteration.
        var_weight (torch.Tensor | None): Per-variable loss weights,
            shape (1, out_ch, 1, 1), or None for equal weighting.

    Args:
        model (torch.nn.Module): Model to train.
        mask (torch.Tensor): Land-sea mask.
        config: Configuration object.
        data_processor: Unused legacy argument (kept for API compatibility).
        start_epoch (int): First epoch index (for resumption).
        best_val_loss (float): Baseline validation loss (for resumption).
        rollout_steps (int): Autoregressive rollout horizon (default 1).
        var_weights (list[float] | None): Per-output-channel loss multipliers.

    Example:
        >>> trainer = Trainer(model, mask, config, rollout_steps=3)
        >>> trainer.train(train_loader, val_loader, nn.L1Loss(), optimizer, scheduler)
    """

    def __init__(self, model, mask, config, data_processor=None, start_epoch=0,
                 best_val_loss=float('inf'), rollout_steps: int = 1,
                 var_weights=None):
        """Initialise trainer and build the optional variable-weight tensor."""
        self.model = model
        self.n_epochs = config.opt.epochs
        self.verbose = config.verbose
        self.device = config.device
        self.mask = mask
        self.min_epoch = config.min_epoch
        self.variable = config.data.variable
        self.config = config
        self.start_epoch = start_epoch
        self.best_val_loss = best_val_loss
        self.rollout_steps = max(1, int(rollout_steps))
        # Number of atmospheric channels (0 for ocean-only models)
        self.n_atm = getattr(config.data, 'atm_chs', 0)
        # Build per-variable loss weight tensor (1, out_ch, 1, 1)
        if var_weights is not None:
            w = torch.tensor(var_weights, dtype=torch.float32)
            self.var_weight = w[None, :, None, None]   # broadcast over batch, H, W
        else:
            self.var_weight = None

    def _apply_mask_and_weights(self, pred: torch.Tensor, target: torch.Tensor,
                                 mask: torch.Tensor):
        """Apply land-sea mask (and optional variable weights) to prediction and target.

        Args:
            pred (torch.Tensor): Model output, shape (B, out_ch, H, W).
            target (torch.Tensor): Ground truth, shape (B, out_ch, H, W).
            mask (torch.Tensor): Land-sea mask, shape (1, 1, H, W).

        Returns:
            tuple: (pred_masked, target_masked) both (B, out_ch, H, W).

        Example:
            >>> pm, tm = trainer._apply_mask_and_weights(pred, target, mask)
        """
        if self.n_atm > 0:
            # Coupled model: atm channels unmasked, ocean channels masked
            pred_m   = torch.cat([pred[:, :self.n_atm],
                                   pred[:, self.n_atm:] * mask], dim=1)
            target_m = torch.cat([target[:, :self.n_atm],
                                   target[:, self.n_atm:] * mask], dim=1)
        else:
            pred_m   = pred * mask
            target_m = target * mask

        # Optional per-variable weighting
        if self.var_weight is not None:
            w = self.var_weight.to(pred_m.device)
            pred_m   = pred_m   * w
            target_m = target_m * w

        return pred_m, target_m

    def train(self, train_data_loader, val_data_loader, loss_fn, optimizer, scheduler):
        """Run the main training loop with optional multi-step rollout.

        In rollout mode (rollout_steps > 1) the training DataLoader must
        yield 3-tuples (input_frame, atm_sequence, ocean_sequence).  The model
        is unrolled for rollout_steps steps autoregressively; the ocean
        prediction at each step is detached before being fed as the next input
        so that gradients are cut between steps (independent per-step
        supervision).  Validation always uses the first step only (comparable
        to single-step baseline).

        Args:
            train_data_loader: Training DataLoader.
            val_data_loader: Validation DataLoader (must yield 2-tuples).
            loss_fn: Loss function (e.g. nn.L1Loss()).
            optimizer: Torch optimiser.
            scheduler: LR scheduler (must accept scheduler.step(val_loss)).

        Returns:
            None: Writes checkpoints and loss CSV as side effects.

        Example:
            >>> trainer.train(train_loader, val_loader, nn.L1Loss(), optimizer, scheduler)
        """
        best_val_loss = self.best_val_loss
        train_losses = []
        val_losses = []
        best_epoch = self.start_epoch if self.best_val_loss != float('inf') else 0

        for epoch in range(self.start_epoch, self.n_epochs):
            print(f"\nStarting epoch {epoch}:")

            # Training phase
            self.model.train()
            train_loss = 0.0
            total_samples = 0
            mask = self.mask.to(self.device)

            pbar = tqdm(train_data_loader, desc=f"Epoch {epoch} [Train]")
            for batch in pbar:
                # ----------------------------------------------------------------
                # Unpack batch — single-step (2-tuple) or rollout (3-tuple)
                # ----------------------------------------------------------------
                if self.rollout_steps > 1:
                    input_frame, atm_sequence, ocean_sequence = batch
                    input_frame    = input_frame.to(self.device)     # (B, 11, H, W)
                    atm_sequence   = atm_sequence.to(self.device)    # (B, k-1, 6, H, W)
                    ocean_sequence = ocean_sequence.to(self.device)  # (B, k, 5, H, W)

                    total_step_loss = torch.tensor(0.0, device=self.device)
                    current_input = input_frame
                    for step in range(self.rollout_steps):
                        pred = self.model(current_input)             # (B, 5, H, W)
                        target_step = ocean_sequence[:, step]        # (B, 5, H, W)
                        pred_m, target_m = self._apply_mask_and_weights(
                            pred, target_step, mask)
                        total_step_loss = total_step_loss + loss_fn(pred_m, target_m)
                        # Build next input: next ERA5 atm + detached ocean prediction
                        if step < self.rollout_steps - 1:
                            next_atm = atm_sequence[:, step]         # (B, 6, H, W)
                            current_input = torch.cat([next_atm, pred.detach()], dim=1)

                    loss = total_step_loss / self.rollout_steps
                    data = input_frame   # for batch-size accounting
                else:
                    data, output = batch
                    data   = data.to(self.device)
                    output = output.to(self.device)
                    generated_data = self.model(data)
                    pred_m, target_m = self._apply_mask_and_weights(
                        generated_data, output, mask)
                    loss = loss_fn(pred_m, target_m)

                train_loss    += loss.item() * len(data)
                total_samples += len(data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({'loss': f'{loss.item():.6f}'})

            train_loss /= total_samples
            train_losses.append(train_loss)

            # Validation phase — always single-step regardless of rollout_steps
            self.model.eval()
            with torch.no_grad():
                test_loss = 0.0
                total_samples = 0

                pbar = tqdm(val_data_loader, desc=f"Epoch {epoch} [Val]")
                for test_input, test_output in pbar:
                    data_test   = test_input.to(self.device)
                    test_output = test_output.to(self.device)
                    output_test = self.model(data_test)
                    pred_m, target_m = self._apply_mask_and_weights(
                        output_test, test_output, mask)
                    loss_test = loss_fn(pred_m, target_m)
                    test_loss     += loss_test.item() * len(data_test)
                    total_samples += len(data_test)
                    pbar.set_postfix({'loss': f'{loss_test.item():.6f}'})

                test_loss /= total_samples
                val_losses.append(test_loss)

                # Step scheduler
                scheduler.step(test_loss)

                print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
                print(f'Device: {self.device}, Epoch: {epoch}, '
                      f'Train Loss: {train_loss:.6f}, Val Loss: {test_loss:.6f}')

            # Save best model
            if test_loss < best_val_loss and epoch > self.min_epoch:
                best_val_loss = test_loss
                best_epoch = epoch

                # Create model filename using experiment name
                filename = get_model_filename(self.config)

                # Save complete checkpoint
                save_dir = self.config.results.model_dir
                os.makedirs(save_dir, exist_ok=True)

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'train_loss': train_loss,
                    'val_loss': test_loss
                }

                # Save full checkpoint
                checkpoint_filename = f"checkpoint_{filename}"
                torch.save(checkpoint, f"{save_dir}/{checkpoint_filename}")

                # Also save just the model state dict for inference
                torch.save(self.model.state_dict(), f"{save_dir}/{filename}")
                print(f"✓ Epoch {epoch}: Model saved with val_loss={test_loss:.6f}")

            # Wandb logging
            if self.config.wb:
                import wandb
                logs = {"train_loss": train_loss, "val_loss": test_loss, "lr": optimizer.param_groups[0]["lr"]}
                wandb.log(logs, step=epoch+1)

        print(f'\n=== Training Complete ===')
        print(f'Best validation loss: {best_val_loss:.6f}')
        print(f'Best epoch: {best_epoch}')

        # Save losses to CSV
        loss_dir = "results/experiments/logs"
        os.makedirs(loss_dir, exist_ok=True)
        filename_csv = f"{loss_dir}/{self.config.name}_train_val_losses.csv"
        with open(filename_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train Loss", "Validation Loss"])
            for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
                writer.writerow([epoch, train_loss, val_loss])

        print(f"Training and validation losses saved to '{filename_csv}'")

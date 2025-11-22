"""
Training Loop for Speech Enhancement

This module implements the main Trainer class that handles the complete
training pipeline including training, validation, checkpointing, and logging.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from .losses import create_loss_function
from .optimizer import create_optimizer, create_scheduler


class Trainer:
    """
    Main trainer class for speech enhancement model.

    Handles the complete training pipeline including:
    - Training and validation loops
    - Loss computation and backpropagation
    - Gradient clipping and optimization
    - Learning rate scheduling
    - Checkpointing (save/load)
    - Logging (TensorBoard)
    - Early stopping
    - Mixed precision training (FP16)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Dict[str, Any],
        device: str = 'cuda',
        resume_from: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to use ('cuda' or 'cpu')
            resume_from: Path to checkpoint to resume from
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Move model to device
        self.model = self.model.to(device)

        # Training config
        self.num_epochs = config.get('num_epochs', 100)
        self.batch_size = config.get('batch_size', 16)
        self.gradient_clip_max_norm = config.get('gradient_clip_max_norm', 1.0)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)

        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None

        # Loss function
        self.criterion = create_loss_function(config)
        self.criterion = self.criterion.to(device)

        # Optimizer
        self.optimizer = create_optimizer(model, config)

        # Learning rate scheduler
        self.scheduler = create_scheduler(self.optimizer, config)

        # Early stopping
        early_stop_config = config.get('early_stopping', {})
        self.early_stopping_enabled = early_stop_config.get('enabled', True)
        self.early_stopping_patience = early_stop_config.get('patience', 10)
        self.early_stopping_min_delta = early_stop_config.get('min_delta', 1e-4)

        # Checkpointing
        checkpoint_config = config.get('checkpoint', {})
        self.checkpoint_dir = Path(checkpoint_config.get('save_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_epochs = checkpoint_config.get('save_every_n_epochs', 5)
        self.save_best_only = checkpoint_config.get('save_best_only', True)

        # Logging
        log_config = config.get('logging', {})
        self.log_dir = Path(log_config.get('log_dir', 'logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_every_n_steps = log_config.get('log_every_n_steps', 10)
        self.use_tensorboard = log_config.get('use_tensorboard', True)

        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        else:
            self.writer = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # Resume from checkpoint if provided
        if resume_from is not None:
            self.load_checkpoint(resume_from)

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {self.current_epoch + 1}/{self.num_epochs} [Train]',
            leave=False
        )

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            noisy_spec = batch['noisy_spec'].to(self.device)
            clean_spec = batch['clean_spec'].to(self.device)

            # Mixed precision training
            with autocast(enabled=self.use_amp):
                # Forward pass
                enhanced_spec = self.model(noisy_spec)

                # Compute loss
                loss = self.criterion(enhanced_spec, clean_spec)

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                if self.gradient_clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_max_norm
                    )

                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                self.global_step += 1

            # Accumulate loss (unscaled)
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': total_loss / num_batches})

            # Logging
            if self.writer and (batch_idx + 1) % self.log_every_n_steps == 0:
                self.writer.add_scalar(
                    'train/batch_loss',
                    loss.item() * self.gradient_accumulation_steps,
                    self.global_step
                )

        # Average loss
        avg_loss = total_loss / num_batches

        return {'loss': avg_loss}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.

        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        # Progress bar
        pbar = tqdm(
            self.val_loader,
            desc=f'Epoch {self.current_epoch + 1}/{self.num_epochs} [Val]',
            leave=False
        )

        for batch in pbar:
            # Move data to device
            noisy_spec = batch['noisy_spec'].to(self.device)
            clean_spec = batch['clean_spec'].to(self.device)

            # Forward pass
            enhanced_spec = self.model(noisy_spec)

            # Compute loss
            loss = self.criterion(enhanced_spec, clean_spec)

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': total_loss / num_batches})

        # Average loss
        avg_loss = total_loss / num_batches

        return {'loss': avg_loss}

    def train(self):
        """
        Main training loop.

        Trains the model for the specified number of epochs with
        validation, checkpointing, and early stopping.
        """
        print(f"\nStarting training for {self.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}\n")

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch

            epoch_start_time = time.time()

            # Training
            train_metrics = self.train_epoch()

            # Validation
            val_metrics = self.validate()

            epoch_time = time.time() - epoch_start_time

            # Print metrics
            print(
                f"Epoch {epoch + 1}/{self.num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.6f}"
            )

            if val_metrics:
                print(f"  Val Loss: {val_metrics['loss']:.6f}")

            print(f"  Time: {epoch_time:.2f}s")

            # Logging
            if self.writer:
                self.writer.add_scalar('train/epoch_loss', train_metrics['loss'], epoch)
                if val_metrics:
                    self.writer.add_scalar('val/epoch_loss', val_metrics['loss'], epoch)

                # Log learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/learning_rate', current_lr, epoch)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metrics:
                        self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # Checkpointing
            val_loss = val_metrics.get('loss', float('inf'))

            # Save best model
            if val_loss < self.best_val_loss:
                print(f"  Validation loss improved from {self.best_val_loss:.6f} to {val_loss:.6f}")
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0

                # Save best checkpoint
                self.save_checkpoint('best_model.pth', is_best=True)

            else:
                self.epochs_without_improvement += 1

            # Save periodic checkpoint
            if (epoch + 1) % self.save_every_n_epochs == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')

            # Early stopping
            if self.early_stopping_enabled:
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(
                        f"\nEarly stopping triggered after {epoch + 1} epochs. "
                        f"No improvement for {self.early_stopping_patience} epochs."
                    )
                    break

            print()  # Empty line for readability

        # Save final checkpoint
        self.save_checkpoint('final_model.pth')

        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")

        if self.writer:
            self.writer.close()

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model so far
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)

        if is_best:
            print(f"  Saved best model to {checkpoint_path}")
        else:
            print(f"  Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load scaler state
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Restore training state
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"Resumed from epoch {self.current_epoch}")

"""
Optimizer and Learning Rate Scheduler Utilities

This module provides factory functions for creating optimizers
and learning rate schedulers with various configurations.
"""

import math
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
    LambdaLR,
)
from typing import Dict, Any


def create_optimizer(
    model: torch.nn.Module,
    config: Dict[str, Any]
) -> optim.Optimizer:
    """
    Create optimizer from configuration.

    Args:
        model: Model to optimize
        config: Optimizer configuration

    Returns:
        PyTorch optimizer
    """
    opt_config = config.get('optimizer', {})

    optimizer_name = opt_config.get('name', 'adamw').lower()
    lr = opt_config.get('lr', 1e-4)
    weight_decay = opt_config.get('weight_decay', 0.01)

    # Get model parameters
    params = model.parameters()

    # Create optimizer
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            params,
            lr=lr,
            betas=opt_config.get('betas', (0.9, 0.999)),
            eps=opt_config.get('eps', 1e-8),
            weight_decay=weight_decay,
            amsgrad=opt_config.get('amsgrad', False),
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            params,
            lr=lr,
            betas=opt_config.get('betas', (0.9, 0.999)),
            eps=opt_config.get('eps', 1e-8),
            weight_decay=weight_decay,
            amsgrad=opt_config.get('amsgrad', False),
        )
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            params,
            lr=lr,
            momentum=opt_config.get('momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=opt_config.get('nesterov', True),
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


def create_scheduler(
    optimizer: optim.Optimizer,
    config: Dict[str, Any]
) -> Any:
    """
    Create learning rate scheduler from configuration.

    Args:
        optimizer: PyTorch optimizer
        config: Scheduler configuration

    Returns:
        Learning rate scheduler
    """
    sched_config = config.get('scheduler', {})

    scheduler_name = sched_config.get('name', 'cosine').lower()
    num_epochs = config.get('num_epochs', 100)

    if scheduler_name == 'cosine':
        # Cosine annealing
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=sched_config.get('min_lr', 1e-6),
        )
    elif scheduler_name == 'warmup_cosine':
        # Cosine annealing with warmup
        warmup_epochs = sched_config.get('warmup_epochs', 5)
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=num_epochs,
            eta_min=sched_config.get('min_lr', 1e-6),
        )
    elif scheduler_name == 'step':
        # Step decay
        scheduler = StepLR(
            optimizer,
            step_size=sched_config.get('step_size', 30),
            gamma=sched_config.get('gamma', 0.1),
        )
    elif scheduler_name == 'plateau':
        # Reduce on plateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=sched_config.get('factor', 0.5),
            patience=sched_config.get('patience', 5),
            min_lr=sched_config.get('min_lr', 1e-6),
        )
    elif scheduler_name == 'none':
        # No scheduler
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return scheduler


class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup and cosine annealing.

    Linearly increases learning rate during warmup phase, then applies
    cosine annealing for the remaining epochs.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        eta_min: float = 0.0,
    ):
        """
        Initialize warmup cosine scheduler.

        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            max_epochs: Total number of epochs
            eta_min: Minimum learning rate
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min

        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        self.current_epoch = 0

    def step(self, epoch: int = None):
        """
        Update learning rate.

        Args:
            epoch: Current epoch number
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        if self.current_epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            lr_scale = (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing phase
            progress = (self.current_epoch - self.warmup_epochs) / \
                      (self.max_epochs - self.warmup_epochs)
            lr_scale = self.eta_min + (1.0 - self.eta_min) * \
                      0.5 * (1.0 + math.cos(math.pi * progress))

        # Update learning rates
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_scale

    def get_last_lr(self):
        """Get last learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]


class WarmupScheduler:
    """
    Learning rate scheduler with linear warmup only.

    Useful for gradually increasing learning rate at the start of training.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        initial_lr: float = 1e-7,
    ):
        """
        Initialize warmup scheduler.

        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            initial_lr: Initial learning rate (before warmup)
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr

        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        self.current_step = 0

    def step(self):
        """Update learning rate."""
        if self.current_step < self.warmup_steps:
            lr_scale = (self.current_step + 1) / self.warmup_steps

            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = self.initial_lr + (base_lr - self.initial_lr) * lr_scale

        self.current_step += 1

    def get_last_lr(self):
        """Get last learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]


def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values
    of the cosine function between the initial lr and 0, after a warmup period.

    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles
        last_epoch: Last epoch number

    Returns:
        LambdaLR scheduler
    """

    def lr_lambda(current_step: int):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay
        progress = float(current_step - num_warmup_steps) / \
                  float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

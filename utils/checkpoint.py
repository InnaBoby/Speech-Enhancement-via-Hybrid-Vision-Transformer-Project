"""
Checkpoint Management Utilities

This module provides utilities for saving and loading model checkpoints,
including model weights, optimizer state, and training metadata.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional


def save_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    global_step: int = 0,
    best_metric: Optional[float] = None,
    config: Optional[Dict] = None,
    **kwargs
) -> None:
    """
    Save model checkpoint.

    Args:
        filepath: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        epoch: Current epoch
        global_step: Global training step
        best_metric: Best validation metric
        config: Training configuration
        **kwargs: Additional items to save
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if best_metric is not None:
        checkpoint['best_metric'] = best_metric

    if config is not None:
        checkpoint['config'] = config

    # Add any additional items
    checkpoint.update(kwargs)

    torch.save(checkpoint, filepath)

    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cpu',
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load checkpoint on
        strict: Whether to strictly enforce state dict keys match

    Returns:
        Dictionary containing checkpoint metadata
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    print(f"Loading checkpoint from {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    else:
        # Assume entire checkpoint is model state dict
        model.load_state_dict(checkpoint, strict=strict)

    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Extract metadata
    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'best_metric': checkpoint.get('best_metric', None),
        'config': checkpoint.get('config', None),
    }

    print(f"Checkpoint loaded (epoch {metadata['epoch']})")

    return metadata


def load_model_weights(
    filepath: str,
    model: nn.Module,
    device: str = 'cpu',
    strict: bool = True,
) -> nn.Module:
    """
    Load only model weights from checkpoint.

    Convenience function for inference.

    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        device: Device to load model on
        strict: Whether to strictly enforce state dict keys match

    Returns:
        Model with loaded weights
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)

    model = model.to(device)

    return model


def save_model_only(
    filepath: str,
    model: nn.Module
) -> None:
    """
    Save only model state dict (no optimizer, scheduler, etc.).

    Useful for distributing trained models.

    Args:
        filepath: Path to save model
        model: Model to save
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), filepath)

    print(f"Model weights saved to {filepath}")


def export_to_onnx(
    model: nn.Module,
    filepath: str,
    input_shape: tuple,
    opset_version: int = 11,
    device: str = 'cpu',
) -> None:
    """
    Export model to ONNX format.

    Useful for deployment and inference optimization.

    Args:
        model: Model to export
        filepath: Path to save ONNX model
        input_shape: Input tensor shape (e.g., (1, 1, 257, 100))
        opset_version: ONNX opset version
        device: Device to use for export
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    model = model.to(device).eval()

    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        opset_version=opset_version,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 3: 'time'},
            'output': {0: 'batch_size', 3: 'time'}
        }
    )

    print(f"Model exported to ONNX format: {filepath}")


def get_checkpoint_info(filepath: str) -> Dict[str, Any]:
    """
    Get information about a checkpoint without loading the full model.

    Args:
        filepath: Path to checkpoint file

    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(filepath, map_location='cpu')

    info = {
        'epoch': checkpoint.get('epoch', 'N/A'),
        'global_step': checkpoint.get('global_step', 'N/A'),
        'best_metric': checkpoint.get('best_metric', 'N/A'),
        'has_optimizer': 'optimizer_state_dict' in checkpoint,
        'has_scheduler': 'scheduler_state_dict' in checkpoint,
        'has_config': 'config' in checkpoint,
    }

    # Count parameters
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        total_params = sum(p.numel() for p in state_dict.values())
        info['total_parameters'] = total_params
    else:
        total_params = sum(p.numel() for p in checkpoint.values() if isinstance(p, torch.Tensor))
        info['total_parameters'] = total_params

    return info


def print_checkpoint_info(filepath: str) -> None:
    """
    Print information about a checkpoint.

    Args:
        filepath: Path to checkpoint file
    """
    info = get_checkpoint_info(filepath)

    print(f"\nCheckpoint Information: {filepath}")
    print("=" * 50)
    for key, value in info.items():
        print(f"{key:20s}: {value}")
    print("=" * 50)

"""
Training Script for Hybrid Vision Transformer Speech Enhancement

This script trains the Hybrid ViT model on the VoiceBank-DEMAND dataset.

Usage:
    python train.py --config config/train_config.yaml
    python train.py --config config/train_config.yaml --resume checkpoints/latest.pth
"""

import argparse
import torch
import random
import numpy as np
from pathlib import Path

from data import VoiceBankDataset, get_data_loader
from models.hybrid_vit import create_hybrid_vit
from training import Trainer
from utils.config import load_all_configs, print_config


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train Hybrid Vision Transformer for Speech Enhancement'
    )
    parser.add_argument(
        '--config-dir',
        type=str,
        default='config',
        help='Directory containing configuration files'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/voicebank_demand',
        help='Root directory of VoiceBank-DEMAND dataset'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training (cuda or cpu)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Load configuration
    print("Loading configuration...")
    config = load_all_configs(args.config_dir)

    # Override config with command-line arguments
    if 'data' not in config:
        config['data'] = {}
    config['data']['data_root'] = args.data_root

    if 'training' not in config:
        config['training'] = {}
    config['training']['device'] = args.device
    config['training']['seed'] = args.seed

    print("\nConfiguration:")
    print("=" * 70)
    print_config(config)
    print("=" * 70)

    # Create datasets
    print("\nCreating datasets...")

    train_dataset = VoiceBankDataset(
        data_root=config['data']['data_root'],
        split='train',
        config=config['data'],
        augment=config['data'].get('augmentation', {}).get('enabled', True),
        cache_spectrograms=config['data'].get('cache', {}).get('enabled', False),
    )

    val_dataset = VoiceBankDataset(
        data_root=config['data']['data_root'],
        split='val',
        config=config['data'],
        augment=False,
        cache_spectrograms=False,
    )

    # Create data loaders
    train_loader = get_data_loader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=config['training'].get('pin_memory', True),
        drop_last=True,
    )

    val_loader = get_data_loader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=config['training'].get('pin_memory', True),
        drop_last=False,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create model
    print("\nCreating model...")
    model = create_hybrid_vit(config)

    # Count parameters
    param_counts = model.count_parameters()
    print(f"Model parameters:")
    print(f"  Encoder: {param_counts['encoder']:,}")
    print(f"  Transformer: {param_counts['transformer']:,}")
    print(f"  Decoder: {param_counts['decoder']:,}")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  Trainable: {param_counts['trainable']:,}")

    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        device=args.device,
        resume_from=args.resume,
    )

    # Train
    print("\nStarting training...")
    print("=" * 70)
    trainer.train()

    print("\nTraining complete!")


if __name__ == '__main__':
    main()

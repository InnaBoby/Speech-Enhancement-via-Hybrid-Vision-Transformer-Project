"""
Evaluation Script for Hybrid Vision Transformer Speech Enhancement

This script evaluates a trained model on the test set and computes
various quality metrics (PESQ, STOI, SI-SDR, etc.).

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth --data-root data/voicebank_demand
    python evaluate.py --checkpoint checkpoints/best_model.pth --save-enhanced --output-dir results/enhanced
"""

import argparse
import torch
from pathlib import Path

from models import create_hybrid_vit
from evaluation import Evaluator
from utils import load_all_configs, load_model_weights


def main():
    """Main evaluation function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Evaluate Hybrid Vision Transformer for Speech Enhancement'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
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
        '--noisy-dir',
        type=str,
        default=None,
        help='Directory containing noisy test audio (overrides default)'
    )
    parser.add_argument(
        '--clean-dir',
        type=str,
        default=None,
        help='Directory containing clean test audio (overrides default)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/evaluation',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--save-enhanced',
        action='store_true',
        help='Save enhanced audio files'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for evaluation (cuda or cpu)'
    )

    args = parser.parse_args()

    # Load configuration
    print("Loading configuration...")
    try:
        config = load_all_configs(args.config_dir)
    except:
        print("Warning: Could not load config files. Using defaults.")
        config = {}

    # Set data paths
    data_root = Path(args.data_root)

    if args.noisy_dir:
        noisy_dir = Path(args.noisy_dir)
    else:
        noisy_dir = data_root / 'noisy_testset_wav'

    if args.clean_dir:
        clean_dir = Path(args.clean_dir)
    else:
        clean_dir = data_root / 'clean_testset_wav'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    print("\nCreating model...")
    model = create_hybrid_vit(config)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    model = load_model_weights(
        args.checkpoint,
        model,
        device=args.device,
        strict=True
    )

    print(f"Model loaded successfully on {args.device}")

    # Create evaluator
    print("\nInitializing evaluator...")

    audio_config = config.get('audio', {})
    evaluator = Evaluator(
        model=model,
        device=args.device,
        sample_rate=audio_config.get('sample_rate', 16000),
        n_fft=audio_config.get('n_fft', 512),
        hop_length=audio_config.get('hop_length', 128),
        win_length=audio_config.get('win_length', 512),
    )

    # Evaluate
    print("\nEvaluating model...")
    print(f"Noisy directory: {noisy_dir}")
    print(f"Clean directory: {clean_dir}")

    enhanced_dir = output_dir / 'enhanced' if args.save_enhanced else None

    results = evaluator.evaluate_dataset(
        noisy_dir=noisy_dir,
        clean_dir=clean_dir,
        output_dir=enhanced_dir,
        save_enhanced=args.save_enhanced,
    )

    # Print results
    evaluator.print_results(results)

    # Save results to JSON
    results_file = output_dir / 'evaluation_results.json'
    evaluator.save_results(results, results_file)

    print(f"\nResults saved to {results_file}")

    if args.save_enhanced:
        print(f"Enhanced audio saved to {enhanced_dir}")

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()

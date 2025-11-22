"""
Audio Enhancement Script

This script enhances noisy audio files using a trained model.

Usage:
    # Single file
    python enhance.py --checkpoint checkpoints/best_model.pth --input noisy.wav --output enhanced.wav

    # Directory
    python enhance.py --checkpoint checkpoints/best_model.pth --input-dir noisy_audios/ --output-dir enhanced_audios/
"""

import argparse
import torch
from pathlib import Path

from models import create_hybrid_vit
from inference import AudioEnhancer
from utils import load_all_configs, load_model_weights


def main():
    """Main enhancement function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Enhance noisy audio using Hybrid Vision Transformer'
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

    # Single file mode
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Path to input noisy audio file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save enhanced audio file'
    )

    # Directory mode
    parser.add_argument(
        '--input-dir',
        type=str,
        default=None,
        help='Directory containing noisy audio files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save enhanced audio files'
    )

    parser.add_argument(
        '--extension',
        type=str,
        default='.wav',
        help='Audio file extension to process (for directory mode)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for inference (cuda or cpu)'
    )
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Disable audio normalization'
    )

    args = parser.parse_args()

    # Validate arguments
    single_file_mode = args.input is not None and args.output is not None
    directory_mode = args.input_dir is not None and args.output_dir is not None

    if not single_file_mode and not directory_mode:
        parser.error(
            "Must specify either:\n"
            "  --input and --output for single file mode, or\n"
            "  --input-dir and --output-dir for directory mode"
        )

    if single_file_mode and directory_mode:
        parser.error("Cannot use both single file and directory mode simultaneously")

    # Load configuration
    print("Loading configuration...")
    try:
        config = load_all_configs(args.config_dir)
    except:
        print("Warning: Could not load config files. Using defaults.")
        config = {}

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

    # Create enhancer
    print("\nInitializing audio enhancer...")

    audio_config = config.get('audio', {})
    enhancer = AudioEnhancer(
        model=model,
        device=args.device,
        sample_rate=audio_config.get('sample_rate', 16000),
        n_fft=audio_config.get('n_fft', 512),
        hop_length=audio_config.get('hop_length', 128),
        win_length=audio_config.get('win_length', 512),
    )

    normalize = not args.no_normalize

    # Enhance audio
    if single_file_mode:
        print(f"\nEnhancing single file...")
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")

        enhancer.enhance_file(
            input_path=args.input,
            output_path=args.output,
            normalize=normalize,
        )

        print("\nEnhancement complete!")

    else:  # directory_mode
        print(f"\nEnhancing directory...")
        print(f"Input directory: {args.input_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"File extension: {args.extension}")

        enhancer.enhance_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            extension=args.extension,
            normalize=normalize,
        )

        print("\nAll files enhanced successfully!")


if __name__ == '__main__':
    main()

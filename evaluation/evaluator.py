"""
Model Evaluator

This module provides the Evaluator class for comprehensive evaluation
of speech enhancement models on test datasets.
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import json

from .metrics import compute_all_metrics, print_metrics


class Evaluator:
    """
    Comprehensive evaluator for speech enhancement models.

    Evaluates models on test datasets and computes various quality metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained speech enhancement model
            device: Device to use for inference
            sample_rate: Audio sample rate
            n_fft: FFT size
            hop_length: Hop length
            win_length: Window length
        """
        self.model = model.to(device).eval()
        self.device = device
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    @torch.no_grad()
    def enhance_audio(self, noisy_audio: np.ndarray) -> np.ndarray:
        """
        Enhance noisy audio.

        Args:
            noisy_audio: Noisy audio waveform

        Returns:
            Enhanced audio waveform
        """
        # Normalize
        max_val = np.abs(noisy_audio).max()
        if max_val > 1e-8:
            noisy_audio = noisy_audio / max_val
        else:
            max_val = 1.0

        # STFT
        noisy_stft = librosa.stft(
            noisy_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        noisy_mag = np.abs(noisy_stft)
        noisy_phase = np.angle(noisy_stft)

        # Normalize magnitude
        mag_max = noisy_mag.max()
        if mag_max > 1e-8:
            noisy_mag_norm = noisy_mag / mag_max
        else:
            noisy_mag_norm = noisy_mag
            mag_max = 1.0

        # To tensor
        noisy_mag_tensor = torch.from_numpy(noisy_mag_norm).float()
        noisy_mag_tensor = noisy_mag_tensor.unsqueeze(0).unsqueeze(0)
        noisy_mag_tensor = noisy_mag_tensor.to(self.device)

        # Enhance
        enhanced_mag_tensor = self.model(noisy_mag_tensor)

        # To numpy
        enhanced_mag = enhanced_mag_tensor.squeeze().cpu().numpy()
        enhanced_mag = enhanced_mag * mag_max

        # Reconstruct
        enhanced_stft = enhanced_mag * np.exp(1j * noisy_phase)

        # ISTFT
        enhanced_audio = librosa.istft(
            enhanced_stft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            length=len(noisy_audio)
        )

        # Denormalize
        enhanced_audio = enhanced_audio * max_val

        return enhanced_audio

    def evaluate_pair(
        self,
        noisy_path: Path,
        clean_path: Path,
    ) -> Dict[str, float]:
        """
        Evaluate a single noisy-clean pair.

        Args:
            noisy_path: Path to noisy audio
            clean_path: Path to clean reference audio

        Returns:
            Dictionary of metrics
        """
        # Load audio
        noisy_audio, _ = librosa.load(noisy_path, sr=self.sample_rate, mono=True)
        clean_audio, _ = librosa.load(clean_path, sr=self.sample_rate, mono=True)

        # Ensure same length
        min_len = min(len(noisy_audio), len(clean_audio))
        noisy_audio = noisy_audio[:min_len]
        clean_audio = clean_audio[:min_len]

        # Enhance
        enhanced_audio = self.enhance_audio(noisy_audio)
        enhanced_audio = enhanced_audio[:min_len]

        # Compute metrics
        metrics = compute_all_metrics(
            clean=clean_audio,
            enhanced=enhanced_audio,
            noisy=noisy_audio,
            sr=self.sample_rate
        )

        return metrics

    def evaluate_dataset(
        self,
        noisy_dir: Path,
        clean_dir: Path,
        output_dir: Optional[Path] = None,
        save_enhanced: bool = False,
    ) -> Dict[str, any]:
        """
        Evaluate model on a dataset.

        Args:
            noisy_dir: Directory containing noisy audio files
            clean_dir: Directory containing clean reference files
            output_dir: Directory to save enhanced audio (if save_enhanced=True)
            save_enhanced: Whether to save enhanced audio files

        Returns:
            Dictionary containing:
                - per_file_metrics: Metrics for each file
                - average_metrics: Average metrics across dataset
        """
        noisy_files = sorted(list(Path(noisy_dir).glob('*.wav')))

        if len(noisy_files) == 0:
            raise ValueError(f"No .wav files found in {noisy_dir}")

        print(f"Evaluating {len(noisy_files)} audio files...")

        all_metrics = []
        per_file_metrics = {}

        # Create output directory if needed
        if save_enhanced and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Evaluate each file
        for noisy_path in tqdm(noisy_files, desc='Evaluating'):
            clean_path = Path(clean_dir) / noisy_path.name

            if not clean_path.exists():
                print(f"Warning: No clean file found for {noisy_path.name}")
                continue

            # Evaluate
            metrics = self.evaluate_pair(noisy_path, clean_path)

            per_file_metrics[noisy_path.name] = metrics
            all_metrics.append(metrics)

            # Save enhanced audio if requested
            if save_enhanced and output_dir:
                # Load and enhance
                noisy_audio, _ = librosa.load(noisy_path, sr=self.sample_rate, mono=True)
                enhanced_audio = self.enhance_audio(noisy_audio)

                # Save
                import soundfile as sf
                output_path = output_dir / noisy_path.name
                sf.write(output_path, enhanced_audio, self.sample_rate)

        # Compute average metrics
        average_metrics = {}
        if all_metrics:
            metric_names = all_metrics[0].keys()
            for metric_name in metric_names:
                values = [m[metric_name] for m in all_metrics]
                average_metrics[metric_name] = np.mean(values)
                average_metrics[f'{metric_name}_std'] = np.std(values)

        return {
            'per_file_metrics': per_file_metrics,
            'average_metrics': average_metrics,
            'num_files': len(all_metrics),
        }

    def save_results(
        self,
        results: Dict,
        output_path: Path,
    ) -> None:
        """
        Save evaluation results to JSON file.

        Args:
            results: Evaluation results dictionary
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        serializable_results = {
            'num_files': results['num_files'],
            'average_metrics': {
                k: float(v) for k, v in results['average_metrics'].items()
            },
            'per_file_metrics': {
                filename: {k: float(v) for k, v in metrics.items()}
                for filename, metrics in results['per_file_metrics'].items()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {output_path}")

    def print_results(self, results: Dict) -> None:
        """
        Print evaluation results.

        Args:
            results: Evaluation results dictionary
        """
        print(f"\nEvaluation Results ({results['num_files']} files)")
        print("=" * 70)

        # Print average metrics
        avg_metrics = results['average_metrics']

        # Group metrics
        main_metrics = ['pesq', 'stoi', 'sisdr', 'snr']
        improvement_metrics = ['pesq_improvement', 'stoi_improvement',
                              'sisdr_improvement', 'snr_improvement']

        print("\nMain Metrics:")
        print("-" * 70)
        for metric in main_metrics:
            if metric in avg_metrics:
                mean = avg_metrics[metric]
                std = avg_metrics.get(f'{metric}_std', 0.0)
                print(f"{metric.upper():15s}: {mean:7.4f} ± {std:6.4f}")

        print("\nImprovement Over Noisy:")
        print("-" * 70)
        for metric in improvement_metrics:
            if metric in avg_metrics:
                mean = avg_metrics[metric]
                std = avg_metrics.get(f'{metric}_std', 0.0)
                name = metric.replace('_improvement', '').upper()
                print(f"{name:15s}: {mean:7.4f} ± {std:6.4f}")

        print("\nOther Metrics:")
        print("-" * 70)
        for metric, value in avg_metrics.items():
            if not metric.endswith('_std') and \
               metric not in main_metrics and \
               metric not in improvement_metrics:
                std = avg_metrics.get(f'{metric}_std', 0.0)
                print(f"{metric.upper():15s}: {value:7.4f} ± {std:6.4f}")

        print("=" * 70)

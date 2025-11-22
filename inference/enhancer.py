"""
Audio Enhancement Inference

This module provides utilities for enhancing noisy audio files
using trained speech enhancement models.
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional, Union, Tuple
from tqdm import tqdm


class AudioEnhancer:
    """
    Audio enhancement inference engine.

    Handles loading trained models and enhancing noisy audio files.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        window: str = 'hann',
    ):
        """
        Initialize audio enhancer.

        Args:
            model: Trained speech enhancement model
            device: Device to run inference on
            sample_rate: Audio sample rate
            n_fft: FFT size for STFT
            hop_length: Hop length for STFT
            win_length: Window length for STFT
            window: Window type for STFT
        """
        self.model = model.to(device).eval()
        self.device = device
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window

    @torch.no_grad()
    def enhance(
        self,
        noisy_audio: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Enhance noisy audio.

        Args:
            noisy_audio: Noisy audio waveform
            normalize: Whether to normalize audio before processing

        Returns:
            Enhanced audio waveform
        """
        # Normalize input
        if normalize:
            max_val = np.abs(noisy_audio).max()
            if max_val > 1e-8:
                noisy_audio = noisy_audio / max_val
            else:
                max_val = 1.0
        else:
            max_val = 1.0

        # Compute STFT
        noisy_stft = librosa.stft(
            noisy_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True
        )

        # Extract magnitude and phase
        noisy_mag = np.abs(noisy_stft)
        noisy_phase = np.angle(noisy_stft)

        # Normalize magnitude
        mag_max = noisy_mag.max()
        if mag_max > 1e-8:
            noisy_mag_norm = noisy_mag / mag_max
        else:
            noisy_mag_norm = noisy_mag
            mag_max = 1.0

        # Convert to tensor
        noisy_mag_tensor = torch.from_numpy(noisy_mag_norm).float()
        noisy_mag_tensor = noisy_mag_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, F, T]
        noisy_mag_tensor = noisy_mag_tensor.to(self.device)

        # Enhance spectrogram
        enhanced_mag_tensor = self.model(noisy_mag_tensor)

        # Convert back to numpy
        enhanced_mag = enhanced_mag_tensor.squeeze().cpu().numpy()

        # Denormalize
        enhanced_mag = enhanced_mag * mag_max

        # Reconstruct complex spectrogram using original phase
        # (Phase-aware enhancement would be more sophisticated)
        enhanced_stft = enhanced_mag * np.exp(1j * noisy_phase)

        # Inverse STFT
        enhanced_audio = librosa.istft(
            enhanced_stft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            length=len(noisy_audio)
        )

        # Denormalize output
        if normalize:
            enhanced_audio = enhanced_audio * max_val

        return enhanced_audio

    def enhance_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        normalize: bool = True,
    ) -> None:
        """
        Enhance audio from file and save result.

        Args:
            input_path: Path to input noisy audio file
            output_path: Path to save enhanced audio file
            normalize: Whether to normalize audio
        """
        # Load audio
        noisy_audio, sr = librosa.load(input_path, sr=self.sample_rate, mono=True)

        # Enhance
        enhanced_audio = self.enhance(noisy_audio, normalize=normalize)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, enhanced_audio, self.sample_rate)

        print(f"Enhanced audio saved to {output_path}")

    def enhance_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        extension: str = '.wav',
        normalize: bool = True,
    ) -> None:
        """
        Enhance all audio files in a directory.

        Args:
            input_dir: Input directory containing noisy audio files
            output_dir: Output directory for enhanced audio files
            extension: Audio file extension to process
            normalize: Whether to normalize audio
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get all audio files
        audio_files = list(input_path.glob(f'*{extension}'))

        print(f"Found {len(audio_files)} audio files to enhance")

        # Process each file
        for audio_file in tqdm(audio_files, desc='Enhancing audio'):
            output_file = output_path / audio_file.name
            self.enhance_file(audio_file, output_file, normalize=normalize)

        print(f"All files enhanced and saved to {output_path}")


def enhance_audio(
    noisy_audio: np.ndarray,
    model: nn.Module,
    device: str = 'cuda',
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 128,
) -> np.ndarray:
    """
    Enhance noisy audio using a trained model.

    Convenience function for quick audio enhancement.

    Args:
        noisy_audio: Noisy audio waveform
        model: Trained enhancement model
        device: Device to use
        sample_rate: Audio sample rate
        n_fft: FFT size
        hop_length: Hop length

    Returns:
        Enhanced audio waveform
    """
    enhancer = AudioEnhancer(
        model=model,
        device=device,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    return enhancer.enhance(noisy_audio)


def enhance_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    model: nn.Module,
    device: str = 'cuda',
    sample_rate: int = 16000,
) -> None:
    """
    Enhance audio file using a trained model.

    Args:
        input_path: Path to input noisy audio
        output_path: Path to save enhanced audio
        model: Trained enhancement model
        device: Device to use
        sample_rate: Audio sample rate
    """
    enhancer = AudioEnhancer(
        model=model,
        device=device,
        sample_rate=sample_rate,
    )

    enhancer.enhance_file(input_path, output_path)


def load_model_for_inference(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    device: str = 'cuda',
) -> nn.Module:
    """
    Load trained model from checkpoint for inference.

    Args:
        checkpoint_path: Path to model checkpoint
        model: Model instance (architecture)
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    print(f"Loading model from {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Set to evaluation mode
    model = model.to(device).eval()

    print(f"Model loaded successfully")

    return model

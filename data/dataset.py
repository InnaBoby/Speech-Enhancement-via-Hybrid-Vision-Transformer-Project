"""
VoiceBank-DEMAND Dataset Implementation

This module implements the PyTorch Dataset class for loading and processing
the VoiceBank-DEMAND speech enhancement dataset.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .preprocessing import AudioPreprocessor
from .augmentation import SpectrogramAugmenter


class VoiceBankDataset(Dataset):
    """
    PyTorch Dataset for VoiceBank-DEMAND speech enhancement.

    This dataset loads pairs of noisy and clean speech audio files,
    computes their spectrograms, and applies optional augmentation.

    The VoiceBank-DEMAND dataset consists of:
    - Training: 11,572 utterances from 28 speakers
    - Testing: 824 utterances from 2 speakers
    - Noise types: 10 different conditions (cafeteria, street, etc.)
    - SNR levels: -5dB to 15dB
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        config: Optional[Dict] = None,
        augment: bool = True,
        cache_spectrograms: bool = False,
    ):
        """
        Initialize VoiceBank-DEMAND dataset.

        Args:
            data_root: Root directory containing the dataset
            split: Dataset split ('train', 'val', or 'test')
            config: Configuration dictionary with audio and augmentation params
            augment: Whether to apply data augmentation (only for training)
            cache_spectrograms: Whether to cache computed spectrograms in memory
                              (faster but requires more RAM)
        """
        super().__init__()

        self.data_root = Path(data_root)
        self.split = split
        self.config = config or self._default_config()
        self.augment = augment and split == 'train'
        self.cache_spectrograms = cache_spectrograms

        # Audio parameters
        self.sample_rate = self.config.get('sample_rate', 16000)
        self.n_fft = self.config.get('n_fft', 512)
        self.hop_length = self.config.get('hop_length', 128)
        self.win_length = self.config.get('win_length', 512)
        self.window = self.config.get('window', 'hann')

        # Initialize preprocessor and augmenter
        self.preprocessor = AudioPreprocessor(self.config)
        self.augmenter = SpectrogramAugmenter(self.config) if self.augment else None

        # Load file pairs
        self.file_pairs = self._load_file_pairs()

        # Cache for spectrograms
        self.spec_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        print(f"Loaded {len(self.file_pairs)} audio pairs for {split} split")

    def _default_config(self) -> Dict:
        """Return default configuration if none provided."""
        return {
            'sample_rate': 16000,
            'n_fft': 512,
            'hop_length': 128,
            'win_length': 512,
            'window': 'hann',
            'normalize_audio': True,
            'normalize_spectrogram': True,
            'normalization_type': 'per_utterance',
        }

    def _load_file_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Load pairs of (noisy, clean) audio file paths.

        Returns:
            List of tuples containing (noisy_path, clean_path)
        """
        if self.split in ['train', 'val']:
            noisy_dir = self.data_root / 'noisy_trainset_28spk_wav'
            clean_dir = self.data_root / 'clean_trainset_28spk_wav'
        elif self.split == 'test':
            noisy_dir = self.data_root / 'noisy_testset_wav'
            clean_dir = self.data_root / 'clean_testset_wav'
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'")

        if not noisy_dir.exists():
            raise FileNotFoundError(
                f"Noisy audio directory not found: {noisy_dir}\n"
                f"Please download the VoiceBank-DEMAND dataset and extract it to {self.data_root}"
            )

        if not clean_dir.exists():
            raise FileNotFoundError(
                f"Clean audio directory not found: {clean_dir}\n"
                f"Please download the VoiceBank-DEMAND dataset and extract it to {self.data_root}"
            )

        # Get all noisy files
        noisy_files = sorted(list(noisy_dir.glob('*.wav')))

        file_pairs = []
        for noisy_path in noisy_files:
            # Match with corresponding clean file
            clean_path = clean_dir / noisy_path.name

            if clean_path.exists():
                file_pairs.append((noisy_path, clean_path))
            else:
                print(f"Warning: No clean file found for {noisy_path.name}")

        # Split train/val if needed
        if self.split in ['train', 'val']:
            val_split = self.config.get('train_val_split', 0.9)
            split_idx = int(len(file_pairs) * val_split)

            if self.split == 'train':
                file_pairs = file_pairs[:split_idx]
            else:  # val
                file_pairs = file_pairs[split_idx:]

        return file_pairs

    def load_audio(self, audio_path: Path) -> np.ndarray:
        """
        Load audio file using librosa.

        Args:
            audio_path: Path to audio file

        Returns:
            Audio waveform as numpy array
        """
        try:
            audio, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                mono=True
            )
            return audio
        except Exception as e:
            raise RuntimeError(f"Failed to load audio from {audio_path}: {e}")

    def compute_spectrogram(
        self,
        audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute magnitude and phase spectrograms using STFT.

        Args:
            audio: Audio waveform

        Returns:
            Tuple of (magnitude, phase) spectrograms
        """
        # Compute STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True
        )

        # Extract magnitude and phase
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        return magnitude, phase

    def normalize_spectrogram(
        self,
        magnitude: np.ndarray,
        norm_type: str = 'per_utterance'
    ) -> np.ndarray:
        """
        Normalize magnitude spectrogram.

        Args:
            magnitude: Magnitude spectrogram
            norm_type: Normalization type ('per_utterance' or 'global')

        Returns:
            Normalized magnitude spectrogram
        """
        if norm_type == 'per_utterance':
            # Normalize to [0, 1] using min-max scaling
            mag_min = magnitude.min()
            mag_max = magnitude.max()

            if mag_max > mag_min:
                magnitude = (magnitude - mag_min) / (mag_max - mag_min)

        elif norm_type == 'global':
            # Use a fixed global normalization (you can compute statistics on train set)
            # For now, use robust scaling
            magnitude = magnitude / (np.percentile(magnitude, 95) + 1e-8)
            magnitude = np.clip(magnitude, 0, 1)

        return magnitude

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.file_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - noisy_spec: Noisy magnitude spectrogram [1, F, T]
                - clean_spec: Clean magnitude spectrogram [1, F, T]
                - noisy_phase: Noisy phase spectrogram [1, F, T]
                - clean_phase: Clean phase spectrogram [1, F, T]
        """
        # Check cache first
        if self.cache_spectrograms and idx in self.spec_cache:
            noisy_spec, clean_spec = self.spec_cache[idx]
        else:
            # Load audio files
            noisy_path, clean_path = self.file_pairs[idx]
            noisy_audio = self.load_audio(noisy_path)
            clean_audio = self.load_audio(clean_path)

            # Apply preprocessing
            noisy_audio = self.preprocessor.process(noisy_audio)
            clean_audio = self.preprocessor.process(clean_audio)

            # Ensure same length (important for paired data)
            min_len = min(len(noisy_audio), len(clean_audio))
            noisy_audio = noisy_audio[:min_len]
            clean_audio = clean_audio[:min_len]

            # Compute spectrograms
            noisy_mag, noisy_phase = self.compute_spectrogram(noisy_audio)
            clean_mag, clean_phase = self.compute_spectrogram(clean_audio)

            # Normalize spectrograms
            norm_type = self.config.get('normalization_type', 'per_utterance')
            noisy_mag = self.normalize_spectrogram(noisy_mag, norm_type)
            clean_mag = self.normalize_spectrogram(clean_mag, norm_type)

            # Convert to tensors
            noisy_spec = torch.from_numpy(noisy_mag).float().unsqueeze(0)
            clean_spec = torch.from_numpy(clean_mag).float().unsqueeze(0)
            noisy_phase = torch.from_numpy(noisy_phase).float().unsqueeze(0)
            clean_phase = torch.from_numpy(clean_phase).float().unsqueeze(0)

            # Cache if enabled
            if self.cache_spectrograms:
                self.spec_cache[idx] = (noisy_spec, clean_spec)

        # Apply augmentation if enabled
        if self.augment and self.augmenter is not None:
            noisy_spec = self.augmenter.augment(noisy_spec)
            # Note: We don't augment clean_spec as it's the target

        return {
            'noisy_spec': noisy_spec,
            'clean_spec': clean_spec,
            'noisy_phase': noisy_phase,
            'clean_phase': clean_phase,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable-length spectrograms.

    Pads spectrograms to the maximum length in the batch.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary with padded tensors
    """
    # Find maximum time dimension in batch
    max_time = max(sample['noisy_spec'].shape[-1] for sample in batch)

    # Pad all spectrograms to max_time
    noisy_specs = []
    clean_specs = []
    noisy_phases = []
    clean_phases = []

    for sample in batch:
        # Get current dimensions
        current_time = sample['noisy_spec'].shape[-1]
        pad_width = max_time - current_time

        # Pad spectrograms (right padding in time dimension)
        if pad_width > 0:
            padding = (0, pad_width)  # (left, right)
            noisy_spec = torch.nn.functional.pad(sample['noisy_spec'], padding)
            clean_spec = torch.nn.functional.pad(sample['clean_spec'], padding)
            noisy_phase = torch.nn.functional.pad(sample['noisy_phase'], padding)
            clean_phase = torch.nn.functional.pad(sample['clean_phase'], padding)
        else:
            noisy_spec = sample['noisy_spec']
            clean_spec = sample['clean_spec']
            noisy_phase = sample['noisy_phase']
            clean_phase = sample['clean_phase']

        noisy_specs.append(noisy_spec)
        clean_specs.append(clean_spec)
        noisy_phases.append(noisy_phase)
        clean_phases.append(clean_phase)

    # Stack into batches
    return {
        'noisy_spec': torch.stack(noisy_specs),
        'clean_spec': torch.stack(clean_specs),
        'noisy_phase': torch.stack(noisy_phases),
        'clean_phase': torch.stack(clean_phases),
    }


def get_data_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for the dataset.

    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory (faster GPU transfer)
        drop_last: Whether to drop last incomplete batch

    Returns:
        PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )

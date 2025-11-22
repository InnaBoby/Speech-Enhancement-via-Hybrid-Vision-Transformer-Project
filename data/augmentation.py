"""
Data Augmentation for Speech Enhancement

This module provides spectrogram augmentation techniques including
SpecAugment, time stretching, pitch shifting, and random gain adjustment.
"""

import numpy as np
import torch
import librosa
from typing import Dict, Optional, Tuple


class SpectrogramAugmenter:
    """
    Spectrogram augmentation for speech enhancement training.

    Implements various augmentation techniques to improve model
    robustness and generalization.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize spectrogram augmenter.

        Args:
            config: Configuration dictionary with augmentation parameters
        """
        self.config = config or {}

        # Get augmentation config
        aug_config = self.config.get('augmentation', {})

        # SpecAugment parameters
        spec_aug = aug_config.get('spec_augment', {})
        self.spec_augment_enabled = spec_aug.get('enabled', True)
        self.freq_mask_num = spec_aug.get('freq_mask_num', 2)
        self.freq_mask_width = spec_aug.get('freq_mask_width', 15)
        self.time_mask_num = spec_aug.get('time_mask_num', 2)
        self.time_mask_width = spec_aug.get('time_mask_width', 30)

        # Random gain parameters
        gain_aug = aug_config.get('random_gain', {})
        self.random_gain_enabled = gain_aug.get('enabled', True)
        self.gain_db_range = gain_aug.get('gain_db_range', [-3, 3])
        self.gain_probability = gain_aug.get('probability', 0.5)

    def augment(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation pipeline to spectrogram.

        Args:
            spectrogram: Input spectrogram [C, F, T]

        Returns:
            Augmented spectrogram
        """
        # Apply SpecAugment
        if self.spec_augment_enabled:
            spectrogram = self.spec_augment(spectrogram)

        # Apply random gain
        if self.random_gain_enabled and np.random.rand() < self.gain_probability:
            spectrogram = self.random_gain(spectrogram)

        return spectrogram

    def spec_augment(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrogram.

        SpecAugment applies random frequency and time masking to spectrograms.
        This augmentation technique was originally proposed for ASR but works
        well for other speech tasks.

        Reference: Park et al. "SpecAugment: A Simple Data Augmentation Method
        for Automatic Speech Recognition" (2019)

        Args:
            spectrogram: Input spectrogram [C, F, T]

        Returns:
            Augmented spectrogram
        """
        spec_aug = spectrogram.clone()
        channels, n_freqs, n_times = spec_aug.shape

        # Apply frequency masking
        for _ in range(self.freq_mask_num):
            f = np.random.randint(0, self.freq_mask_width)
            f0 = np.random.randint(0, max(1, n_freqs - f))
            spec_aug[:, f0:f0 + f, :] = 0

        # Apply time masking
        for _ in range(self.time_mask_num):
            t = np.random.randint(0, min(self.time_mask_width, n_times))
            t0 = np.random.randint(0, max(1, n_times - t))
            spec_aug[:, :, t0:t0 + t] = 0

        return spec_aug

    def random_gain(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply random gain adjustment to spectrogram.

        Simulates variations in recording levels and distances.

        Args:
            spectrogram: Input spectrogram [C, F, T]

        Returns:
            Gain-adjusted spectrogram
        """
        # Random gain in dB
        gain_db = np.random.uniform(*self.gain_db_range)
        gain_linear = 10 ** (gain_db / 20)

        return spectrogram * gain_linear

    @staticmethod
    def freq_mask(
        spectrogram: torch.Tensor,
        num_masks: int = 2,
        mask_width: int = 15
    ) -> torch.Tensor:
        """
        Apply frequency masking to spectrogram.

        Args:
            spectrogram: Input spectrogram [C, F, T]
            num_masks: Number of frequency masks to apply
            mask_width: Maximum width of each mask

        Returns:
            Frequency-masked spectrogram
        """
        spec_masked = spectrogram.clone()
        _, n_freqs, _ = spec_masked.shape

        for _ in range(num_masks):
            f = np.random.randint(0, mask_width)
            f0 = np.random.randint(0, max(1, n_freqs - f))
            spec_masked[:, f0:f0 + f, :] = 0

        return spec_masked

    @staticmethod
    def time_mask(
        spectrogram: torch.Tensor,
        num_masks: int = 2,
        mask_width: int = 30
    ) -> torch.Tensor:
        """
        Apply time masking to spectrogram.

        Args:
            spectrogram: Input spectrogram [C, F, T]
            num_masks: Number of time masks to apply
            mask_width: Maximum width of each mask

        Returns:
            Time-masked spectrogram
        """
        spec_masked = spectrogram.clone()
        _, _, n_times = spec_masked.shape

        for _ in range(num_masks):
            t = np.random.randint(0, min(mask_width, n_times))
            t0 = np.random.randint(0, max(1, n_times - t))
            spec_masked[:, :, t0:t0 + t] = 0

        return spec_masked


class AudioAugmenter:
    """
    Time-domain audio augmentation.

    Applies augmentation directly to audio waveforms before
    spectrogram computation. This is more computationally expensive
    but can provide more realistic augmentations.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize audio augmenter.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Get augmentation config
        aug_config = self.config.get('augmentation', {})

        # Time stretch parameters
        time_stretch = aug_config.get('time_stretch', {})
        self.time_stretch_enabled = time_stretch.get('enabled', False)
        self.time_stretch_rate = time_stretch.get('rate_range', [0.9, 1.1])
        self.time_stretch_prob = time_stretch.get('probability', 0.3)

        # Pitch shift parameters
        pitch_shift = aug_config.get('pitch_shift', {})
        self.pitch_shift_enabled = pitch_shift.get('enabled', False)
        self.pitch_shift_range = pitch_shift.get('n_steps_range', [-2, 2])
        self.pitch_shift_prob = pitch_shift.get('probability', 0.3)

    def augment(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Apply audio augmentation pipeline.

        Args:
            audio: Input audio waveform
            sr: Sample rate

        Returns:
            Augmented audio
        """
        # Time stretching
        if self.time_stretch_enabled and np.random.rand() < self.time_stretch_prob:
            audio = self.time_stretch(audio, sr)

        # Pitch shifting
        if self.pitch_shift_enabled and np.random.rand() < self.pitch_shift_prob:
            audio = self.pitch_shift(audio, sr)

        return audio

    def time_stretch(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply time stretching to audio.

        Changes the speed of audio without changing pitch.

        Args:
            audio: Input audio waveform
            sr: Sample rate

        Returns:
            Time-stretched audio
        """
        rate = np.random.uniform(*self.time_stretch_rate)
        return librosa.effects.time_stretch(audio, rate=rate)

    def pitch_shift(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply pitch shifting to audio.

        Changes the pitch without changing duration.

        Args:
            audio: Input audio waveform
            sr: Sample rate

        Returns:
            Pitch-shifted audio
        """
        n_steps = np.random.uniform(*self.pitch_shift_range)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    @staticmethod
    def add_reverb(
        audio: np.ndarray,
        room_scale: float = 0.5,
        damping: float = 0.5
    ) -> np.ndarray:
        """
        Add simple reverb effect to audio.

        This is a simplified reverb using comb filters.
        For production use, consider more sophisticated reverb algorithms.

        Args:
            audio: Input audio waveform
            room_scale: Room size parameter (0-1)
            damping: Damping parameter (0-1)

        Returns:
            Audio with reverb
        """
        # Simple comb filter-based reverb
        # Note: This is a basic implementation. For better quality,
        # use external libraries like pyroomacoustics

        delay_samples = int(room_scale * 4800)  # Delay based on room size
        reverb_audio = audio.copy()

        # Add delayed and attenuated copies
        for i in range(3):
            delay = delay_samples * (i + 1)
            attenuation = damping ** (i + 1)

            if delay < len(audio):
                delayed = np.pad(audio[:-delay], (delay, 0), mode='constant')
                reverb_audio = reverb_audio + attenuation * delayed

        # Normalize
        max_val = np.abs(reverb_audio).max()
        if max_val > 0:
            reverb_audio = reverb_audio / max_val

        return reverb_audio

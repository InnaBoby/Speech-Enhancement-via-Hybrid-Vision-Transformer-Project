"""
Audio Preprocessing Utilities

This module provides audio preprocessing functions including
resampling, normalization, pre-emphasis, and voice activity detection.
"""

import numpy as np
import librosa
from scipy import signal
from typing import Dict, Optional, Tuple


class AudioPreprocessor:
    """
    Audio preprocessing pipeline for speech enhancement.

    Applies various preprocessing steps to raw audio waveforms
    before spectrogram computation.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize audio preprocessor.

        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config or {}

        # Preprocessing flags
        self.normalize_audio = self.config.get('normalize_audio', True)
        self.apply_pre_emphasis = self.config.get('apply_pre_emphasis', False)
        self.pre_emphasis_coef = self.config.get('pre_emphasis_coef', 0.97)
        self.apply_vad = self.config.get('apply_vad', False)
        self.vad_threshold = self.config.get('vad_threshold', 0.01)
        self.trim_silence = self.config.get('trim_silence', False)
        self.sample_rate = self.config.get('sample_rate', 16000)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline to audio.

        Args:
            audio: Input audio waveform

        Returns:
            Preprocessed audio waveform
        """
        # Apply pre-emphasis filter
        if self.apply_pre_emphasis:
            audio = self.pre_emphasis(audio, self.pre_emphasis_coef)

        # Trim silence
        if self.trim_silence:
            audio = self.trim_silence_vad(audio, self.vad_threshold)

        # Normalize audio
        if self.normalize_audio:
            audio = self.normalize(audio)

        return audio

    @staticmethod
    def normalize(audio: np.ndarray, target_level: float = 1.0) -> np.ndarray:
        """
        Normalize audio to target level.

        Scales audio so that its maximum absolute value equals target_level.

        Args:
            audio: Input audio waveform
            target_level: Target maximum absolute value

        Returns:
            Normalized audio
        """
        max_val = np.abs(audio).max()

        if max_val > 1e-8:  # Avoid division by zero
            audio = audio * (target_level / max_val)

        return audio

    @staticmethod
    def pre_emphasis(audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
        """
        Apply pre-emphasis filter to emphasize high frequencies.

        The pre-emphasis filter is defined as:
        y[n] = x[n] - coef * x[n-1]

        This helps to balance the frequency spectrum since high frequencies
        typically have lower magnitudes in speech signals.

        Args:
            audio: Input audio waveform
            coef: Pre-emphasis coefficient (typically 0.95-0.97)

        Returns:
            Pre-emphasized audio
        """
        return np.append(audio[0], audio[1:] - coef * audio[:-1])

    @staticmethod
    def de_emphasis(audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
        """
        Reverse pre-emphasis filter.

        Args:
            audio: Pre-emphasized audio waveform
            coef: Pre-emphasis coefficient

        Returns:
            De-emphasized audio
        """
        return signal.lfilter([1], [1, -coef], audio)

    def trim_silence_vad(
        self,
        audio: np.ndarray,
        threshold: float = 0.01,
        frame_length: int = 512,
        hop_length: int = 128
    ) -> np.ndarray:
        """
        Trim silence from audio using simple energy-based VAD.

        Voice Activity Detection (VAD) identifies regions with speech
        and removes silent regions at the beginning and end.

        Args:
            audio: Input audio waveform
            threshold: Energy threshold for voice activity
            frame_length: Frame length for energy computation
            hop_length: Hop length between frames

        Returns:
            Trimmed audio
        """
        # Compute frame-wise energy
        energy = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        # Find frames with energy above threshold
        voice_frames = energy > threshold

        if not voice_frames.any():
            # No voice detected, return original
            return audio

        # Find start and end of voice activity
        voice_indices = np.where(voice_frames)[0]
        start_frame = voice_indices[0]
        end_frame = voice_indices[-1]

        # Convert frame indices to sample indices
        start_sample = start_frame * hop_length
        end_sample = min((end_frame + 1) * hop_length, len(audio))

        return audio[start_sample:end_sample]

    @staticmethod
    def resample(
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.

        Args:
            audio: Input audio waveform
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio
        """
        if orig_sr == target_sr:
            return audio

        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    @staticmethod
    def dynamic_range_compression(
        audio: np.ndarray,
        threshold: float = 0.5,
        ratio: float = 4.0
    ) -> np.ndarray:
        """
        Apply dynamic range compression to audio.

        Compression reduces the dynamic range by attenuating signals
        above a threshold, making quiet sounds more audible.

        Args:
            audio: Input audio waveform
            threshold: Threshold for compression (0-1)
            ratio: Compression ratio (e.g., 4:1)

        Returns:
            Compressed audio
        """
        # Compute envelope
        envelope = np.abs(audio)

        # Apply compression above threshold
        mask = envelope > threshold
        compressed = audio.copy()

        if mask.any():
            # Compute gain reduction
            gain_reduction = 1.0 + (1.0 / ratio - 1.0) * (
                (envelope[mask] - threshold) / (1.0 - threshold)
            )
            compressed[mask] = audio[mask] * gain_reduction

        return compressed

    @staticmethod
    def add_noise(
        audio: np.ndarray,
        noise: np.ndarray,
        snr_db: float
    ) -> np.ndarray:
        """
        Add noise to audio at specified SNR.

        Args:
            audio: Clean audio signal
            noise: Noise signal
            snr_db: Signal-to-Noise Ratio in dB

        Returns:
            Noisy audio
        """
        # Ensure noise is same length as audio
        if len(noise) < len(audio):
            # Repeat noise if too short
            repeats = int(np.ceil(len(audio) / len(noise)))
            noise = np.tile(noise, repeats)

        noise = noise[:len(audio)]

        # Compute signal and noise power
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)

        # Compute required noise scaling factor for target SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_scale = np.sqrt(signal_power / (snr_linear * noise_power))

        # Add scaled noise to signal
        noisy_audio = audio + noise_scale * noise

        return noisy_audio

    @staticmethod
    def compute_snr(
        clean: np.ndarray,
        noisy: np.ndarray
    ) -> float:
        """
        Compute Signal-to-Noise Ratio between clean and noisy signals.

        Args:
            clean: Clean signal
            noisy: Noisy signal

        Returns:
            SNR in dB
        """
        noise = noisy - clean
        signal_power = np.mean(clean ** 2)
        noise_power = np.mean(noise ** 2)

        if noise_power < 1e-10:
            return float('inf')

        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def process_file(
        self,
        input_path: str,
        output_path: str
    ) -> None:
        """
        Process a single audio file and save the result.

        Args:
            input_path: Path to input audio file
            output_path: Path to save processed audio
        """
        # Load audio
        audio, sr = librosa.load(input_path, sr=self.sample_rate, mono=True)

        # Process
        processed_audio = self.process(audio)

        # Save
        import soundfile as sf
        sf.write(output_path, processed_audio, self.sample_rate)

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        extension: str = '.wav'
    ) -> None:
        """
        Process all audio files in a directory.

        Args:
            input_dir: Directory containing input audio files
            output_dir: Directory to save processed audio files
            extension: Audio file extension to process
        """
        import os
        from pathlib import Path
        from tqdm import tqdm

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get all audio files
        audio_files = list(input_path.glob(f'*{extension}'))

        # Process each file
        for audio_file in tqdm(audio_files, desc='Processing audio files'):
            output_file = output_path / audio_file.name
            self.process_file(str(audio_file), str(output_file))

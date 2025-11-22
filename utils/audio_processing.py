"""
Audio Processing Utilities

This module provides utilities for audio I/O and signal processing
operations including STFT, inverse STFT, and normalization.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, Union


def load_audio(
    file_path: Union[str, Path],
    sr: int = 16000,
    mono: bool = True,
    offset: float = 0.0,
    duration: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """
    Load audio file.

    Args:
        file_path: Path to audio file
        sr: Target sample rate
        mono: Convert to mono
        offset: Start reading after this time (in seconds)
        duration: Only load up to this duration (in seconds)

    Returns:
        Tuple of (audio waveform, sample rate)
    """
    audio, sample_rate = librosa.load(
        file_path,
        sr=sr,
        mono=mono,
        offset=offset,
        duration=duration
    )

    return audio, sample_rate


def save_audio(
    file_path: Union[str, Path],
    audio: np.ndarray,
    sr: int = 16000,
    subtype: str = 'PCM_16',
) -> None:
    """
    Save audio to file.

    Args:
        file_path: Path to save audio
        audio: Audio waveform
        sr: Sample rate
        subtype: Audio subtype (e.g., 'PCM_16', 'FLOAT')
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(file_path, audio, sr, subtype=subtype)


def compute_stft(
    audio: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int = 512,
    window: str = 'hann',
    center: bool = True,
) -> np.ndarray:
    """
    Compute Short-Time Fourier Transform.

    Args:
        audio: Audio waveform
        n_fft: FFT size
        hop_length: Hop length between frames
        win_length: Window length
        window: Window function type
        center: Whether to center frames

    Returns:
        Complex STFT [n_freq_bins, n_frames]
    """
    stft = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center
    )

    return stft


def compute_istft(
    stft: np.ndarray,
    hop_length: int = 128,
    win_length: int = 512,
    window: str = 'hann',
    center: bool = True,
    length: Optional[int] = None,
) -> np.ndarray:
    """
    Compute inverse Short-Time Fourier Transform.

    Args:
        stft: Complex STFT [n_freq_bins, n_frames]
        hop_length: Hop length between frames
        win_length: Window length
        window: Window function type
        center: Whether frames were centered
        length: Target length of output audio

    Returns:
        Audio waveform
    """
    audio = librosa.istft(
        stft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        length=length
    )

    return audio


def normalize_audio(
    audio: np.ndarray,
    target_level: float = 1.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize audio to target level.

    Args:
        audio: Audio waveform
        target_level: Target maximum absolute value
        eps: Small constant for numerical stability

    Returns:
        Normalized audio
    """
    max_val = np.abs(audio).max()

    if max_val > eps:
        audio = audio * (target_level / max_val)

    return audio


def compute_magnitude_phase(
    stft: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract magnitude and phase from complex STFT.

    Args:
        stft: Complex STFT

    Returns:
        Tuple of (magnitude, phase)
    """
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    return magnitude, phase


def reconstruct_from_magnitude_phase(
    magnitude: np.ndarray,
    phase: np.ndarray
) -> np.ndarray:
    """
    Reconstruct complex STFT from magnitude and phase.

    Args:
        magnitude: Magnitude spectrogram
        phase: Phase spectrogram

    Returns:
        Complex STFT
    """
    stft = magnitude * np.exp(1j * phase)

    return stft


def griffin_lim(
    magnitude: np.ndarray,
    n_iter: int = 32,
    hop_length: int = 128,
    win_length: int = 512,
    window: str = 'hann',
) -> np.ndarray:
    """
    Griffin-Lim algorithm for phase reconstruction.

    Reconstructs audio from magnitude spectrogram only.
    Useful when phase information is not available.

    Args:
        magnitude: Magnitude spectrogram
        n_iter: Number of iterations
        hop_length: Hop length
        win_length: Window length
        window: Window function type

    Returns:
        Reconstructed audio
    """
    audio = librosa.griffinlim(
        magnitude,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=win_length,
        window=window
    )

    return audio


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int
) -> np.ndarray:
    """
    Resample audio to target sample rate.

    Args:
        audio: Audio waveform
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio
    """
    if orig_sr == target_sr:
        return audio

    audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    return audio


def trim_silence(
    audio: np.ndarray,
    top_db: float = 20,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Trim leading and trailing silence from audio.

    Args:
        audio: Audio waveform
        top_db: Threshold in dB below reference
        frame_length: Frame length for analysis
        hop_length: Hop length for analysis

    Returns:
        Trimmed audio
    """
    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length
    )

    return trimmed


def compute_energy(
    audio: np.ndarray,
    frame_length: int = 512,
    hop_length: int = 128,
) -> np.ndarray:
    """
    Compute frame-wise energy.

    Args:
        audio: Audio waveform
        frame_length: Frame length
        hop_length: Hop length

    Returns:
        Frame-wise energy
    """
    energy = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    return energy


def apply_pre_emphasis(
    audio: np.ndarray,
    coef: float = 0.97
) -> np.ndarray:
    """
    Apply pre-emphasis filter.

    Args:
        audio: Audio waveform
        coef: Pre-emphasis coefficient

    Returns:
        Pre-emphasized audio
    """
    return np.append(audio[0], audio[1:] - coef * audio[:-1])


def apply_de_emphasis(
    audio: np.ndarray,
    coef: float = 0.97
) -> np.ndarray:
    """
    Apply de-emphasis filter.

    Args:
        audio: Pre-emphasized audio waveform
        coef: Pre-emphasis coefficient

    Returns:
        De-emphasized audio
    """
    from scipy import signal
    return signal.lfilter([1], [1, -coef], audio)

"""
Speech Quality Metrics

This module implements various metrics for evaluating speech enhancement
quality including PESQ, STOI, SI-SDR, and SNR.
"""

import numpy as np
from typing import Dict, Optional
import warnings

# Suppress warnings from metric libraries
warnings.filterwarnings('ignore')


def compute_pesq(
    clean: np.ndarray,
    enhanced: np.ndarray,
    sr: int = 16000,
    mode: str = 'wb'
) -> float:
    """
    Compute PESQ (Perceptual Evaluation of Speech Quality).

    PESQ is an ITU-T standard (P.862) for assessing speech quality.
    Range: -0.5 to 4.5 (higher is better)

    Args:
        clean: Clean reference audio
        enhanced: Enhanced audio
        sr: Sample rate (must be 8000 or 16000)
        mode: PESQ mode ('wb' for wideband 16kHz, 'nb' for narrowband 8kHz)

    Returns:
        PESQ score
    """
    try:
        from pesq import pesq

        # Ensure same length
        min_len = min(len(clean), len(enhanced))
        clean = clean[:min_len]
        enhanced = enhanced[:min_len]

        # Compute PESQ
        score = pesq(sr, clean, enhanced, mode)

        return float(score)

    except ImportError:
        print("Warning: pesq library not installed. Install with: pip install pesq")
        return 0.0
    except Exception as e:
        print(f"Error computing PESQ: {e}")
        return 0.0


def compute_stoi(
    clean: np.ndarray,
    enhanced: np.ndarray,
    sr: int = 16000,
    extended: bool = False
) -> float:
    """
    Compute STOI (Short-Time Objective Intelligibility).

    STOI predicts speech intelligibility.
    Range: 0 to 1 (higher is better)

    Args:
        clean: Clean reference audio
        enhanced: Enhanced audio
        sr: Sample rate
        extended: Whether to use extended STOI (provides negative values too)

    Returns:
        STOI score
    """
    try:
        from pystoi import stoi

        # Ensure same length
        min_len = min(len(clean), len(enhanced))
        clean = clean[:min_len]
        enhanced = enhanced[:min_len]

        # Compute STOI
        score = stoi(clean, enhanced, sr, extended=extended)

        return float(score)

    except ImportError:
        print("Warning: pystoi library not installed. Install with: pip install pystoi")
        return 0.0
    except Exception as e:
        print(f"Error computing STOI: {e}")
        return 0.0


def compute_sisdr(
    clean: np.ndarray,
    enhanced: np.ndarray,
    eps: float = 1e-8
) -> float:
    """
    Compute SI-SDR (Scale-Invariant Signal-to-Distortion Ratio).

    SI-SDR is commonly used in speech separation and enhancement.
    Range: -inf to +inf dB (higher is better, typical range: -10 to 20 dB)

    Args:
        clean: Clean reference audio
        enhanced: Enhanced audio
        eps: Small constant for numerical stability

    Returns:
        SI-SDR in dB
    """
    try:
        # Ensure same length
        min_len = min(len(clean), len(enhanced))
        clean = clean[:min_len]
        enhanced = enhanced[:min_len]

        # Zero-mean normalization
        clean = clean - np.mean(clean)
        enhanced = enhanced - np.mean(enhanced)

        # Compute scale factor
        alpha = np.dot(enhanced, clean) / (np.dot(clean, clean) + eps)

        # Scale target
        clean_scaled = alpha * clean

        # Compute SI-SDR
        numerator = np.sum(clean_scaled ** 2)
        denominator = np.sum((enhanced - clean_scaled) ** 2)

        sisdr = 10 * np.log10(numerator / (denominator + eps))

        return float(sisdr)

    except Exception as e:
        print(f"Error computing SI-SDR: {e}")
        return 0.0


def compute_snr(
    clean: np.ndarray,
    noisy_or_enhanced: np.ndarray,
    eps: float = 1e-8
) -> float:
    """
    Compute SNR (Signal-to-Noise Ratio).

    Args:
        clean: Clean reference audio
        noisy_or_enhanced: Noisy or enhanced audio
        eps: Small constant for numerical stability

    Returns:
        SNR in dB
    """
    try:
        # Ensure same length
        min_len = min(len(clean), len(noisy_or_enhanced))
        clean = clean[:min_len]
        noisy_or_enhanced = noisy_or_enhanced[:min_len]

        # Compute noise
        noise = noisy_or_enhanced - clean

        # Compute powers
        signal_power = np.mean(clean ** 2)
        noise_power = np.mean(noise ** 2)

        # Compute SNR
        snr = 10 * np.log10(signal_power / (noise_power + eps))

        return float(snr)

    except Exception as e:
        print(f"Error computing SNR: {e}")
        return 0.0


def compute_segsnr(
    clean: np.ndarray,
    enhanced: np.ndarray,
    frame_length: int = 512,
    hop_length: int = 256,
    eps: float = 1e-8
) -> float:
    """
    Compute Segmental SNR.

    Computes SNR per frame and averages. More robust to silent regions.

    Args:
        clean: Clean reference audio
        enhanced: Enhanced audio
        frame_length: Frame length in samples
        hop_length: Hop length between frames
        eps: Small constant for numerical stability

    Returns:
        Segmental SNR in dB
    """
    try:
        # Ensure same length
        min_len = min(len(clean), len(enhanced))
        clean = clean[:min_len]
        enhanced = enhanced[:min_len]

        # Compute frame-wise SNR
        snr_frames = []

        for i in range(0, len(clean) - frame_length, hop_length):
            clean_frame = clean[i:i + frame_length]
            enhanced_frame = enhanced[i:i + frame_length]

            noise_frame = enhanced_frame - clean_frame

            signal_power = np.mean(clean_frame ** 2)
            noise_power = np.mean(noise_frame ** 2)

            if signal_power > eps and noise_power > eps:
                snr_frame = 10 * np.log10(signal_power / noise_power)
                # Clip to reasonable range
                snr_frame = np.clip(snr_frame, -10, 35)
                snr_frames.append(snr_frame)

        # Average over frames
        if snr_frames:
            segsnr = np.mean(snr_frames)
        else:
            segsnr = 0.0

        return float(segsnr)

    except Exception as e:
        print(f"Error computing SegSNR: {e}")
        return 0.0


def compute_lsd(
    clean: np.ndarray,
    enhanced: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 128,
    eps: float = 1e-10
) -> float:
    """
    Compute Log-Spectral Distance.

    Measures the difference between spectrograms in log domain.

    Args:
        clean: Clean reference audio
        enhanced: Enhanced audio
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        eps: Small constant for numerical stability

    Returns:
        LSD in dB
    """
    try:
        import librosa

        # Ensure same length
        min_len = min(len(clean), len(enhanced))
        clean = clean[:min_len]
        enhanced = enhanced[:min_len]

        # Compute spectrograms
        clean_spec = np.abs(librosa.stft(clean, n_fft=n_fft, hop_length=hop_length))
        enhanced_spec = np.abs(librosa.stft(enhanced, n_fft=n_fft, hop_length=hop_length))

        # Compute log-spectral distance
        lsd = np.mean(
            np.sqrt(
                np.mean(
                    (np.log(clean_spec + eps) - np.log(enhanced_spec + eps)) ** 2,
                    axis=0
                )
            )
        )

        return float(lsd)

    except Exception as e:
        print(f"Error computing LSD: {e}")
        return 0.0


def compute_all_metrics(
    clean: np.ndarray,
    enhanced: np.ndarray,
    noisy: Optional[np.ndarray] = None,
    sr: int = 16000
) -> Dict[str, float]:
    """
    Compute all available metrics.

    Args:
        clean: Clean reference audio
        enhanced: Enhanced audio
        noisy: Noisy input audio (optional, for computing improvement)
        sr: Sample rate

    Returns:
        Dictionary of metric scores
    """
    metrics = {}

    # PESQ
    metrics['pesq'] = compute_pesq(clean, enhanced, sr)

    # STOI
    metrics['stoi'] = compute_stoi(clean, enhanced, sr)

    # SI-SDR
    metrics['sisdr'] = compute_sisdr(clean, enhanced)

    # SNR
    metrics['snr'] = compute_snr(clean, enhanced)

    # Segmental SNR
    metrics['segsnr'] = compute_segsnr(clean, enhanced)

    # LSD
    metrics['lsd'] = compute_lsd(clean, enhanced, sr)

    # If noisy audio provided, compute improvements
    if noisy is not None:
        noisy_pesq = compute_pesq(clean, noisy, sr)
        noisy_stoi = compute_stoi(clean, noisy, sr)
        noisy_sisdr = compute_sisdr(clean, noisy)
        noisy_snr = compute_snr(clean, noisy)

        metrics['pesq_improvement'] = metrics['pesq'] - noisy_pesq
        metrics['stoi_improvement'] = metrics['stoi'] - noisy_stoi
        metrics['sisdr_improvement'] = metrics['sisdr'] - noisy_sisdr
        metrics['snr_improvement'] = metrics['snr'] - noisy_snr

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics
        title: Title to display
    """
    print(f"\n{title}")
    print("=" * 50)

    for metric_name, value in metrics.items():
        # Format metric name
        name = metric_name.upper().replace('_', ' ')
        print(f"{name:30s}: {value:8.4f}")

    print("=" * 50)

"""
Visualization Utilities

This module provides utilities for visualizing audio waveforms,
spectrograms, attention maps, and training curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path
from typing import Optional, Tuple, List, Union


def plot_waveform(
    audio: np.ndarray,
    sr: int = 16000,
    title: str = 'Waveform',
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot audio waveform.

    Args:
        audio: Audio waveform
        sr: Sample rate
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    librosa.display.waveshow(audio, sr=sr, ax=ax)

    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved waveform plot to {save_path}")

    return fig


def plot_spectrogram(
    spectrogram: np.ndarray,
    sr: int = 16000,
    hop_length: int = 128,
    title: str = 'Spectrogram',
    cmap: str = 'viridis',
    figsize: Tuple[int, int] = (12, 6),
    db_scale: bool = True,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot spectrogram.

    Args:
        spectrogram: Magnitude spectrogram
        sr: Sample rate
        hop_length: Hop length used for STFT
        title: Plot title
        cmap: Colormap
        figsize: Figure size
        db_scale: Convert to dB scale
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to dB if requested
    if db_scale:
        spec_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
        img = librosa.display.specshow(
            spec_db,
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='hz',
            cmap=cmap,
            ax=ax
        )
        cbar = plt.colorbar(img, ax=ax, format='%+2.0f dB')
    else:
        img = librosa.display.specshow(
            spectrogram,
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='hz',
            cmap=cmap,
            ax=ax
        )
        cbar = plt.colorbar(img, ax=ax)

    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved spectrogram plot to {save_path}")

    return fig


def plot_comparison(
    noisy: np.ndarray,
    clean: np.ndarray,
    enhanced: np.ndarray,
    sr: int = 16000,
    hop_length: int = 128,
    plot_type: str = 'both',
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot comparison of noisy, clean, and enhanced audio.

    Args:
        noisy: Noisy audio
        clean: Clean reference audio
        enhanced: Enhanced audio
        sr: Sample rate
        hop_length: Hop length
        plot_type: 'waveform', 'spectrogram', or 'both'
        figsize: Figure size
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    if plot_type == 'both':
        fig, axes = plt.subplots(3, 2, figsize=figsize)

        # Waveforms
        librosa.display.waveshow(noisy, sr=sr, ax=axes[0, 0])
        axes[0, 0].set_title('Noisy Waveform')
        axes[0, 0].set_xlabel('Time (s)')

        librosa.display.waveshow(clean, sr=sr, ax=axes[1, 0])
        axes[1, 0].set_title('Clean Waveform')
        axes[1, 0].set_xlabel('Time (s)')

        librosa.display.waveshow(enhanced, sr=sr, ax=axes[2, 0])
        axes[2, 0].set_title('Enhanced Waveform')
        axes[2, 0].set_xlabel('Time (s)')

        # Spectrograms
        noisy_spec = np.abs(librosa.stft(noisy))
        clean_spec = np.abs(librosa.stft(clean))
        enhanced_spec = np.abs(librosa.stft(enhanced))

        librosa.display.specshow(
            librosa.amplitude_to_db(noisy_spec, ref=np.max),
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='hz',
            ax=axes[0, 1],
            cmap='viridis'
        )
        axes[0, 1].set_title('Noisy Spectrogram')

        librosa.display.specshow(
            librosa.amplitude_to_db(clean_spec, ref=np.max),
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='hz',
            ax=axes[1, 1],
            cmap='viridis'
        )
        axes[1, 1].set_title('Clean Spectrogram')

        librosa.display.specshow(
            librosa.amplitude_to_db(enhanced_spec, ref=np.max),
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='hz',
            ax=axes[2, 1],
            cmap='viridis'
        )
        axes[2, 1].set_title('Enhanced Spectrogram')

    elif plot_type == 'waveform':
        fig, axes = plt.subplots(3, 1, figsize=figsize)

        librosa.display.waveshow(noisy, sr=sr, ax=axes[0])
        axes[0].set_title('Noisy Waveform')

        librosa.display.waveshow(clean, sr=sr, ax=axes[1])
        axes[1].set_title('Clean Waveform')

        librosa.display.waveshow(enhanced, sr=sr, ax=axes[2])
        axes[2].set_title('Enhanced Waveform')

    elif plot_type == 'spectrogram':
        fig, axes = plt.subplots(3, 1, figsize=figsize)

        noisy_spec = np.abs(librosa.stft(noisy))
        clean_spec = np.abs(librosa.stft(clean))
        enhanced_spec = np.abs(librosa.stft(enhanced))

        librosa.display.specshow(
            librosa.amplitude_to_db(noisy_spec, ref=np.max),
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='hz',
            ax=axes[0],
            cmap='viridis'
        )
        axes[0].set_title('Noisy Spectrogram')

        librosa.display.specshow(
            librosa.amplitude_to_db(clean_spec, ref=np.max),
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='hz',
            ax=axes[1],
            cmap='viridis'
        )
        axes[1].set_title('Clean Spectrogram')

        librosa.display.specshow(
            librosa.amplitude_to_db(enhanced_spec, ref=np.max),
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='hz',
            ax=axes[2],
            cmap='viridis'
        )
        axes[2].set_title('Enhanced Spectrogram')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")

    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = 'Training Curves',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses (optional)
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)

    if val_losses:
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    return fig


def plot_attention_map(
    attention: np.ndarray,
    title: str = 'Attention Map',
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'viridis',
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot attention map from transformer.

    Args:
        attention: Attention weights [num_heads, seq_len, seq_len] or [seq_len, seq_len]
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    if attention.ndim == 3:
        # Multiple heads: plot first head
        attention = attention[0]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(attention, cmap=cmap, aspect='auto')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)

    plt.colorbar(im, ax=ax, label='Attention Weight')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention map to {save_path}")

    return fig


def plot_metrics_comparison(
    metrics_dict: dict,
    title: str = 'Metrics Comparison',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot bar chart comparing different metrics.

    Args:
        metrics_dict: Dictionary of metrics {metric_name: value}
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    bars = ax.bar(metrics, values, color='steelblue', alpha=0.8)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics comparison to {save_path}")

    return fig

"""
Loss Functions for Speech Enhancement

This module implements various loss functions including L1, MSE,
STOI-based loss, and combined losses for speech enhancement training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np


class SpectrogramLoss(nn.Module):
    """
    Loss function for spectrogram-based speech enhancement.

    Supports L1, L2 (MSE), and combinations of both.
    """

    def __init__(
        self,
        loss_type: str = 'l1',
        reduction: str = 'mean',
        use_log_compression: bool = False,
    ):
        """
        Initialize spectrogram loss.

        Args:
            loss_type: Type of loss ('l1', 'mse', 'l1+mse')
            reduction: Reduction method ('mean', 'sum', 'none')
            use_log_compression: Apply log compression before computing loss
        """
        super().__init__()

        self.loss_type = loss_type
        self.reduction = reduction
        self.use_log_compression = use_log_compression

        # Loss functions
        self.l1_loss = nn.L1Loss(reduction=reduction)
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def _apply_log_compression(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Apply log compression to spectrogram.

        Args:
            x: Input spectrogram
            eps: Small constant for numerical stability

        Returns:
            Log-compressed spectrogram
        """
        return torch.log(x + eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute spectrogram loss.

        Args:
            pred: Predicted spectrogram [B, C, F, T]
            target: Target clean spectrogram [B, C, F, T]

        Returns:
            Loss value
        """
        # Apply log compression if enabled
        if self.use_log_compression:
            pred = self._apply_log_compression(pred)
            target = self._apply_log_compression(target)

        # Compute loss based on type
        if self.loss_type == 'l1':
            loss = self.l1_loss(pred, target)
        elif self.loss_type == 'mse':
            loss = self.mse_loss(pred, target)
        elif self.loss_type == 'l1+mse':
            loss = self.l1_loss(pred, target) + self.mse_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss


class STOILoss(nn.Module):
    """
    STOI-based loss for speech enhancement.

    Note: This is a simplified differentiable approximation of STOI.
    For actual STOI metric computation, use the evaluation module.

    The actual STOI computation requires waveforms and is not differentiable.
    This approximation uses correlation between spectrograms as a proxy.
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Initialize STOI loss.

        Args:
            reduction: Reduction method
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute STOI-based loss.

        Uses negative correlation as loss (higher correlation = lower loss).

        Args:
            pred: Predicted spectrogram [B, C, F, T]
            target: Target spectrogram [B, C, F, T]

        Returns:
            Loss value
        """
        # Flatten spectrograms for correlation computation
        pred_flat = pred.flatten(1)
        target_flat = target.flatten(1)

        # Normalize
        pred_norm = F.normalize(pred_flat, dim=1)
        target_norm = F.normalize(target_flat, dim=1)

        # Compute correlation (cosine similarity)
        correlation = (pred_norm * target_norm).sum(dim=1)

        # Loss is negative correlation (we want to maximize correlation)
        loss = 1.0 - correlation

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss.

    Computes loss at multiple STFT resolutions to capture both
    fine-grained and coarse-grained spectral information.

    This requires waveform inputs rather than spectrograms.
    """

    def __init__(
        self,
        fft_sizes: list = [512, 1024, 2048],
        hop_sizes: list = [128, 256, 512],
        win_sizes: list = [512, 1024, 2048],
        reduction: str = 'mean',
    ):
        """
        Initialize multi-resolution STFT loss.

        Args:
            fft_sizes: List of FFT sizes
            hop_sizes: List of hop lengths
            win_sizes: List of window sizes
            reduction: Reduction method
        """
        super().__init__()

        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes), \
            "All STFT parameter lists must have the same length"

        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes
        self.reduction = reduction

    def stft(
        self,
        x: torch.Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
    ) -> torch.Tensor:
        """
        Compute STFT.

        Args:
            x: Waveform [B, T]
            n_fft: FFT size
            hop_length: Hop length
            win_length: Window size

        Returns:
            STFT magnitude [B, F, T]
        """
        # Create window
        window = torch.hann_window(win_length).to(x.device)

        # Compute STFT
        stft = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )

        # Get magnitude
        magnitude = torch.abs(stft)

        return magnitude

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-resolution STFT loss.

        Args:
            pred: Predicted waveform [B, T]
            target: Target waveform [B, T]

        Returns:
            Combined loss across all resolutions
        """
        total_loss = 0.0

        for n_fft, hop_length, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_sizes
        ):
            # Compute STFTs
            pred_stft = self.stft(pred, n_fft, hop_length, win_length)
            target_stft = self.stft(target, n_fft, hop_length, win_length)

            # Spectral convergence loss
            sc_loss = torch.norm(target_stft - pred_stft, p='fro') / \
                      torch.norm(target_stft, p='fro')

            # Log STFT magnitude loss
            log_loss = F.l1_loss(
                torch.log(pred_stft + 1e-5),
                torch.log(target_stft + 1e-5)
            )

            total_loss += sc_loss + log_loss

        # Average over resolutions
        total_loss /= len(self.fft_sizes)

        return total_loss


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained features.

    Note: This is a placeholder. For production use, you would load
    a pre-trained audio model (e.g., VGGish, Wav2Vec) and compute
    loss in feature space.
    """

    def __init__(self):
        """Initialize perceptual loss."""
        super().__init__()
        # Placeholder: In production, load pre-trained model here
        self.feature_extractor = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.

        Args:
            pred: Predicted spectrogram
            target: Target spectrogram

        Returns:
            Loss value
        """
        # Placeholder: Return L1 loss for now
        # In production, extract features and compute loss in feature space
        return F.l1_loss(pred, target)


class CombinedLoss(nn.Module):
    """
    Combined loss function for speech enhancement.

    Combines multiple loss functions with configurable weights.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        mse_weight: float = 0.0,
        stoi_weight: float = 0.1,
        perceptual_weight: float = 0.0,
        use_log_compression: bool = False,
    ):
        """
        Initialize combined loss.

        Args:
            l1_weight: Weight for L1 loss
            mse_weight: Weight for MSE loss
            stoi_weight: Weight for STOI loss
            perceptual_weight: Weight for perceptual loss
            use_log_compression: Apply log compression to spectrograms
        """
        super().__init__()

        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.stoi_weight = stoi_weight
        self.perceptual_weight = perceptual_weight

        # Initialize loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.stoi_loss = STOILoss()
        self.perceptual_loss = PerceptualLoss()

        self.use_log_compression = use_log_compression

    def _apply_log_compression(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Apply log compression to spectrogram."""
        return torch.log(x + eps)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            pred: Predicted spectrogram [B, C, F, T]
            target: Target spectrogram [B, C, F, T]
            return_components: Whether to return individual loss components

        Returns:
            Total loss value
            Optionally, dictionary of individual loss components
        """
        losses = {}
        total_loss = 0.0

        # Prepare inputs
        pred_input = pred
        target_input = target

        if self.use_log_compression:
            pred_input = self._apply_log_compression(pred)
            target_input = self._apply_log_compression(target)

        # L1 loss
        if self.l1_weight > 0:
            l1 = self.l1_loss(pred_input, target_input)
            losses['l1'] = l1.item()
            total_loss += self.l1_weight * l1

        # MSE loss
        if self.mse_weight > 0:
            mse = self.mse_loss(pred_input, target_input)
            losses['mse'] = mse.item()
            total_loss += self.mse_weight * mse

        # STOI loss
        if self.stoi_weight > 0:
            stoi = self.stoi_loss(pred, target)
            losses['stoi'] = stoi.item()
            total_loss += self.stoi_weight * stoi

        # Perceptual loss
        if self.perceptual_weight > 0:
            perceptual = self.perceptual_loss(pred, target)
            losses['perceptual'] = perceptual.item()
            total_loss += self.perceptual_weight * perceptual

        losses['total'] = total_loss.item()

        if return_components:
            return total_loss, losses
        return total_loss


def create_loss_function(config: Dict) -> nn.Module:
    """
    Create loss function from configuration.

    Args:
        config: Loss configuration dictionary

    Returns:
        Loss function module
    """
    loss_config = config.get('loss', {})

    return CombinedLoss(
        l1_weight=loss_config.get('l1_weight', 1.0),
        mse_weight=loss_config.get('mse_weight', 0.0),
        stoi_weight=loss_config.get('stoi_weight', 0.1),
        perceptual_weight=loss_config.get('perceptual_weight', 0.0),
        use_log_compression=loss_config.get('use_log_compression', False),
    )

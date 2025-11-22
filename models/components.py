"""
Neural Network Components

This module provides reusable building blocks for the Hybrid Vision Transformer,
including convolutional blocks, feed-forward networks, and embedding layers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ConvBlock(nn.Module):
    """
    Convolutional block with Conv2d, BatchNorm, Activation, and optional Pooling.

    This is a standard building block for the CNN encoder. It extracts
    local features from spectrograms while progressively reducing spatial dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool_size: Optional[int] = 2,
        activation: str = 'relu',
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize convolutional block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            pool_size: Max pooling size (None to disable pooling)
            activation: Activation function ('relu', 'gelu', 'leaky_relu')
            use_batchnorm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()

        layers = []

        # Convolution
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not use_batchnorm,  # No bias if using batchnorm
            )
        )

        # Batch normalization
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        # Activation
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        # Pooling
        if pool_size is not None and pool_size > 1:
            layers.append(nn.MaxPool2d(kernel_size=pool_size))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Output tensor [B, C', H', W']
        """
        return self.block(x)


class TransposeConvBlock(nn.Module):
    """
    Transpose convolutional block for upsampling in the decoder.

    Uses transposed convolution (deconvolution) to increase spatial resolution
    while maintaining feature quality.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        output_padding: int = 0,
        upsample_factor: Optional[int] = 2,
        activation: str = 'relu',
        use_batchnorm: bool = True,
        dropout: float = 0.0,
        final_layer: bool = False,
    ):
        """
        Initialize transpose convolutional block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size
            stride: Stride
            padding: Padding
            output_padding: Output padding for transpose conv
            upsample_factor: Upsampling factor (None to disable)
            activation: Activation function
            use_batchnorm: Whether to use batch normalization
            dropout: Dropout probability
            final_layer: If True, use Tanh activation for output layer
        """
        super().__init__()

        layers = []

        # Upsampling
        if upsample_factor is not None and upsample_factor > 1:
            layers.append(nn.Upsample(scale_factor=upsample_factor, mode='nearest'))

        # Transpose convolution (or regular conv after upsampling)
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not use_batchnorm,
            )
        )

        # Batch normalization (not for final layer)
        if use_batchnorm and not final_layer:
            layers.append(nn.BatchNorm2d(out_channels))

        # Activation
        if final_layer:
            # Final layer uses Tanh to bound output
            layers.append(nn.Tanh())
        else:
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Dropout
        if dropout > 0 and not final_layer:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Output tensor [B, C', H', W']
        """
        return self.block(x)


class FeedForward(nn.Module):
    """
    Feed-forward network for Transformer blocks.

    Implements the MLP component of Transformer:
    FFN(x) = max(0, xW1 + b1)W2 + b2

    Uses GELU activation as in BERT and GPT.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """
        Initialize feed-forward network.

        Args:
            dim: Input and output dimension
            hidden_dim: Hidden layer dimension (default: 4 * dim)
            dropout: Dropout probability
        """
        super().__init__()

        hidden_dim = hidden_dim or 4 * dim

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [..., dim]

        Returns:
            Output tensor [..., dim]
        """
        return self.net(x)


class PatchEmbedding(nn.Module):
    """
    Convert feature maps to patch embeddings for Vision Transformer.

    Splits the feature map into patches and projects them to embedding space.
    This is a crucial component that bridges CNNs and Transformers.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int = 4,
        flatten: bool = True,
    ):
        """
        Initialize patch embedding layer.

        Args:
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            patch_size: Size of patches (height and width)
            flatten: Whether to flatten patches
        """
        super().__init__()

        self.patch_size = patch_size
        self.flatten = flatten

        # Use convolution to extract and project patches
        # This is equivalent to splitting into patches and linear projection
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Forward pass.

        Args:
            x: Input feature map [B, C, H, W]

        Returns:
            Tuple of:
                - Patch embeddings [B, num_patches, embed_dim] if flatten=True
                  or [B, embed_dim, H', W'] if flatten=False
                - Original spatial dimensions (H', W') after patching
        """
        # Project patches
        x = self.projection(x)  # [B, embed_dim, H', W']

        # Save spatial dimensions
        B, C, H, W = x.shape
        spatial_shape = (H, W)

        if self.flatten:
            # Flatten spatial dimensions
            x = x.flatten(2)  # [B, embed_dim, H'*W']
            x = x.transpose(1, 2)  # [B, num_patches, embed_dim]

        return x, spatial_shape


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer.

    Adds positional information to patch embeddings since Transformers
    don't have inherent notion of position. Can use either learnable
    embeddings or fixed sinusoidal encodings.
    """

    def __init__(
        self,
        embed_dim: int,
        max_len: int = 5000,
        learnable: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize positional encoding.

        Args:
            embed_dim: Embedding dimension
            max_len: Maximum sequence length
            learnable: If True, use learnable embeddings; else use sinusoidal
            dropout: Dropout probability
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.learnable = learnable
        self.dropout = nn.Dropout(dropout)

        if learnable:
            # Learnable positional embeddings
            self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            # Fixed sinusoidal positional encodings
            self.register_buffer('pos_embed', self._sinusoidal_encoding(max_len, embed_dim))

    def _sinusoidal_encoding(self, max_len: int, embed_dim: int) -> torch.Tensor:
        """
        Create sinusoidal positional encodings.

        Args:
            max_len: Maximum sequence length
            embed_dim: Embedding dimension

        Returns:
            Positional encodings [1, max_len, embed_dim]
        """
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )

        pos_embed = torch.zeros(1, max_len, embed_dim)
        pos_embed[0, :, 0::2] = torch.sin(position * div_term)
        pos_embed[0, :, 1::2] = torch.cos(position * div_term)

        return pos_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor [B, N, embed_dim]

        Returns:
            Output with positional encoding [B, N, embed_dim]
        """
        B, N, D = x.shape

        # Add positional encoding
        x = x + self.pos_embed[:, :N, :]

        return self.dropout(x)


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.

    This is a regularization technique that randomly drops entire
    residual branches during training. Helps prevent overfitting.
    """

    def __init__(self, drop_prob: float = 0.0):
        """
        Initialize drop path.

        Args:
            drop_prob: Probability of dropping path
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor (possibly dropped)
        """
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize

        output = x.div(keep_prob) * random_tensor

        return output

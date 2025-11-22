"""
Attention Mechanisms for Vision Transformer

This module implements multi-head self-attention and transformer encoder blocks
for the Hybrid Vision Transformer architecture.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .components import FeedForward, DropPath


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.

    Implements the core attention mechanism from "Attention is All You Need".
    Multiple attention heads allow the model to jointly attend to information
    from different representation subspaces.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        """
        Initialize multi-head self-attention.

        Args:
            embed_dim: Embedding dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projections
            attn_dropout: Dropout probability for attention weights
            proj_dropout: Dropout probability for output projection
        """
        super().__init__()

        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1/sqrt(d_k) for scaled dot-product

        # QKV projection (combined for efficiency)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)

        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of multi-head self-attention.

        Args:
            x: Input tensor [B, N, embed_dim]
            mask: Optional attention mask [B, N] or [B, N, N]
            return_attention: Whether to return attention weights

        Returns:
            Output tensor [B, N, embed_dim]
            Optionally, attention weights [B, num_heads, N, N]
        """
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)  # [B, N, 3 * embed_dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, num_heads, N, head_dim]

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                # Expand mask to [B, 1, 1, N] for broadcasting
                mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        # Softmax to get attention weights
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        x = (attn @ v)  # [B, num_heads, N, head_dim]
        x = x.transpose(1, 2)  # [B, N, num_heads, head_dim]
        x = x.reshape(B, N, C)  # [B, N, embed_dim]

        # Output projection
        x = self.proj(x)
        x = self.proj_dropout(x)

        if return_attention:
            return x, attn
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block.

    Combines multi-head self-attention and feed-forward network with
    residual connections and layer normalization. This is the fundamental
    building block of the Vision Transformer.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        """
        Initialize transformer encoder block.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            qkv_bias: Whether to use bias in attention QKV projection
            dropout: Dropout probability
            attn_dropout: Attention dropout probability
            drop_path: Stochastic depth probability
        """
        super().__init__()

        # Layer normalization (pre-norm architecture)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Multi-head self-attention
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
        )

        # Feed-forward network
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = FeedForward(
            dim=embed_dim,
            hidden_dim=mlp_hidden_dim,
            dropout=dropout,
        )

        # Stochastic depth (drop path)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of transformer encoder block.

        Uses pre-norm architecture:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

        Args:
            x: Input tensor [B, N, embed_dim]
            mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            Output tensor [B, N, embed_dim]
            Optionally, attention weights
        """
        # Self-attention with residual connection
        if return_attention:
            attn_out, attn_weights = self.attn(
                self.norm1(x),
                mask=mask,
                return_attention=True
            )
            x = x + self.drop_path(attn_out)
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))

        # Feed-forward with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if return_attention:
            return x, attn_weights
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer Encoder.

    Stack of Transformer encoder blocks for processing patch embeddings.
    This forms the core of the Vision Transformer architecture.
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        """
        Initialize Vision Transformer encoder.

        Args:
            embed_dim: Embedding dimension
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            qkv_bias: Use bias in QKV projection
            dropout: Dropout probability
            attn_dropout: Attention dropout probability
            drop_path_rate: Stochastic depth rate (linearly increases per layer)
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Stochastic depth decay rule (linearly increase drop path rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        # Stack of transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                attn_dropout=attn_dropout,
                drop_path=dpr[i],
            )
            for i in range(num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_all_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through transformer encoder.

        Args:
            x: Input embeddings [B, N, embed_dim]
            mask: Optional attention mask
            return_all_attentions: Whether to return attention from all layers

        Returns:
            Output embeddings [B, N, embed_dim]
            Optionally, list of attention weights from all layers
        """
        all_attentions = []

        for block in self.blocks:
            if return_all_attentions:
                x, attn = block(x, mask=mask, return_attention=True)
                all_attentions.append(attn)
            else:
                x = block(x, mask=mask)

        x = self.norm(x)

        if return_all_attentions:
            return x, all_attentions
        return x


class EfficientAttention(nn.Module):
    """
    Efficient attention mechanism for long sequences.

    Uses linear attention approximation to reduce complexity from O(N^2) to O(N).
    Useful for very long spectrograms.

    Note: This is an optional component for future optimization.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize efficient attention.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            qkv_bias: Use bias in QKV projection
            dropout: Dropout probability
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with linear attention.

        Args:
            x: Input tensor [B, N, embed_dim]

        Returns:
            Output tensor [B, N, embed_dim]
        """
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply softmax to K and V dimensions separately (linear attention)
        k = k.softmax(dim=-2)
        v = v.softmax(dim=-1)

        # Linear attention: Q(K^T V) instead of (QK^T)V
        context = torch.einsum('bhnd,bhne->bhde', k, v)
        x = torch.einsum('bhnd,bhde->bhne', q, context)

        # Reshape and project
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)

        return x

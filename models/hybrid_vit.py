"""
Hybrid Vision Transformer for Speech Enhancement

This module implements the complete Hybrid ViT architecture combining
CNN encoders, Vision Transformer, and CNN decoders for speech enhancement.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List

from .components import (
    ConvBlock,
    TransposeConvBlock,
    PatchEmbedding,
    PositionalEncoding,
)
from .attention import VisionTransformer


class HybridViT(nn.Module):
    """
    Hybrid Vision Transformer for Speech Enhancement.

    Architecture:
    1. CNN Encoder: Extract local acoustic features from spectrogram
    2. Patch Embedding: Convert feature maps to sequence of patches
    3. Vision Transformer: Capture global temporal dependencies
    4. CNN Decoder: Reconstruct enhanced spectrogram

    This hybrid approach combines the best of both worlds:
    - CNNs excel at local feature extraction
    - Transformers excel at modeling long-range dependencies
    """

    def __init__(
        self,
        # Input/Output
        input_channels: int = 1,
        output_channels: int = 1,

        # CNN Encoder
        encoder_channels: List[int] = [64, 128, 256],
        encoder_kernel_sizes: List[int] = [3, 3, 3],
        encoder_pool_sizes: List[int] = [2, 2, 1],

        # Vision Transformer
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        patch_size: int = 4,

        # CNN Decoder
        decoder_channels: List[int] = [256, 128, 64, 1],
        decoder_kernel_sizes: List[int] = [3, 3, 3, 3],
        decoder_upsample_factors: List[int] = [1, 2, 2, 1],

        # Regularization
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        drop_path_rate: float = 0.1,

        # Architecture options
        use_skip_connections: bool = True,
        use_cls_token: bool = False,
    ):
        """
        Initialize Hybrid Vision Transformer.

        Args:
            input_channels: Number of input channels (1 for magnitude spec)
            output_channels: Number of output channels (1 for enhanced spec)
            encoder_channels: Channel progression for CNN encoder
            encoder_kernel_sizes: Kernel sizes for encoder conv layers
            encoder_pool_sizes: Pooling sizes for encoder (1 means no pooling)
            embed_dim: Transformer embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            mlp_ratio: MLP hidden dimension ratio
            patch_size: Patch size for Vision Transformer
            decoder_channels: Channel progression for CNN decoder
            decoder_kernel_sizes: Kernel sizes for decoder conv layers
            decoder_upsample_factors: Upsampling factors for decoder
            dropout: Dropout probability
            attn_dropout: Attention dropout probability
            drop_path_rate: Stochastic depth rate
            use_skip_connections: Use U-Net style skip connections
            use_cls_token: Add CLS token to transformer input
        """
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.use_skip_connections = use_skip_connections
        self.use_cls_token = use_cls_token

        # ===== CNN Encoder =====
        self.encoder = self._build_encoder(
            input_channels,
            encoder_channels,
            encoder_kernel_sizes,
            encoder_pool_sizes,
            dropout,
        )

        # ===== Patch Embedding =====
        encoder_out_channels = encoder_channels[-1]
        self.patch_embed = PatchEmbedding(
            in_channels=encoder_out_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            flatten=True,
        )

        # ===== CLS Token (optional) =====
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        # ===== Positional Encoding =====
        max_patches = 10000  # Maximum number of patches
        self.pos_encoding = PositionalEncoding(
            embed_dim=embed_dim,
            max_len=max_patches,
            learnable=True,
            dropout=dropout,
        )

        # ===== Vision Transformer =====
        self.transformer = VisionTransformer(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            dropout=dropout,
            attn_dropout=attn_dropout,
            drop_path_rate=drop_path_rate,
        )

        # ===== Projection back to feature map =====
        self.to_feature_map = nn.Linear(embed_dim, encoder_out_channels)

        # ===== CNN Decoder =====
        self.decoder = self._build_decoder(
            decoder_channels,
            decoder_kernel_sizes,
            decoder_upsample_factors,
            dropout,
        )

        # ===== Skip Connection Projections (U-Net style) =====
        if use_skip_connections:
            self.skip_projections = nn.ModuleList([
                nn.Conv2d(enc_ch, dec_ch, kernel_size=1)
                for enc_ch, dec_ch in zip(
                    encoder_channels[::-1],
                    decoder_channels[:-1]
                )
            ])
        else:
            self.skip_projections = None

        # Initialize weights
        self.apply(self._init_weights)

    def _build_encoder(
        self,
        input_channels: int,
        channels: List[int],
        kernel_sizes: List[int],
        pool_sizes: List[int],
        dropout: float,
    ) -> nn.ModuleList:
        """
        Build CNN encoder.

        Args:
            input_channels: Number of input channels
            channels: List of output channels for each block
            kernel_sizes: List of kernel sizes
            pool_sizes: List of pooling sizes
            dropout: Dropout probability

        Returns:
            ModuleList of encoder blocks
        """
        encoder = nn.ModuleList()
        in_ch = input_channels

        for out_ch, kernel_size, pool_size in zip(channels, kernel_sizes, pool_sizes):
            encoder.append(
                ConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    pool_size=pool_size if pool_size > 1 else None,
                    activation='relu',
                    use_batchnorm=True,
                    dropout=dropout,
                )
            )
            in_ch = out_ch

        return encoder

    def _build_decoder(
        self,
        channels: List[int],
        kernel_sizes: List[int],
        upsample_factors: List[int],
        dropout: float,
    ) -> nn.ModuleList:
        """
        Build CNN decoder.

        Args:
            channels: List of output channels for each block
            kernel_sizes: List of kernel sizes
            upsample_factors: List of upsampling factors
            dropout: Dropout probability

        Returns:
            ModuleList of decoder blocks
        """
        decoder = nn.ModuleList()

        for i, (out_ch, kernel_size, upsample) in enumerate(
            zip(channels, kernel_sizes, upsample_factors)
        ):
            # Input channels (accounting for skip connections)
            if i == 0:
                in_ch = channels[0]  # From transformer output
            else:
                in_ch = channels[i - 1]

            # Double input channels if using skip connections (except last layer)
            if self.use_skip_connections and i < len(channels) - 1:
                in_ch = in_ch * 2

            is_final = (i == len(channels) - 1)

            decoder.append(
                TransposeConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    upsample_factor=upsample if upsample > 1 else None,
                    activation='relu',
                    use_batchnorm=True,
                    dropout=dropout if not is_final else 0.0,
                    final_layer=is_final,
                )
            )

        return decoder

    def _init_weights(self, m: nn.Module):
        """
        Initialize model weights.

        Uses Xavier/Kaiming initialization for better convergence.

        Args:
            m: Module to initialize
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward_encoder(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through CNN encoder.

        Args:
            x: Input spectrogram [B, C, F, T]

        Returns:
            Tuple of (features, skip_features)
            - features: Encoded features [B, C', F', T']
            - skip_features: List of intermediate features for skip connections
        """
        skip_features = []

        for block in self.encoder:
            x = block(x)
            skip_features.append(x)

        return x, skip_features

    def forward_transformer(
        self,
        x: torch.Tensor,
        spatial_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Forward pass through Vision Transformer.

        Args:
            x: Patch embeddings [B, N, embed_dim]
            spatial_shape: Original spatial dimensions (H, W)

        Returns:
            Transformed features [B, C, H, W]
        """
        B, N, D = x.shape

        # Add CLS token if used
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            N += 1

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer forward pass
        x = self.transformer(x)

        # Remove CLS token if used
        if self.cls_token is not None:
            x = x[:, 1:, :]  # Remove CLS token

        # Project back to feature space
        x = self.to_feature_map(x)  # [B, N, C]

        # Reshape to spatial feature map
        H, W = spatial_shape
        C = x.shape[-1]
        x = x.transpose(1, 2).reshape(B, C, H, W)

        return x

    def forward_decoder(
        self,
        x: torch.Tensor,
        skip_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass through CNN decoder.

        Args:
            x: Input features [B, C, H, W]
            skip_features: List of skip connection features

        Returns:
            Reconstructed spectrogram [B, C_out, F, T]
        """
        # Reverse skip features (decoder goes from deep to shallow)
        skip_features = skip_features[::-1]

        for i, block in enumerate(self.decoder):
            # Apply skip connection if enabled (except for last layer)
            if self.use_skip_connections and i < len(self.decoder) - 1:
                if i < len(skip_features):
                    skip = skip_features[i]

                    # Project skip feature to match decoder channels
                    skip = self.skip_projections[i](skip)

                    # Resize skip feature if needed
                    if skip.shape[2:] != x.shape[2:]:
                        skip = nn.functional.interpolate(
                            skip,
                            size=x.shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )

                    # Concatenate skip connection
                    x = torch.cat([x, skip], dim=1)

            # Apply decoder block
            x = block(x)

        return x

    def forward(
        self,
        x: torch.Tensor,
        return_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of Hybrid Vision Transformer.

        Args:
            x: Input noisy spectrogram [B, C, F, T]
            return_attentions: Whether to return attention maps

        Returns:
            Enhanced spectrogram [B, C_out, F, T]
            Optionally, attention maps from transformer
        """
        # Save input shape for final resizing
        input_shape = x.shape[2:]

        # ===== 1. CNN Encoder =====
        features, skip_features = self.forward_encoder(x)

        # ===== 2. Patch Embedding =====
        patches, spatial_shape = self.patch_embed(features)

        # ===== 3. Vision Transformer =====
        if return_attentions:
            # Get attention maps from transformer
            B, N, D = patches.shape

            # Add CLS token if used
            if self.cls_token is not None:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                patches_with_cls = torch.cat([cls_tokens, patches], dim=1)
            else:
                patches_with_cls = patches

            # Add positional encoding
            patches_with_pos = self.pos_encoding(patches_with_cls)

            # Transformer with attention
            transformed, attentions = self.transformer(
                patches_with_pos,
                return_all_attentions=True
            )

            # Remove CLS token if used
            if self.cls_token is not None:
                transformed = transformed[:, 1:, :]

            # Project back to feature map
            features = self.to_feature_map(transformed)
            H, W = spatial_shape
            C = features.shape[-1]
            features = features.transpose(1, 2).reshape(B, C, H, W)
        else:
            features = self.forward_transformer(patches, spatial_shape)
            attentions = None

        # ===== 4. CNN Decoder =====
        output = self.forward_decoder(features, skip_features)

        # ===== 5. Resize to input shape if needed =====
        if output.shape[2:] != input_shape:
            output = nn.functional.interpolate(
                output,
                size=input_shape,
                mode='bilinear',
                align_corners=False
            )

        if return_attentions:
            return output, attentions
        return output

    def count_parameters(self) -> Dict[str, int]:
        """
        Count parameters in each component.

        Returns:
            Dictionary with parameter counts
        """
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        return {
            'encoder': encoder_params,
            'transformer': transformer_params,
            'decoder': decoder_params,
            'total': total_params,
            'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


def create_hybrid_vit(config: Optional[Dict] = None) -> HybridViT:
    """
    Create Hybrid ViT model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        HybridViT model instance
    """
    if config is None:
        config = {}

    model_config = config.get('model', {})

    return HybridViT(
        input_channels=model_config.get('input_channels', 1),
        output_channels=model_config.get('output_channels', 1),
        encoder_channels=model_config.get('encoder', {}).get('channels', [64, 128, 256]),
        encoder_kernel_sizes=model_config.get('encoder', {}).get('kernel_sizes', [3, 3, 3]),
        encoder_pool_sizes=model_config.get('encoder', {}).get('pool_sizes', [2, 2, 1]),
        embed_dim=model_config.get('transformer', {}).get('embed_dim', 512),
        num_heads=model_config.get('transformer', {}).get('num_heads', 8),
        num_layers=model_config.get('transformer', {}).get('num_layers', 6),
        mlp_ratio=model_config.get('transformer', {}).get('mlp_ratio', 4),
        patch_size=model_config.get('transformer', {}).get('patch_size', 4),
        decoder_channels=model_config.get('decoder', {}).get('channels', [256, 128, 64, 1]),
        decoder_kernel_sizes=model_config.get('decoder', {}).get('kernel_sizes', [3, 3, 3, 3]),
        decoder_upsample_factors=model_config.get('decoder', {}).get('upsample_factors', [1, 2, 2, 1]),
        dropout=model_config.get('encoder', {}).get('dropout', 0.1),
        attn_dropout=model_config.get('transformer', {}).get('attention_dropout', 0.1),
        drop_path_rate=model_config.get('transformer', {}).get('drop_path_rate', 0.1),
        use_skip_connections=model_config.get('decoder', {}).get('use_skip_connections', True),
    )

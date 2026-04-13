"""Conv block + Transformer encoder (LCM backbone)."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Lightweight 1D conv block for local temporal feature extraction.
    Applied on the embedding dimension of each token.
    """

    def __init__(self, embed_dim: int, channels: List[int], kernel_size: int = 7):
        super().__init__()
        layers = []
        in_ch = embed_dim
        for out_ch in channels:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ])
            in_ch = out_ch
        # Project back to embed_dim
        layers.append(nn.Conv1d(in_ch, embed_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] — token sequence

        Returns:
            x: [B, N, D] — with local features extracted
        """
        # Conv1d expects [B, D, N]
        residual = x
        out = self.net(x.transpose(1, 2)).transpose(1, 2)
        return out + residual


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with multi-head self-attention and FFN."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm, need_weights=False)[0]
        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


class LCMEncoder(nn.Module):
    """Conv block followed by a stack of transformer blocks."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        conv_channels: Optional[List[int]] = None,
        conv_kernel_size: int = 7,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [256, 512]

        self.conv_block = ConvBlock(embed_dim, conv_channels, conv_kernel_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, D]

        Returns:
            features: [B, N, D]
        """
        x = self.conv_block(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

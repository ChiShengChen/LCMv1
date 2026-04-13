from __future__ import annotations
"""Learnable cross-montage channel mapping (Eq. 1)."""

import torch
import torch.nn as nn


class ChannelMapping(nn.Module):
    """Map EEG from any montage (M channels, zero-padded to max_channels)
    to a unified space (M' channels).

    x̃ = W_c @ x + bias

    Also provides a learnable channel embedding for each unified channel.
    """

    def __init__(self, max_channels: int, unified_channels: int, embed_dim: int):
        super().__init__()
        self.max_channels = max_channels
        self.unified_channels = unified_channels
        self.embed_dim = embed_dim

        # Linear mapping: [max_channels] -> [unified_channels]
        self.linear = nn.Linear(max_channels, unified_channels, bias=True)

        # Channel embedding: lookup table for each unified channel
        self.channel_embedding = nn.Embedding(unified_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, max_channels, T] — zero-padded raw EEG

        Returns:
            x_mapped: [B, M', T] — unified channel space
            channel_emb: [M', embed_dim] — channel embeddings
        """
        # x: [B, max_channels, T] -> transpose -> [B, T, max_channels]
        x_t = x.transpose(1, 2)
        # Apply linear mapping: [B, T, max_channels] -> [B, T, M']
        x_mapped = self.linear(x_t)
        # Transpose back: [B, T, M'] -> [B, M', T]
        x_mapped = x_mapped.transpose(1, 2)

        # Channel embeddings
        ch_indices = torch.arange(self.unified_channels, device=x.device)
        channel_emb = self.channel_embedding(ch_indices)  # [M', embed_dim]

        return x_mapped, channel_emb

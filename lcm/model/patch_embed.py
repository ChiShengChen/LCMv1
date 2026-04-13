"""Spatio-temporal patching and embedding (Eq. 2)."""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Segment unified EEG into spatio-temporal tokens.

    Input:  x̃ ∈ [B, M', T]
    Output: tokens ∈ [B, M' * num_patches, embed_dim]
    """

    def __init__(
        self,
        unified_channels: int,
        segment_length: int,
        patch_size_time: int,
        embed_dim: int,
    ):
        super().__init__()
        self.unified_channels = unified_channels
        self.patch_size_time = patch_size_time
        self.num_patches = segment_length // patch_size_time
        self.num_tokens = unified_channels * self.num_patches
        self.embed_dim = embed_dim

        # Linear projection per patch: [patch_size_time] -> [embed_dim]
        self.proj = nn.Linear(patch_size_time, embed_dim)

        # Temporal positional embedding (learnable, over patch index)
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )

    def forward(
        self, x_mapped: torch.Tensor, channel_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_mapped: [B, M', T] — unified channel space
            channel_emb: [M', embed_dim] — channel embeddings

        Returns:
            tokens: [B, M' * num_patches, embed_dim]
        """
        B, M, T = x_mapped.shape

        # Reshape to per-channel patches: [B, M', num_patches, patch_size]
        x = x_mapped.reshape(B, M, self.num_patches, self.patch_size_time)

        # Linear projection: [B, M', num_patches, patch_size] -> [B, M', num_patches, embed_dim]
        tokens = self.proj(x)

        # Add channel embedding: [M', embed_dim] -> [1, M', 1, embed_dim]
        tokens = tokens + channel_emb.unsqueeze(0).unsqueeze(2)

        # Add temporal positional embedding: [1, 1, num_patches, embed_dim]
        tokens = tokens + self.temporal_pos_embed.unsqueeze(1)

        # Flatten spatial and temporal dims: [B, M' * num_patches, embed_dim]
        tokens = tokens.reshape(B, self.num_tokens, self.embed_dim)

        return tokens

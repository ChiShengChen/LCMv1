from __future__ import annotations
"""Masked patch reconstruction head."""

import torch
import torch.nn as nn


class Reconstructor(nn.Module):
    """Decode encoder features back to original patch values.

    Maps each token from embed_dim back to patch_size_time.
    """

    def __init__(self, embed_dim: int, patch_size_time: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, patch_size_time),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, N, D] — encoder output features

        Returns:
            x_hat: [B, N, patch_size_time] — reconstructed patches
        """
        return self.head(z)

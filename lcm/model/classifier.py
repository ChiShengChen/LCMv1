from __future__ import annotations
"""Downstream classification head for fine-tuning."""

import torch
import torch.nn as nn

from ..config import ModelConfig, DataConfig
from .lcm import LCM


class LCMClassifier(nn.Module):
    """Wraps a pretrained LCM with a classification head.

    Prepends a [CLS] token and uses its output for classification.
    """

    def __init__(
        self,
        pretrained_lcm: LCM,
        num_classes: int,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        embed_dim = pretrained_lcm.config.embed_dim

        self.channel_mapping = pretrained_lcm.channel_mapping
        self.patch_embed = pretrained_lcm.patch_embed
        self.encoder = pretrained_lcm.online_encoder
        self.config = pretrained_lcm.config

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

        if freeze_encoder:
            for p in self.channel_mapping.parameters():
                p.requires_grad = False
            for p in self.patch_embed.parameters():
                p.requires_grad = False
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, max_channels, T] — zero-padded raw EEG

        Returns:
            logits: [B, num_classes]
        """
        # Channel mapping
        if self.config.use_channel_mapping:
            x_mapped, channel_emb = self.channel_mapping(x)
        else:
            B, _, T = x.shape
            x_mapped = x[:, :self.config.unified_channels, :]
            if x.size(1) < self.config.unified_channels:
                pad = torch.zeros(
                    B, self.config.unified_channels - x.size(1), T, device=x.device
                )
                x_mapped = torch.cat([x_mapped, pad], dim=1)
            channel_emb = torch.zeros(
                self.config.unified_channels, self.config.embed_dim, device=x.device
            )

        # Patch embedding
        tokens = self.patch_embed(x_mapped, channel_emb)  # [B, N, D]

        # Prepend CLS token
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # [B, N+1, D]

        # Encode
        features = self.encoder(tokens)  # [B, N+1, D]

        # Classify from CLS token
        cls_out = features[:, 0]  # [B, D]
        return self.classifier(cls_out)

from __future__ import annotations
"""Full LCM model: online encoder + momentum target encoder + losses."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig, DataConfig
from .channel_mapping import ChannelMapping
from .patch_embed import PatchEmbedding
from .encoder import LCMEncoder
from .reconstructor import Reconstructor


class LCM(nn.Module):
    """Large Cognition Model for self-supervised EEG pretraining.

    Uses momentum contrastive learning + masked reconstruction.
    """

    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__()
        self.config = model_config

        # Channel mapping
        self.channel_mapping = ChannelMapping(
            max_channels=model_config.max_channels,
            unified_channels=model_config.unified_channels,
            embed_dim=model_config.embed_dim,
        )

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            unified_channels=model_config.unified_channels,
            segment_length=data_config.segment_length,
            patch_size_time=model_config.patch_size_time,
            embed_dim=model_config.embed_dim,
        )

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(model_config.embed_dim) * 0.02)

        # Online encoder (updated by gradient descent)
        self.online_encoder = LCMEncoder(
            embed_dim=model_config.embed_dim,
            num_heads=model_config.num_heads,
            num_layers=model_config.num_layers,
            mlp_ratio=model_config.mlp_ratio,
            dropout=model_config.dropout,
            conv_channels=model_config.conv_channels,
            conv_kernel_size=model_config.conv_kernel_size,
        )

        # Target encoder (updated by EMA)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Reconstruction head
        self.reconstructor = Reconstructor(
            embed_dim=model_config.embed_dim,
            patch_size_time=model_config.patch_size_time,
        )

    def generate_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Generate random mask for tokens.

        Args:
            tokens: [B, N, D]

        Returns:
            mask: [B, N] boolean tensor (True = masked)
        """
        B, N, _ = tokens.shape
        num_mask = int(N * self.config.mask_ratio)
        mask = torch.zeros(B, N, dtype=torch.bool, device=tokens.device)
        for i in range(B):
            indices = torch.randperm(N, device=tokens.device)[:num_mask]
            mask[i, indices] = True
        return mask

    def get_original_patches(self, x_mapped: torch.Tensor) -> torch.Tensor:
        """Extract original patch values from mapped EEG.

        Args:
            x_mapped: [B, M', T]

        Returns:
            patches: [B, N, patch_size_time]
        """
        B, M, T = x_mapped.shape
        num_patches = T // self.config.patch_size_time
        # [B, M', num_patches, patch_size] -> [B, M'*num_patches, patch_size]
        patches = x_mapped.reshape(B, M, num_patches, self.config.patch_size_time)
        patches = patches.reshape(B, M * num_patches, self.config.patch_size_time)
        return patches

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        """
        Args:
            x: [B, max_channels, T] — zero-padded raw EEG

        Returns:
            loss: total loss
            l_a: alignment loss value
            l_r: reconstruction loss value
        """
        # 1. Channel mapping
        if self.config.use_channel_mapping:
            x_mapped, channel_emb = self.channel_mapping(x)
        else:
            # Bypass: zero-pad/truncate to unified_channels
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

        # 2. Patch embedding
        tokens = self.patch_embed(x_mapped, channel_emb)  # [B, N, D]

        # 3. Generate mask
        mask = self.generate_mask(tokens)  # [B, N]

        # 4. Online encoder: masked input
        masked_tokens = tokens.clone()
        masked_tokens[mask] = self.mask_token

        z = self.online_encoder(masked_tokens)  # [B, N, D]

        # 5. Compute losses
        l_a = torch.tensor(0.0, device=x.device)
        l_r = torch.tensor(0.0, device=x.device)

        # Contrastive alignment loss (Eq. 6)
        if self.config.use_contrastive_loss:
            with torch.no_grad():
                if self.config.use_momentum_encoder:
                    h = self.target_encoder(tokens)  # [B, N, D]
                else:
                    # SimSiam style: stop-gradient on online encoder copy
                    h = self.online_encoder(tokens)
                h = h.detach()

            l_a = F.mse_loss(
                F.layer_norm(z, [z.size(-1)]),
                F.layer_norm(h, [h.size(-1)]),
            )

        # Reconstruction loss (Eq. 9) — only on masked positions
        if self.config.use_reconstruction_loss:
            x_hat = self.reconstructor(z)  # [B, N, patch_size]
            original_patches = self.get_original_patches(x_mapped)
            l_r = F.mse_loss(x_hat[mask], original_patches[mask])

        # Total loss (Eq. 12)
        loss = l_a + self.config.reconstruction_weight * l_r

        return loss, l_a.item(), l_r.item()

    @torch.no_grad()
    def update_target_encoder(self, current_step: int, total_steps: int) -> None:
        """EMA update of target encoder parameters.

        Cosine schedule for momentum: 0.996 → 1.0
        """
        m = self.config.ema_momentum_end - (
            self.config.ema_momentum_end - self.config.ema_momentum_start
        ) * (math.cos(math.pi * current_step / total_steps) + 1) / 2

        for param_o, param_t in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_t.data = m * param_t.data + (1 - m) * param_o.data

    def save_checkpoint(self, path: str, epoch: int, optimizer=None, scheduler=None):
        state = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "model_config": self.config,
        }
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(state, path)

    @classmethod
    def load_from_checkpoint(
        cls, path: str, model_config: ModelConfig, data_config: DataConfig
    ) -> "LCM":
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(model_config, data_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

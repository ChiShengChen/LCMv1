from __future__ import annotations
"""LCM configuration dataclasses."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ModelConfig:
    # Channel mapping
    unified_channels: int = 22          # M' — target unified channel dim
    max_channels: int = 128             # max input channels (zero-pad smaller montages)
    channel_embed_dim: int = 512        # channel embedding dimension

    # Patching
    patch_size_time: int = 64           # temporal patch length (samples)

    # Transformer encoder
    embed_dim: int = 512                # transformer hidden dim
    num_heads: int = 8
    num_layers: int = 10                # calibrated to hit ~33.9M params
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    # Conv block (before transformer)
    conv_channels: List[int] = field(default_factory=lambda: [256, 512])
    conv_kernel_size: int = 7
    conv_stride: int = 1

    # Masking
    mask_ratio: float = 0.75

    # Momentum encoder
    ema_momentum_start: float = 0.996
    ema_momentum_end: float = 1.0

    # Loss
    reconstruction_weight: float = 1.0  # λ in L = L_A + λ * L_R

    # Ablation toggles
    use_contrastive_loss: bool = True
    use_reconstruction_loss: bool = True
    use_channel_mapping: bool = True
    use_momentum_encoder: bool = True


@dataclass
class PretrainConfig:
    epochs: int = 200
    batch_size: int = 1024
    lr: float = 1.5e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    betas: Tuple[float, float] = (0.9, 0.95)
    seed: int = 42
    num_workers: int = 4
    save_every: int = 10
    log_every: int = 50
    checkpoint_dir: str = "checkpoints/pretrain"
    use_wandb: bool = False
    wandb_project: str = "lcm-pretrain"


@dataclass
class FinetuneConfig:
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    betas: Tuple[float, float] = (0.9, 0.95)
    seed: int = 42
    num_workers: int = 4
    save_every: int = 10
    checkpoint_dir: str = "checkpoints/finetune"
    pretrained_path: str = ""
    freeze_encoder: bool = False        # True = linear probing, False = full fine-tuning
    use_wandb: bool = False
    wandb_project: str = "lcm-finetune"


@dataclass
class DataConfig:
    sample_rate: int = 256              # Hz, all data resampled to this
    segment_length_sec: float = 4.0     # seconds
    segment_length: int = 1024          # = sample_rate * segment_length_sec
    bandpass_low: float = 0.0           # Hz (for MI downstream)
    bandpass_high: float = 38.0         # Hz (for MI downstream)
    data_root: str = "data"
    datasets: List[str] = field(default_factory=lambda: [
        "physio_mi", "tsu_ssvep", "seed", "bcic2a", "bcic2b"
    ])


# Dataset-specific metadata
DATASET_INFO = {
    "physio_mi": {
        "paradigm": "MI+ME",
        "channels": 64,
        "subjects": 109,
        "num_classes": 5,
        "class_names": ["rest", "left_fist", "right_fist", "both_fists", "both_feet"],
    },
    "tsu_ssvep": {
        "paradigm": "SSVEP",
        "channels": 64,
        "subjects": 35,
        "num_classes": 40,
    },
    "seed": {
        "paradigm": "emotion",
        "channels": 62,
        "subjects": 15,
        "num_classes": 3,
        "class_names": ["negative", "neutral", "positive"],
    },
    "bcic2a": {
        "paradigm": "MI",
        "channels": 22,
        "subjects": 9,
        "num_classes": 4,
        "class_names": ["left_hand", "right_hand", "both_feet", "tongue"],
    },
    "bcic2b": {
        "paradigm": "MI",
        "channels": 3,
        "subjects": 9,
        "num_classes": 2,
        "class_names": ["left_hand", "right_hand"],
    },
}

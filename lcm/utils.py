from __future__ import annotations
"""Logging, EMA update, gradient monitoring, checkpointing utilities."""

import os
import random
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name: str, log_file: str | None = None, level=logging.INFO) -> logging.Logger:
    """Setup a logger with console and optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model: nn.Module) -> int:
    """Count all parameters (trainable + frozen)."""
    return sum(p.numel() for p in model.parameters())


def log_gradient_stats(model: nn.Module, logger: logging.Logger | None = None) -> dict:
    """Log gradient statistics for monitoring training (Eq. 13-14).

    Returns:
        stats: dict with gradient norm, mean, max per named parameter group
    """
    stats = {}
    total_norm = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            stats[name] = {
                "grad_norm": param_norm,
                "grad_mean": p.grad.data.mean().item(),
                "grad_max": p.grad.data.abs().max().item(),
            }

    total_norm = total_norm ** 0.5
    stats["total_grad_norm"] = total_norm

    if logger is not None:
        logger.debug(f"Total gradient norm: {total_norm:.4f}")

    return stats


class CosineWarmupScheduler:
    """Cosine annealing with linear warmup.

    Linear warmup from 0 → base_lr over warmup_steps,
    then cosine decay from base_lr → min_lr.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self) -> None:
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            return self.base_lr * self.current_step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress)
            )

    def state_dict(self) -> dict:
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict: dict) -> None:
        self.current_step = state_dict["current_step"]


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_metric: float | None = None,
) -> None:
    """Save training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_metric": best_metric,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
) -> dict:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint

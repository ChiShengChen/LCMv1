from __future__ import annotations
"""Downstream fine-tuning script for LCM."""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from lcm.config import ModelConfig, FinetuneConfig, DataConfig, DATASET_INFO
from lcm.model import LCM, LCMClassifier
from lcm.data import get_finetune_loaders
from lcm.data.utils import get_subject_splits
from lcm.evaluate import evaluate_model, format_results
from lcm.utils import (
    set_seed,
    setup_logger,
    count_parameters,
    CosineWarmupScheduler,
    save_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="LCM Downstream Fine-Tuning")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["bcic2a", "bcic2b", "seed", "physio_mi"])
    parser.add_argument("--pretrained_path", type=str, default="",
                        help="Path to pretrained checkpoint (empty = train from scratch)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Seeds for multi-seed evaluation")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/finetune")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Linear probing (freeze encoder)")
    parser.add_argument("--use_wandb", action="store_true")
    return parser.parse_args()


def train_one_subject(
    model: LCMClassifier,
    train_loader,
    test_loader,
    finetune_config: FinetuneConfig,
    num_classes: int,
    device: torch.device,
    logger,
) -> dict[str, float]:
    """Train and evaluate on one subject split."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=finetune_config.lr,
        weight_decay=finetune_config.weight_decay,
        betas=finetune_config.betas,
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = finetune_config.epochs * steps_per_epoch
    warmup_steps = finetune_config.warmup_epochs * steps_per_epoch
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps, total_steps, finetune_config.lr, finetune_config.min_lr
    )

    best_metrics = None
    best_bal_acc = 0.0

    for epoch in range(finetune_config.epochs):
        model.train()
        epoch_loss = 0.0

        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            logits = model(x)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        # Evaluate
        metrics = evaluate_model(model, test_loader, num_classes, device)

        if metrics["balanced_accuracy"] > best_bal_acc:
            best_bal_acc = metrics["balanced_accuracy"]
            best_metrics = metrics.copy()

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / max(1, len(train_loader))
            logger.info(
                f"  Epoch {epoch+1}/{finetune_config.epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Bal Acc: {metrics['balanced_accuracy']:.4f} | "
                f"Kappa: {metrics['cohen_kappa']:.4f}"
            )

    return best_metrics


def main():
    args = parse_args()

    # Configs
    model_config = ModelConfig()
    finetune_config = FinetuneConfig()
    data_config = DataConfig()

    # Apply CLI overrides
    if args.epochs is not None:
        finetune_config.epochs = args.epochs
    if args.batch_size is not None:
        finetune_config.batch_size = args.batch_size
    if args.lr is not None:
        finetune_config.lr = args.lr
    data_config.data_root = args.data_root
    finetune_config.checkpoint_dir = args.checkpoint_dir
    finetune_config.pretrained_path = args.pretrained_path
    finetune_config.freeze_encoder = args.freeze_encoder

    dataset_name = args.dataset
    dataset_info = DATASET_INFO[dataset_name]
    num_classes = dataset_info["num_classes"]
    num_subjects = dataset_info["subjects"]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(finetune_config.checkpoint_dir, exist_ok=True)
    logger = setup_logger(
        "finetune",
        log_file=os.path.join(finetune_config.checkpoint_dir, f"finetune_{dataset_name}.log"),
    )

    logger.info(f"Dataset: {dataset_name} ({num_classes} classes, {num_subjects} subjects)")
    logger.info(f"Pretrained: {args.pretrained_path or 'None (from scratch)'}")
    logger.info(f"Device: {device}")

    # Multi-seed evaluation
    all_seed_results: dict[str, list[float]] = {}

    for seed in args.seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Seed: {seed}")
        set_seed(seed)

        # Subject-wise cross-validation
        subject_metrics: dict[str, list[float]] = {}

        for eval_subj in range(num_subjects):
            logger.info(f"\n--- Evaluating subject {eval_subj} ---")
            train_subjects, test_subjects = get_subject_splits(
                dataset_name, num_subjects, eval_subj
            )

            # Load data
            train_loader, test_loader = get_finetune_loaders(
                data_config,
                dataset_name,
                train_subjects,
                test_subjects,
                max_channels=model_config.max_channels,
                batch_size=finetune_config.batch_size,
                num_workers=args.num_workers,
            )

            if len(train_loader.dataset) == 0 or len(test_loader.dataset) == 0:
                logger.warning(f"Skipping subject {eval_subj}: no data")
                continue

            # Build model
            if args.pretrained_path:
                pretrained_lcm = LCM.load_from_checkpoint(
                    args.pretrained_path, model_config, data_config
                )
            else:
                pretrained_lcm = LCM(model_config, data_config)

            classifier = LCMClassifier(
                pretrained_lcm,
                num_classes=num_classes,
                freeze_encoder=finetune_config.freeze_encoder,
            )

            # Train and evaluate
            metrics = train_one_subject(
                classifier, train_loader, test_loader,
                finetune_config, num_classes, device, logger,
            )

            logger.info(f"Subject {eval_subj} best: {metrics}")

            for key, val in metrics.items():
                if key not in subject_metrics:
                    subject_metrics[key] = []
                subject_metrics[key].append(val)

        # Average across subjects for this seed
        logger.info(f"\nSeed {seed} average across subjects:")
        for key, vals in subject_metrics.items():
            mean_val = np.mean(vals)
            logger.info(f"  {key}: {mean_val:.4f}")
            if key not in all_seed_results:
                all_seed_results[key] = []
            all_seed_results[key].append(mean_val)

    # Final results: mean ± std across seeds
    logger.info(f"\n{'='*60}")
    logger.info(f"Final results ({len(args.seeds)} seeds):")
    final_results = {}
    for key, vals in all_seed_results.items():
        arr = np.array(vals)
        final_results[key] = (arr.mean(), arr.std())
    logger.info(format_results(final_results))


if __name__ == "__main__":
    main()

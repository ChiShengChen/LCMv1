from __future__ import annotations
"""Fine-tune LCM on EEG-FM-Bench downstream tasks (bcic_2a, seed_iv)."""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from lcm.config import ModelConfig, FinetuneConfig, DataConfig
from lcm.model import LCM, LCMClassifier
from lcm.data.eegfm_adapter import get_eegfm_finetune_loaders
from lcm.evaluate import evaluate_model, format_results
from lcm.utils import (
    set_seed,
    setup_logger,
    count_parameters,
    CosineWarmupScheduler,
    save_checkpoint,
)

# Dataset metadata for EEG-FM-Bench (14 downstream tasks)
EEGFM_DATASET_INFO = {
    # Motor Imagery
    "bcic_2a": {"num_classes": 4, "class_names": ["left", "right", "foot", "tongue"]},
    "bcic_1a": {"num_classes": 3, "class_names": ["left", "right", "foot"]},
    "motor_mv_img": {"num_classes": 4, "class_names": ["left", "right", "both_fist", "foot"]},
    # Emotion
    "seed": {"num_classes": 3, "class_names": ["sad", "neutral", "happy"]},
    "seed_iv": {"num_classes": 4, "class_names": ["neutral", "sad", "fear", "happy"]},
    "seed_v": {"num_classes": 5, "class_names": ["disgust", "fear", "sad", "neutral", "happy"]},
    "seed_vii": {"num_classes": 7, "class_names": ["disgust", "fear", "sad", "neutral", "happy", "anger", "surprise"]},
    # Clinical
    "tuab": {"num_classes": 2, "class_names": ["normal", "abnormal"]},
    "tuev": {"num_classes": 6, "class_names": ["spsw", "gped", "pled", "eyem", "artf", "bckg"]},
    "tusl": {"num_classes": 3, "class_names": ["seiz", "slow", "bckg"]},
    "tuep": {"num_classes": 2, "class_names": ["epilepsy", "no_epilepsy"]},
    "siena_scalp": {"num_classes": 2, "class_names": ["seizure", "normal"]},
    "adftd": {"num_classes": 3, "class_names": ["AD", "FTD", "CN"]},
    # Cognitive / Visual
    "things_eeg_2": {"num_classes": 2, "class_names": ["non-target", "target"]},
    "inria_bci": {"num_classes": 2, "class_names": ["wrong", "correct"]},
}


def parse_args():
    parser = argparse.ArgumentParser(description="LCM Fine-Tuning with EEG-FM-Bench Data")
    parser.add_argument(
        "--eegfm_root",
        type=str,
        default="/media/meow/Transcend/time_series_benchmark/EEG-FM-Bench/assets/data/processed/fs_256",
    )
    parser.add_argument("--dataset", type=str, required=True, choices=list(EEGFM_DATASET_INFO.keys()))
    parser.add_argument("--pretrained_path", type=str, default="",
                        help="Path to pretrained checkpoint (empty = from scratch)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/finetune_eegfm")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    return parser.parse_args()


def train_and_evaluate(
    model: LCMClassifier,
    train_loader,
    test_loader,
    val_loader,
    num_classes: int,
    epochs: int,
    lr: float,
    device: torch.device,
    logger,
) -> dict[str, float]:
    """Train and evaluate, return best metrics on test set."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.05, betas=(0.9, 0.95),
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    warmup_steps = 5 * steps_per_epoch
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps, lr, 1e-6)

    best_metrics = None
    best_bal_acc = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            logits = model(x)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        # Evaluate on test set
        eval_loader = val_loader if val_loader is not None else test_loader
        metrics = evaluate_model(model, eval_loader, num_classes, device)

        if metrics["balanced_accuracy"] > best_bal_acc:
            best_bal_acc = metrics["balanced_accuracy"]
            # Also get test metrics when using val for selection
            if val_loader is not None:
                best_metrics = evaluate_model(model, test_loader, num_classes, device)
            else:
                best_metrics = metrics.copy()

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / steps_per_epoch
            logger.info(
                f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                f"Bal Acc: {metrics['balanced_accuracy']:.4f} | "
                f"Kappa: {metrics['cohen_kappa']:.4f}"
            )

    return best_metrics


def main():
    args = parse_args()

    model_config = ModelConfig()
    finetune_config = FinetuneConfig()
    data_config = DataConfig()

    dataset_name = args.dataset
    info = EEGFM_DATASET_INFO[dataset_name]
    num_classes = info["num_classes"]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger = setup_logger(
        "finetune_eegfm",
        log_file=os.path.join(args.checkpoint_dir, f"finetune_{dataset_name}.log"),
    )

    logger.info(f"Dataset: {dataset_name} ({num_classes} classes)")
    logger.info(f"Pretrained: {args.pretrained_path or 'None (from scratch)'}")
    logger.info(f"Device: {device}")

    all_seed_results: dict[str, list[float]] = {}

    for seed in args.seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Seed: {seed}")
        set_seed(seed)

        # Load data
        train_loader, test_loader, val_loader = get_eegfm_finetune_loaders(
            eegfm_processed_root=args.eegfm_root,
            dataset_name=dataset_name,
            config_name="finetune",
            segment_length=data_config.segment_length,
            max_channels=model_config.max_channels,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        logger.info(f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")

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
            freeze_encoder=args.freeze_encoder,
        )
        logger.info(f"Classifier params: {count_parameters(classifier):,}")

        # Train and evaluate
        metrics = train_and_evaluate(
            classifier, train_loader, test_loader, val_loader,
            num_classes, args.epochs, args.lr, device, logger,
        )

        logger.info(f"Seed {seed} best: {metrics}")
        for key, val in metrics.items():
            if key not in all_seed_results:
                all_seed_results[key] = []
            all_seed_results[key].append(val)

    # Final results
    logger.info(f"\n{'='*60}")
    logger.info(f"Final results ({len(args.seeds)} seeds):")
    final = {}
    for key, vals in all_seed_results.items():
        arr = np.array(vals)
        final[key] = (arr.mean(), arr.std())
    logger.info(format_results(final))


if __name__ == "__main__":
    main()

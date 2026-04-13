from __future__ import annotations
"""Evaluation metrics: balanced accuracy, Cohen's kappa, weighted F1, AUROC."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a classification model on a dataset.

    Args:
        model: LCMClassifier model
        dataloader: test DataLoader
        num_classes: number of classes
        device: torch device

    Returns:
        metrics: dict with balanced_accuracy, cohen_kappa, and
                 weighted_f1 (multi-class) or auroc (binary)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            x, labels = batch[0].to(device), batch[1]
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels.numpy())
            all_probs.append(probs)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    metrics = {}

    # Balanced accuracy
    metrics["balanced_accuracy"] = balanced_accuracy_score(all_labels, all_preds)

    # Cohen's kappa
    metrics["cohen_kappa"] = cohen_kappa_score(all_labels, all_preds)

    # Weighted F1 or AUROC depending on number of classes
    if num_classes == 2:
        # Binary: compute AUROC
        try:
            metrics["auroc"] = roc_auc_score(all_labels, all_probs[:, 1])
        except ValueError:
            metrics["auroc"] = 0.0
    else:
        # Multi-class: weighted F1
        metrics["weighted_f1"] = f1_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )

    return metrics


def run_multi_seed_evaluation(
    model_cls,
    pretrained_path: str,
    dataset_name: str,
    num_classes: int,
    seeds: list[int],
    train_fn,
    eval_fn,
) -> dict[str, tuple[float, float]]:
    """Run evaluation across multiple seeds and report mean ± std.

    Args:
        model_cls: model class constructor
        pretrained_path: path to pretrained checkpoint
        dataset_name: dataset to evaluate on
        num_classes: number of classes
        seeds: list of random seeds
        train_fn: function(seed) -> trained_model
        eval_fn: function(model) -> metrics_dict

    Returns:
        results: dict mapping metric_name -> (mean, std)
    """
    all_metrics: dict[str, list[float]] = {}

    for seed in seeds:
        model = train_fn(seed)
        metrics = eval_fn(model)

        for key, value in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = []
            all_metrics[key].append(value)

    results = {}
    for key, values in all_metrics.items():
        arr = np.array(values)
        results[key] = (arr.mean(), arr.std())

    return results


def format_results(results: dict[str, tuple[float, float]]) -> str:
    """Format multi-seed results as a readable string."""
    lines = []
    for metric, (mean, std) in results.items():
        lines.append(f"  {metric}: {mean:.4f} ± {std:.4f}")
    return "\n".join(lines)

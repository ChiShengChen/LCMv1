from __future__ import annotations
"""Channel name mappings, montage info, data splits."""

import torch
import numpy as np

# Standard 10-20 system channel names for common EEG datasets
CHANNEL_NAMES = {
    "bcic2a": [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
        "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz",
        "P2", "POz",
    ],
    "bcic2b": ["C3", "Cz", "C4"],
    "physio_mi": [
        "Fc5.", "Fc3.", "Fc1.", "Fcz.", "Fc2.", "Fc4.", "Fc6.", "C5..",
        "C3..", "C1..", "Cz..", "C2..", "C4..", "C6..", "Cp5.", "Cp3.",
        "Cp1.", "Cpz.", "Cp2.", "Cp4.", "Cp6.", "Fp1.", "Fpz.", "Fp2.",
        "Af7.", "Af3.", "Afz.", "Af4.", "Af8.", "F7..", "F5..", "F3..",
        "F1..", "Fz..", "F2..", "F4..", "F6..", "F8..", "Ft7.", "Ft8.",
        "T7..", "T8..", "T9..", "T10.", "Tp7.", "Tp8.", "P7..", "P5..",
        "P3..", "P1..", "Pz..", "P2..", "P4..", "P6..", "P8..", "Po7.",
        "Po3.", "Poz.", "Po4.", "Po8.", "O1..", "Oz..", "O2..", "Iz..",
    ],
    "seed": [
        "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ",
        "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
        "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
        "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
        "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
        "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1",
        "OZ", "O2", "CB2",
    ],
}


def get_dataset_channels(dataset_name: str) -> int:
    """Return the number of EEG channels for a given dataset."""
    from ..config import DATASET_INFO
    return DATASET_INFO[dataset_name]["channels"]


def get_subject_splits(
    dataset_name: str, num_subjects: int, eval_subject: int
) -> tuple[list[int], list[int]]:
    """Subject-dependent train/test split.

    Args:
        dataset_name: name of the dataset
        num_subjects: total number of subjects
        eval_subject: subject index to use as test set

    Returns:
        train_subjects, test_subjects
    """
    all_subjects = list(range(num_subjects))
    test_subjects = [eval_subject]
    train_subjects = [s for s in all_subjects if s != eval_subject]
    return train_subjects, test_subjects


def collate_fn(batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom collate function that zero-pads EEG to max_channels in batch.

    Each sample is (eeg_segment, channel_count, label_or_dataset_id).
    """
    segments, channel_counts, labels = zip(*batch)

    max_ch = max(s.shape[0] for s in segments)
    T = segments[0].shape[1]
    B = len(segments)

    padded = torch.zeros(B, max_ch, T)
    for i, seg in enumerate(segments):
        padded[i, :seg.shape[0], :] = seg

    channel_counts = torch.tensor(channel_counts, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return padded, channel_counts, labels


def pad_to_max_channels(
    x: np.ndarray | torch.Tensor, max_channels: int
) -> np.ndarray | torch.Tensor:
    """Zero-pad EEG channels to max_channels.

    Args:
        x: [C, T] — EEG segment
        max_channels: target number of channels

    Returns:
        x_padded: [max_channels, T]
    """
    C, T = x.shape
    if C >= max_channels:
        return x[:max_channels]

    if isinstance(x, torch.Tensor):
        pad = torch.zeros(max_channels - C, T, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=0)
    else:
        pad = np.zeros((max_channels - C, T), dtype=x.dtype)
        return np.concatenate([x, pad], axis=0)

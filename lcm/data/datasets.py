from __future__ import annotations
"""Dataset classes for EEG pretraining and fine-tuning."""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from ..config import DataConfig, DATASET_INFO
from .preprocessing import preprocess_raw
from .utils import pad_to_max_channels, collate_fn


class EEGPretrainDataset(Dataset):
    """Dataset for self-supervised pretraining.

    Loads preprocessed EEG segments from .npy files.
    Each sample returns (eeg_segment, channel_count, dataset_id).
    """

    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        max_channels: int = 128,
        segment_length: int = 1024,
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.max_channels = max_channels
        self.segment_length = segment_length
        self.transform = transform
        self.num_channels = DATASET_INFO[dataset_name]["channels"]

        # Load preprocessed segments
        self.segments = self._load_segments()

    def _load_segments(self) -> list[np.ndarray]:
        """Load all preprocessed segments for this dataset."""
        seg_dir = self.data_dir / self.dataset_name / "pretrain"
        segments = []

        if seg_dir.exists():
            for f in sorted(seg_dir.glob("*.npy")):
                seg = np.load(f)
                if seg.shape[-1] == self.segment_length:
                    segments.append(seg)

        if not segments:
            # Placeholder: return empty dataset (user needs to preprocess data first)
            print(
                f"Warning: No preprocessed segments found at {seg_dir}. "
                f"Run preprocessing first."
            )
        return segments

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        seg = self.segments[idx].astype(np.float32)

        if self.transform is not None:
            seg = self.transform(seg)

        # Zero-pad to max_channels
        seg = pad_to_max_channels(seg, self.max_channels)
        seg = torch.from_numpy(seg)

        # dataset_id for identifying which dataset this came from
        dataset_id = list(DATASET_INFO.keys()).index(self.dataset_name)

        return seg, self.num_channels, dataset_id


class EEGFinetuneDataset(Dataset):
    """Dataset for supervised fine-tuning on downstream tasks.

    Each sample returns (eeg_segment, label).
    """

    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        subjects: list[int],
        max_channels: int = 128,
        segment_length: int = 1024,
        split: str = "train",
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.max_channels = max_channels
        self.segment_length = segment_length
        self.transform = transform
        self.num_channels = DATASET_INFO[dataset_name]["channels"]

        self.segments, self.labels = self._load_subject_data(subjects, split)

    def _load_subject_data(
        self, subjects: list[int], split: str
    ) -> tuple[list[np.ndarray], list[int]]:
        """Load data for specified subjects."""
        all_segments = []
        all_labels = []

        for subj in subjects:
            seg_file = (
                self.data_dir
                / self.dataset_name
                / split
                / f"subject_{subj:03d}_segments.npy"
            )
            label_file = (
                self.data_dir
                / self.dataset_name
                / split
                / f"subject_{subj:03d}_labels.npy"
            )

            if seg_file.exists() and label_file.exists():
                segs = np.load(seg_file)    # [N_seg, C, T]
                labels = np.load(label_file)  # [N_seg]
                for i in range(len(segs)):
                    all_segments.append(segs[i])
                    all_labels.append(int(labels[i]))
            else:
                print(
                    f"Warning: Data not found for subject {subj} at {seg_file}. "
                    f"Run preprocessing first."
                )

        return all_segments, all_labels

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        seg = self.segments[idx].astype(np.float32)

        if self.transform is not None:
            seg = self.transform(seg)

        # Zero-pad to max_channels
        seg = pad_to_max_channels(seg, self.max_channels)
        seg = torch.from_numpy(seg)

        return seg, self.labels[idx]


def get_pretrain_loader(
    data_config: DataConfig, max_channels: int, batch_size: int, num_workers: int = 4
) -> DataLoader:
    """Create a DataLoader combining all pretraining datasets."""
    datasets = []
    for ds_name in data_config.datasets:
        ds = EEGPretrainDataset(
            data_dir=data_config.data_root,
            dataset_name=ds_name,
            max_channels=max_channels,
            segment_length=data_config.segment_length,
        )
        if len(ds) > 0:
            datasets.append(ds)

    if not datasets:
        raise ValueError(
            f"No data found in {data_config.data_root}. "
            f"Please preprocess the datasets first."
        )

    combined = ConcatDataset(datasets)
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )


def get_finetune_loaders(
    data_config: DataConfig,
    dataset_name: str,
    train_subjects: list[int],
    test_subjects: list[int],
    max_channels: int,
    batch_size: int,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Create train and test DataLoaders for fine-tuning."""
    train_ds = EEGFinetuneDataset(
        data_dir=data_config.data_root,
        dataset_name=dataset_name,
        subjects=train_subjects,
        max_channels=max_channels,
        split="train",
    )
    test_ds = EEGFinetuneDataset(
        data_dir=data_config.data_root,
        dataset_name=dataset_name,
        subjects=test_subjects,
        max_channels=max_channels,
        split="test",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader

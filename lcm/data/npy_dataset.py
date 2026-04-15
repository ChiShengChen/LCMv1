from __future__ import annotations
"""Fast numpy-based dataset for LCM training (reads from SSD cache)."""

import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class NpyEEGDataset(Dataset):
    """Load preprocessed EEG segments from numpy files.

    Uses memory-mapped numpy for fast random access from SSD.
    Zero-pads channels to max_channels on-the-fly.
    """

    def __init__(
        self,
        cache_dir: str,
        dataset_name: str,
        split: str = "train",
        max_channels: int = 128,
    ):
        self.max_channels = max_channels

        data_path = os.path.join(cache_dir, f"{dataset_name}_{split}_data.npy")
        meta_path = os.path.join(cache_dir, f"{dataset_name}_{split}_meta.npz")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Cache not found: {data_path}")

        # Memory-mapped for fast random access without loading all into RAM
        self.data = np.load(data_path, mmap_mode="r")  # [N, C, T]
        meta = np.load(meta_path)
        self.n_channels = int(meta["n_channels"])
        self.labels = meta["labels"]  # [N]

        print(
            f"[NpyEEGDataset] {dataset_name}/{split}: "
            f"{len(self.data)} segments, {self.n_channels}ch, "
            f"shape={self.data.shape}"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        seg = self.data[idx].astype(np.float32)  # [C, T]
        C, T = seg.shape

        # Zero-pad to max_channels
        if C < self.max_channels:
            pad = np.zeros((self.max_channels - C, T), dtype=np.float32)
            seg = np.concatenate([seg, pad], axis=0)
        else:
            seg = seg[: self.max_channels]

        return torch.from_numpy(seg.copy()), self.n_channels, int(self.labels[idx])


class NpyFinetuneDataset(Dataset):
    """Numpy-based dataset for fine-tuning (with labels)."""

    def __init__(self, cache_dir: str, dataset_name: str, split: str, max_channels: int = 128):
        self.inner = NpyEEGDataset(cache_dir, dataset_name, split, max_channels)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        seg, _, label = self.inner[idx]
        return seg, label


def pretrain_collate_fn(batch):
    segs, chs, labels = zip(*batch)
    return torch.stack(segs), torch.tensor(chs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def get_npy_pretrain_loader(
    cache_dir: str,
    dataset_splits: list[tuple[str, str]],
    max_channels: int = 128,
    batch_size: int = 64,
    num_workers: int = 4,
) -> DataLoader:
    """Create pretrain DataLoader from numpy cache.

    Args:
        cache_dir: path to numpy cache directory
        dataset_splits: list of (dataset_name, split) tuples
        max_channels: zero-pad to this
        batch_size: batch size
        num_workers: DataLoader workers
    """
    datasets = []
    for ds_name, split in dataset_splits:
        try:
            ds = NpyEEGDataset(cache_dir, ds_name, split, max_channels)
            if len(ds) > 0:
                datasets.append(ds)
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    if not datasets:
        raise ValueError(f"No data found in {cache_dir}")

    combined = ConcatDataset(datasets)
    print(f"[Pretrain Loader] Total: {len(combined)} segments")

    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=pretrain_collate_fn,
        pin_memory=True,
        drop_last=True,
    )


def get_npy_finetune_loaders(
    cache_dir: str,
    dataset_name: str,
    max_channels: int = 128,
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Create train/test/val DataLoaders from numpy cache."""
    train_ds = NpyFinetuneDataset(cache_dir, dataset_name, "train", max_channels)
    test_ds = NpyFinetuneDataset(cache_dir, dataset_name, "test", max_channels)

    val_ds = None
    try:
        val_ds = NpyFinetuneDataset(cache_dir, dataset_name, "validation", max_channels)
    except FileNotFoundError:
        pass

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = None
    if val_ds and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
    return train_loader, test_loader, val_loader

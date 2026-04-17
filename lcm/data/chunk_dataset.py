from __future__ import annotations
"""Chunk-based dataset for tuab (multiple .npy files)."""

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader


class ChunkEEGDataset(Dataset):
    """Load EEG from a single .npy chunk file with mmap."""

    def __init__(self, npy_path: str, max_channels: int = 128):
        self.max_channels = max_channels
        self.data = np.load(npy_path)  # [N, C, T] loaded into RAM
        self.n_channels = self.data.shape[1]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        seg = self.data[idx].astype(np.float32)
        C, T = seg.shape
        if C < self.max_channels:
            pad = np.zeros((self.max_channels - C, T), dtype=np.float32)
            seg = np.concatenate([seg, pad], axis=0)
        else:
            seg = seg[:self.max_channels]
        return torch.from_numpy(seg.copy()), self.n_channels, -1


def get_chunk_pretrain_loader(
    chunk_dirs: list[str],
    other_npy_datasets: list[tuple[str, str, str]] = None,
    max_channels: int = 128,
    batch_size: int = 64,
    num_workers: int = 4,
) -> DataLoader:
    """Create DataLoader from chunk dirs + regular npy datasets.

    Args:
        chunk_dirs: list of directories containing chunk_*.npy files
        other_npy_datasets: list of (cache_dir, dataset_name, split) for NpyEEGDataset
        max_channels: zero-pad channels
        batch_size: batch size
        num_workers: workers
    """
    from .npy_dataset import NpyEEGDataset, pretrain_collate_fn

    datasets = []

    # Load chunk dirs
    for chunk_dir in chunk_dirs:
        files = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.npy")))
        for f in files:
            ds = ChunkEEGDataset(f, max_channels)
            if len(ds) > 0:
                datasets.append(ds)
        if files:
            print(f"[Chunks] {chunk_dir}: {len(files)} chunks loaded")

    # Load regular npy datasets
    if other_npy_datasets:
        for cache_dir, ds_name, split in other_npy_datasets:
            try:
                ds = NpyEEGDataset(cache_dir, ds_name, split, max_channels)
                if len(ds) > 0:
                    datasets.append(ds)
            except FileNotFoundError as e:
                print(f"Warning: {e}")

    if not datasets:
        raise ValueError("No data loaded")

    combined = ConcatDataset(datasets)
    print(f"[Pretrain Loader] Total: {len(combined)} segments from {len(datasets)} sources")

    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=pretrain_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

from __future__ import annotations
"""Adapter to load EEG-FM-Bench processed Arrow data for LCM pretraining/finetuning."""

import glob
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


def _find_arrow_files(arrow_dir: str, dataset_name: str, split: str) -> list[str]:
    """Find Arrow files, auto-detecting version subdirectories."""
    pattern = f"{arrow_dir}/{dataset_name}-{split}-*.arrow"
    files = sorted(glob.glob(pattern))
    if not files:
        for sub in sorted(glob.glob(f"{arrow_dir}/*")):
            if os.path.isdir(sub):
                alt_pattern = f"{sub}/{dataset_name}-{split}-*.arrow"
                files = sorted(glob.glob(alt_pattern))
                if files:
                    break
    return files


def _find_arrow_dir(eegfm_processed_root: str, name: str, config: str) -> str | None:
    """Auto-detect version directory for a dataset."""
    base = f"{eegfm_processed_root}/{name}/{config}"
    if os.path.isdir(base):
        for sub in sorted(os.listdir(base)):
            sub_path = os.path.join(base, sub)
            if os.path.isdir(sub_path) and glob.glob(f"{sub_path}/*.arrow"):
                return sub_path
    return None


class EEGFMDataset(Dataset):
    """Lazy-loading EEG dataset from EEG-FM-Bench Arrow files.

    Only builds an index on init. Data is read on-the-fly per __getitem__.
    """

    def __init__(
        self,
        arrow_dir: str,
        dataset_name: str,
        split: str = "train",
        segment_length: int = 1024,
        max_channels: int = 128,
    ):
        self.segment_length = segment_length
        self.max_channels = max_channels
        self.dataset_name = dataset_name

        files = _find_arrow_files(arrow_dir, dataset_name, split)
        if not files:
            raise FileNotFoundError(
                f"No Arrow files for {dataset_name}/{split} in {arrow_dir}"
            )

        # Build index: (file_idx, row_idx, seg_offset, n_channels, label)
        self.index = []
        self.files = files
        self._tables = {}  # lazy cache

        for file_idx, f in enumerate(files):
            reader = pa.ipc.open_stream(f)
            table = reader.read_all()
            n_rows = table.num_rows

            has_label = "label" in table.column_names
            labels = table.column("label").to_pylist() if has_label else [-1] * n_rows

            # Peek at first row to get shape info
            row0 = table.column("data")[0].as_py()
            n_ch = len(row0)
            wnd_len = len(row0[0])
            num_segs = wnd_len // segment_length

            for row_idx in range(n_rows):
                for seg_off in range(num_segs):
                    self.index.append((file_idx, row_idx, seg_off, n_ch, labels[row_idx]))

            del table  # free memory

        print(
            f"[EEGFMDataset] {dataset_name}/{split}: "
            f"{len(self.index)} segments from {len(files)} files"
        )

    def _get_table(self, file_idx: int) -> pa.Table:
        """Lazy-load and cache Arrow table."""
        if file_idx not in self._tables:
            reader = pa.ipc.open_stream(self.files[file_idx])
            self._tables[file_idx] = reader.read_all()
            # Keep cache bounded: evict old tables if too many
            if len(self._tables) > 3:
                oldest = min(self._tables.keys())
                if oldest != file_idx:
                    del self._tables[oldest]
        return self._tables[file_idx]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        file_idx, row_idx, seg_off, n_ch, label = self.index[idx]

        table = self._get_table(file_idx)
        row_data = table.column("data")[row_idx].as_py()
        arr = np.array(row_data, dtype=np.float32)  # [C, T]

        # Slice segment
        start = seg_off * self.segment_length
        seg = arr[:, start : start + self.segment_length]

        # Zero-pad to max_channels
        C, T = seg.shape
        if C < self.max_channels:
            pad = np.zeros((self.max_channels - C, T), dtype=np.float32)
            seg = np.concatenate([seg, pad], axis=0)
        else:
            seg = seg[: self.max_channels]

        return torch.from_numpy(seg), n_ch, label


class EEGFMFinetuneDataset(Dataset):
    """EEG-FM-Bench data for LCM fine-tuning (with labels)."""

    def __init__(
        self,
        arrow_dir: str,
        dataset_name: str,
        split: str = "train",
        segment_length: int = 1024,
        max_channels: int = 128,
    ):
        self.inner = EEGFMDataset(
            arrow_dir, dataset_name, split, segment_length, max_channels
        )

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        seg, _, label = self.inner[idx]
        return seg, label


def pretrain_collate_fn(
    batch: list[tuple[torch.Tensor, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate for pretraining: (eeg, channel_count, dataset_id)."""
    segs, chs, labels = zip(*batch)
    return (
        torch.stack(segs),
        torch.tensor(chs, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def get_eegfm_pretrain_loader(
    eegfm_processed_root: str,
    dataset_configs: list[dict],
    segment_length: int = 1024,
    max_channels: int = 128,
    batch_size: int = 64,
    num_workers: int = 4,
) -> DataLoader:
    """Create pretrain DataLoader from multiple EEG-FM-Bench datasets."""
    datasets = []
    for dc in dataset_configs:
        arrow_dir = _find_arrow_dir(eegfm_processed_root, dc["name"], dc["config"])
        if arrow_dir is None:
            arrow_dir = f"{eegfm_processed_root}/{dc['name']}/{dc['config']}/1.0.0"
        for split in dc.get("splits", ["train"]):
            try:
                ds = EEGFMDataset(
                    arrow_dir=arrow_dir,
                    dataset_name=dc["name"],
                    split=split,
                    segment_length=segment_length,
                    max_channels=max_channels,
                )
                if len(ds) > 0:
                    datasets.append(ds)
            except FileNotFoundError as e:
                print(f"Warning: {e}")

    if not datasets:
        raise ValueError("No data loaded. Check paths and dataset_configs.")

    combined = ConcatDataset(datasets)
    print(f"[Pretrain Loader] Total: {len(combined)} segments from {len(datasets)} dataset-splits")

    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=pretrain_collate_fn,
        pin_memory=True,
        drop_last=True,
    )


def get_eegfm_finetune_loaders(
    eegfm_processed_root: str,
    dataset_name: str,
    config_name: str = "finetune",
    segment_length: int = 1024,
    max_channels: int = 128,
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Create train/test/validation DataLoaders for fine-tuning."""
    arrow_dir = _find_arrow_dir(eegfm_processed_root, dataset_name, config_name)
    if arrow_dir is None:
        arrow_dir = f"{eegfm_processed_root}/{dataset_name}/{config_name}/1.0.0"

    train_ds = EEGFMFinetuneDataset(arrow_dir, dataset_name, "train", segment_length, max_channels)
    test_ds = EEGFMFinetuneDataset(arrow_dir, dataset_name, "test", segment_length, max_channels)

    val_ds = None
    try:
        val_ds = EEGFMFinetuneDataset(arrow_dir, dataset_name, "validation", segment_length, max_channels)
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

from __future__ import annotations
"""Adapter to load EEG-FM-Bench processed Arrow data for LCM pretraining/finetuning."""

import glob
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class EEGFMDataset(Dataset):
    """Load EEG data from EEG-FM-Bench Arrow files.

    Handles variable-length windows by slicing into fixed-length segments.
    Zero-pads channels to max_channels.
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

        # Load all Arrow files for this split
        # Auto-detect version directory (1.0.0, 3.0.1, etc.)
        pattern = f"{arrow_dir}/{dataset_name}-{split}-*.arrow"
        files = sorted(glob.glob(pattern))
        if not files:
            # Try searching in versioned subdirs or incomplete dirs
            for sub in sorted(glob.glob(f"{arrow_dir}/*")):
                if os.path.isdir(sub):
                    alt_pattern = f"{sub}/{dataset_name}-{split}-*.arrow"
                    files = sorted(glob.glob(alt_pattern))
                    if files:
                        break
        if not files:
            raise FileNotFoundError(
                f"No Arrow files found: {pattern} "
                f"(also checked subdirs of {arrow_dir})"
            )

        self.segments = []
        self.labels = []
        self.n_channels_list = []

        for f in files:
            reader = pa.ipc.open_stream(f)
            table = reader.read_all()
            n_rows = table.num_rows

            # Batch extract labels
            has_label = "label" in table.column_names
            labels_col = table.column("label").to_pylist() if has_label else [-1] * n_rows

            # Process rows
            data_col = table.column("data")
            for i in range(n_rows):
                row_data = data_col[i].as_py()  # list[list[float]]
                n_ch = len(row_data)
                wnd_len = len(row_data[0])
                arr = np.array(row_data, dtype=np.float32)  # [C, T]

                label = labels_col[i]

                # Slice into segment_length chunks
                for start in range(0, wnd_len - segment_length + 1, segment_length):
                    seg = arr[:, start : start + segment_length]
                    self.segments.append(seg)
                    self.labels.append(label)
                    self.n_channels_list.append(n_ch)

        print(
            f"[EEGFMDataset] {dataset_name}/{split}: "
            f"{len(self.segments)} segments from {len(files)} files, "
            f"channels={self.n_channels_list[0] if self.segments else 0}, "
            f"seg_len={segment_length}"
        )

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        """Returns (eeg_padded, n_channels, label)."""
        seg = self.segments[idx]  # [C, T]
        n_ch = self.n_channels_list[idx]

        # Zero-pad to max_channels
        C, T = seg.shape
        if C < self.max_channels:
            pad = np.zeros((self.max_channels - C, T), dtype=np.float32)
            seg = np.concatenate([seg, pad], axis=0)
        else:
            seg = seg[: self.max_channels]

        return torch.from_numpy(seg), n_ch, self.labels[idx]


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
        """Returns (eeg_padded, label)."""
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
    """Create pretrain DataLoader from multiple EEG-FM-Bench datasets.

    Args:
        eegfm_processed_root: root path to EEG-FM-Bench processed data
            e.g., '.../EEG-FM-Bench/assets/data/processed/fs_256'
        dataset_configs: list of dicts with keys:
            - name: dataset name (e.g., 'spis_resting_state')
            - config: config name (e.g., 'pretrain' or 'finetune')
            - splits: list of splits to use (e.g., ['train'] or ['train', 'validation'])
        segment_length: fixed segment length in samples
        max_channels: zero-pad channels to this number
        batch_size: batch size
        num_workers: DataLoader workers

    Returns:
        DataLoader
    """
    datasets = []
    for dc in dataset_configs:
        # Auto-detect version directory
        base = f"{eegfm_processed_root}/{dc['name']}/{dc['config']}"
        arrow_dir = None
        if os.path.isdir(base):
            for sub in sorted(os.listdir(base)):
                sub_path = os.path.join(base, sub)
                if os.path.isdir(sub_path) and glob.glob(f"{sub_path}/*.arrow"):
                    arrow_dir = sub_path
                    break
        if arrow_dir is None:
            arrow_dir = f"{base}/1.0.0"  # fallback
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
    """Create train/test/validation DataLoaders for fine-tuning.

    Returns:
        (train_loader, test_loader, val_loader or None)
    """
    # Auto-detect version directory
    base = f"{eegfm_processed_root}/{dataset_name}/{config_name}"
    arrow_dir = None
    if os.path.isdir(base):
        for sub in sorted(os.listdir(base)):
            sub_path = os.path.join(base, sub)
            if os.path.isdir(sub_path) and glob.glob(f"{sub_path}/*.arrow"):
                arrow_dir = sub_path
                break
    if arrow_dir is None:
        arrow_dir = f"{base}/1.0.0"  # fallback

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

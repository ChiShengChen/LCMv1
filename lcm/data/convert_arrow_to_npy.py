from __future__ import annotations
"""Convert EEG-FM-Bench Arrow files to numpy cache for fast training."""

import argparse
import glob
import os

import numpy as np
import pyarrow as pa


def convert_dataset(
    arrow_dir: str,
    dataset_name: str,
    split: str,
    output_dir: str,
    segment_length: int = 1024,
):
    """Convert Arrow files to numpy arrays cached on SSD.

    Creates:
        {output_dir}/{dataset_name}_{split}_data.npy   - [N, C, seg_len] float32
        {output_dir}/{dataset_name}_{split}_meta.npz    - n_channels, labels
    """
    # Find arrow files
    pattern = f"{arrow_dir}/{dataset_name}-{split}-*.arrow"
    files = sorted(glob.glob(pattern))
    if not files:
        for sub in sorted(glob.glob(f"{arrow_dir}/*")):
            if os.path.isdir(sub):
                alt = f"{sub}/{dataset_name}-{split}-*.arrow"
                files = sorted(glob.glob(alt))
                if files:
                    break
    if not files:
        print(f"  No files for {dataset_name}/{split}, skipping")
        return 0

    # First pass: count segments and get channel count
    total_segs = 0
    n_ch = None
    for f in files:
        reader = pa.ipc.open_stream(f)
        table = reader.read_all()
        row0 = table.column("data")[0].as_py()
        if n_ch is None:
            n_ch = len(row0)
        wnd_len = len(row0[0])
        num_segs_per_row = wnd_len // segment_length
        total_segs += table.num_rows * num_segs_per_row
        del table

    print(f"  {dataset_name}/{split}: {total_segs} segments, {n_ch} channels")

    # Second pass: fill memory-mapped numpy array (no RAM blow-up)
    os.makedirs(output_dir, exist_ok=True)
    data_path = os.path.join(output_dir, f"{dataset_name}_{split}_data.npy")
    meta_path = os.path.join(output_dir, f"{dataset_name}_{split}_meta.npz")

    # Create memmap file
    data = np.lib.format.open_memmap(
        data_path, mode="w+", dtype=np.float32,
        shape=(total_segs, n_ch, segment_length),
    )
    labels = np.zeros(total_segs, dtype=np.int32)

    idx = 0
    for fi, f in enumerate(files):
        reader = pa.ipc.open_stream(f)
        table = reader.read_all()
        has_label = "label" in table.column_names
        labels_col = table.column("label").to_pylist() if has_label else [-1] * table.num_rows

        for row_i in range(table.num_rows):
            row_data = table.column("data")[row_i].as_py()
            arr = np.array(row_data, dtype=np.float32)
            wnd_len = arr.shape[1]
            num_segs = wnd_len // segment_length

            for s in range(num_segs):
                start = s * segment_length
                data[idx] = arr[:, start:start + segment_length]
                labels[idx] = labels_col[row_i]
                idx += 1

        del table
        if (fi + 1) % 10 == 0:
            print(f"    Processed {fi+1}/{len(files)} files ({idx}/{total_segs} segments)", flush=True)

    # Flush memmap
    del data
    np.savez(meta_path, n_channels=n_ch, labels=labels)
    size_gb = os.path.getsize(data_path) / 1e9
    print(f"  Saved: {data_path} ({size_gb:.1f} GB)", flush=True)
    return total_segs


def main():
    parser = argparse.ArgumentParser(description="Convert Arrow to numpy cache")
    parser.add_argument(
        "--eegfm_root", type=str,
        default="/media/meow/Transcend/time_series_benchmark/EEG-FM-Bench/assets/data/processed/fs_256",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/media/meow/Transcend/time_series_benchmark/LCMv1/data_cache",
    )
    parser.add_argument("--segment_length", type=int, default=1024)
    args = parser.parse_args()

    datasets = [
        ("spis_resting_state", "pretrain", ["train", "validation"]),
        ("bcic_2a", "finetune", ["train", "test", "validation"]),
        ("seed_iv", "finetune", ["train", "test", "validation"]),
        ("tuab", "pretrain", ["train"]),
        ("tuab", "finetune", ["train", "test", "validation"]),
    ]

    total = 0
    for ds_name, config, splits in datasets:
        base = f"{args.eegfm_root}/{ds_name}/{config}"
        # Auto-detect version dir
        arrow_dir = None
        if os.path.isdir(base):
            for sub in sorted(os.listdir(base)):
                sub_path = os.path.join(base, sub)
                if os.path.isdir(sub_path) and glob.glob(f"{sub_path}/*.arrow"):
                    arrow_dir = sub_path
                    break
        if arrow_dir is None:
            print(f"Skipping {ds_name}/{config}: no arrow files")
            continue

        print(f"\n=== {ds_name}/{config} (from {arrow_dir}) ===")
        for split in splits:
            n = convert_dataset(arrow_dir, ds_name, split, args.output_dir, args.segment_length)
            total += n

    print(f"\n=== Total: {total:,} segments converted ===")


if __name__ == "__main__":
    main()

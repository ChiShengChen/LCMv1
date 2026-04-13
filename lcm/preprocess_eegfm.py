from __future__ import annotations
"""Preprocess raw EEG datasets from Elements drive using EEG-FM-Bench pipeline.

This script creates symlinks from the Elements drive to the EEG-FM-Bench
expected raw data layout, then runs the EEG-FM-Bench preprocessing pipeline.

Usage:
    python -m lcm.preprocess_eegfm --elements_root /media/meow/Elements \
        --eegfm_root /media/meow/Transcend/time_series_benchmark/EEG-FM-Bench \
        --datasets tuab spis_resting_state motor_mv_img
"""

import argparse
import os
import sys
from pathlib import Path

# Mapping: EEG-FM-Bench dataset ID -> (Elements folder, suffix_path expected by builder)
ELEMENTS_TO_EEGFM = {
    # === PRETRAIN DATASETS ===
    # TU Hospital (Clinical, large-scale)
    "tueg": ("TU_EEG", os.path.join("TUE", "tueg")),
    "tuab": ("TUAB", os.path.join("TUE", "tuab")),
    "tuar": ("TUAR", os.path.join("TUE", "tuar")),
    "tusz": ("TUSZ", os.path.join("TUE", "tusz")),
    "tuep": ("TUEP", os.path.join("TUE", "tuep")),
    "tuev": ("TUEV", os.path.join("TUE", "tuev")),
    "tusl": ("TUSL", os.path.join("TUE", "tusl")),
    # Motor
    "motor_mv_img": (
        "EEGMotorMovement_ImageryDataset",
        os.path.join("Motor Movement Imagery", "eeg-motor-movementimagery-dataset-1.0.0"),
    ),
    "grasp_and_lift": (
        "grasp_n_lift",
        os.path.join("Grasp and Lift EEG Challenge", "grasp-and-lift-eeg-detection"),
    ),
    # Resting
    "spis_resting_state": (
        "SPIS-Resting-State-Dataset",
        os.path.join("SPIS Resting State Dataset", "SPIS-Resting-State-Dataset-master"),
    ),
    # Emotion (pretrain-only)
    "emobrain": (
        "ems_call",
        os.path.join("EmoBrain", "eNTERFACE06_EMOBRAIN"),
    ),
    # ERP
    "target_versus_non": (
        "target_vs_nontarget",
        os.path.join("Target Versus Non-Target", "Target non Target bi2015a"),
    ),
    # Visual
    "things_eeg": ("ThingsEEG", "THINGS-EEG"),
    # Lingual
    "inner_speech": ("raw_eeg_data", "Inner Speech"),

    # === FINETUNE DATASETS ===
    # BCI Competition
    "bcic_2a": ("BCICIV_2", os.path.join("BCI Competition IV", "2a")),
    "bcic_1a": ("BCICIV_1", os.path.join("BCI Competition IV", "1a")),
    # SEED family
    "seed": ("SEED", os.path.join("SEED", "SEED")),
    "seed_iv": ("SEED_IV", os.path.join("SEED", "SEED_IV")),
    "seed_v": ("SEED-V", os.path.join("SEED", "SEED-V")),
    "seed_vii": ("SEED-VIG", os.path.join("SEED", "SEED-VII", "SEED-VII")),
    "seed_fra": ("SEED_FRA", os.path.join("SEED", "SEED_FRA")),
    # Clinical
    "siena_scalp": ("siena_scalp_eeg", "Siena Scalp EEG Dataset"),
    "adftd": ("ds003775", "ADFTD"),
    # Visual
    "things_eeg_2": ("ThingsEEG", "THINGS-EEG-2"),
    # ERP
    "inria_bci": ("inriaBCI", os.path.join("Infra BCI Challenge", "inria-bci-challenge")),
}

# Recommended pretrain datasets
PRETRAIN_DATASETS = [
    "tueg", "tuab", "tuar", "tusz",
    "spis_resting_state", "motor_mv_img", "grasp_and_lift",
    "emobrain", "target_versus_non", "things_eeg", "inner_speech",
]

# Recommended finetune datasets
FINETUNE_DATASETS = [
    "bcic_2a", "bcic_1a",
    "seed", "seed_iv", "seed_v", "seed_vii",
    "tuab", "tuev", "tusl", "tuep",
    "siena_scalp", "adftd", "things_eeg_2", "inria_bci",
]


def create_symlinks(elements_root: str, eegfm_raw_root: str, datasets: list[str]) -> list[str]:
    """Create symlinks from Elements drive to EEG-FM-Bench raw data layout.

    Returns list of successfully linked dataset IDs.
    """
    linked = []
    for ds_id in datasets:
        if ds_id not in ELEMENTS_TO_EEGFM:
            print(f"[SKIP] {ds_id}: not in mapping")
            continue

        elements_folder, suffix_path = ELEMENTS_TO_EEGFM[ds_id]
        src = os.path.join(elements_root, elements_folder)
        dst = os.path.join(eegfm_raw_root, suffix_path)

        if not os.path.exists(src):
            print(f"[SKIP] {ds_id}: source not found at {src}")
            continue

        if os.path.exists(dst):
            print(f"[OK]   {ds_id}: already exists at {dst}")
            linked.append(ds_id)
            continue

        # Create parent dirs
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        # Create symlink
        os.symlink(src, dst)
        print(f"[LINK] {ds_id}: {src} -> {dst}")
        linked.append(ds_id)

    return linked


def generate_preproc_yaml(
    output_path: str,
    pretrain_datasets: list[str],
    finetune_datasets: list[str],
    fs: int = 256,
) -> None:
    """Generate a preprocessing YAML config for EEG-FM-Bench."""
    lines = [
        f"fs: {fs}",
        "clean_middle_cache: false",
        "clean_shared_info: false",
        "num_preproc_arrow_writers: 8",
        "num_preproc_mid_workers: 8",
        "",
        "pretrain_datasets:",
    ]
    for ds in pretrain_datasets:
        lines.append(f"  - {ds}")

    lines.append("")
    lines.append("finetune_datasets:")
    for ds in finetune_datasets:
        lines.append(f"  {ds}: finetune")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nGenerated config: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Setup and preprocess EEG datasets for LCM pretraining"
    )
    parser.add_argument(
        "--elements_root", type=str, default="/media/meow/Elements",
        help="Path to Elements drive with raw EEG datasets",
    )
    parser.add_argument(
        "--eegfm_root", type=str,
        default="/media/meow/Transcend/time_series_benchmark/EEG-FM-Bench",
        help="Path to EEG-FM-Bench project",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=None,
        help="Specific datasets to process (default: all available)",
    )
    parser.add_argument(
        "--pretrain_only", action="store_true",
        help="Only setup pretrain datasets",
    )
    parser.add_argument(
        "--finetune_only", action="store_true",
        help="Only setup finetune datasets",
    )
    parser.add_argument(
        "--link_only", action="store_true",
        help="Only create symlinks, don't run preprocessing",
    )
    parser.add_argument(
        "--generate_config", type=str, default=None,
        help="Path to write the generated YAML config",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    eegfm_raw_root = os.path.join(args.eegfm_root, "assets", "data", "raw")

    # Determine which datasets to process
    if args.datasets:
        datasets = args.datasets
    elif args.pretrain_only:
        datasets = PRETRAIN_DATASETS
    elif args.finetune_only:
        datasets = FINETUNE_DATASETS
    else:
        datasets = list(ELEMENTS_TO_EEGFM.keys())

    print(f"Elements root: {args.elements_root}")
    print(f"EEG-FM-Bench raw root: {eegfm_raw_root}")
    print(f"Datasets to setup: {len(datasets)}")
    print()

    # Step 1: Create symlinks
    linked = create_symlinks(args.elements_root, eegfm_raw_root, datasets)
    print(f"\n{len(linked)}/{len(datasets)} datasets linked successfully")

    # Step 2: Generate config
    pretrain_linked = [d for d in linked if d in PRETRAIN_DATASETS]
    finetune_linked = [d for d in linked if d in FINETUNE_DATASETS]

    config_path = args.generate_config or os.path.join(
        args.eegfm_root, "assets", "conf", "preproc", "preproc_lcm.yaml"
    )
    generate_preproc_yaml(config_path, pretrain_linked, finetune_linked)

    if args.link_only:
        print("\nSymlinks created. Run EEG-FM-Bench preprocessing manually:")
        print(f"  cd {args.eegfm_root}")
        print(f"  python preproc.py --config {config_path}")
        return

    # Step 3: Run preprocessing
    print("\n" + "=" * 60)
    print("To run preprocessing, execute:")
    print(f"  cd {args.eegfm_root}")
    print(f"  python preproc.py --config {config_path}")
    print()
    print("Note: Preprocessing large datasets (TUEG, TUSZ) can take hours.")
    print("Consider processing in batches by using --datasets flag.")


if __name__ == "__main__":
    main()

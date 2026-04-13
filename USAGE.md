# LCM Usage Guide

## Prerequisites

- Python >= 3.9
- CUDA GPU (recommended)
- Raw EEG datasets on `/media/meow/Elements`
- EEG-FM-Bench at `/media/meow/Transcend/time_series_benchmark/EEG-FM-Bench`

```bash
cd /media/meow/Transcend/time_series_benchmark/LCMv1
pip install -r requirements.txt
```

---

## Step 1: Setup Data Symlinks

Create symlinks from Elements drive to EEG-FM-Bench expected layout:

```bash
# Link all available datasets
python -m lcm.preprocess_eegfm \
    --elements_root /media/meow/Elements \
    --eegfm_root /media/meow/Transcend/time_series_benchmark/EEG-FM-Bench \
    --link_only

# Or link only pretrain datasets
python -m lcm.preprocess_eegfm \
    --elements_root /media/meow/Elements \
    --eegfm_root /media/meow/Transcend/time_series_benchmark/EEG-FM-Bench \
    --pretrain_only --link_only

# Or link only finetune datasets
python -m lcm.preprocess_eegfm \
    --elements_root /media/meow/Elements \
    --eegfm_root /media/meow/Transcend/time_series_benchmark/EEG-FM-Bench \
    --finetune_only --link_only

# Or link specific datasets
python -m lcm.preprocess_eegfm \
    --elements_root /media/meow/Elements \
    --eegfm_root /media/meow/Transcend/time_series_benchmark/EEG-FM-Bench \
    --datasets tuab spis_resting_state bcic_2a seed_iv \
    --link_only
```

This generates a YAML config at `EEG-FM-Bench/assets/conf/preproc/preproc_lcm.yaml`.

---

## Step 2: Preprocess with EEG-FM-Bench

Run the EEG-FM-Bench preprocessing pipeline (resample to 256Hz, windowing, Arrow format):

```bash
cd /media/meow/Transcend/time_series_benchmark/EEG-FM-Bench

# Process all datasets in the generated config
python preproc.py --config assets/conf/preproc/preproc_lcm.yaml
```

> **Note:** Processing large datasets (TUEG, TUSZ) can take several hours. Consider processing in batches by editing the YAML config.

Processed data will be saved at:
```
EEG-FM-Bench/assets/data/processed/fs_256/{dataset_name}/{config}/1.0.0/*.arrow
```

---

## Step 3: Pretrain (Self-Supervised)

The pretrain script auto-detects all available preprocessed datasets on disk:

```bash
cd /media/meow/Transcend/time_series_benchmark/LCMv1

# Standard pretraining
python -m lcm.train_pretrain_eegfm \
    --eegfm_root /media/meow/Transcend/time_series_benchmark/EEG-FM-Bench/assets/data/processed/fs_256 \
    --epochs 200 \
    --batch_size 64 \
    --lr 1.5e-4 \
    --seed 42 \
    --gpu 0

# With wandb logging
python -m lcm.train_pretrain_eegfm \
    --eegfm_root /media/meow/Transcend/time_series_benchmark/EEG-FM-Bench/assets/data/processed/fs_256 \
    --epochs 200 \
    --batch_size 64 \
    --gpu 0 \
    --use_wandb --wandb_project lcm-pretrain
```

Checkpoints saved at `checkpoints/pretrain_eegfm/checkpoint_epoch{N}.pt`.

---

## Step 4: Fine-Tune on Downstream Tasks

### Full Fine-Tuning (with pretrained weights)

```bash
python -m lcm.train_finetune_eegfm \
    --dataset bcic_2a \
    --pretrained_path checkpoints/pretrain_eegfm/checkpoint_epoch200.pt \
    --eegfm_root /media/meow/Transcend/time_series_benchmark/EEG-FM-Bench/assets/data/processed/fs_256 \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4 \
    --seeds 42 123 456 \
    --gpu 0
```

### Linear Probing (freeze encoder)

```bash
python -m lcm.train_finetune_eegfm \
    --dataset bcic_2a \
    --pretrained_path checkpoints/pretrain_eegfm/checkpoint_epoch200.pt \
    --freeze_encoder \
    --epochs 100 \
    --seeds 42 123 456 \
    --gpu 0
```

### From Scratch (no pretrain baseline)

```bash
python -m lcm.train_finetune_eegfm \
    --dataset bcic_2a \
    --epochs 100 \
    --seeds 42 123 456 \
    --gpu 0
```

### Available Finetune Datasets

```bash
# Motor Imagery
--dataset bcic_2a       # 4 classes (left, right, foot, tongue)
--dataset bcic_1a       # 3 classes (left, right, foot)
--dataset motor_mv_img  # 4 classes (left, right, both_fist, foot)

# Emotion
--dataset seed          # 3 classes (sad, neutral, happy)
--dataset seed_iv       # 4 classes (neutral, sad, fear, happy)
--dataset seed_v        # 5 classes (disgust, fear, sad, neutral, happy)
--dataset seed_vii      # 7 classes (disgust, fear, sad, neutral, happy, anger, surprise)

# Clinical
--dataset tuab          # 2 classes (normal, abnormal)
--dataset tuev          # 6 classes (spsw, gped, pled, eyem, artf, bckg)
--dataset tusl          # 3 classes (seiz, slow, bckg)
--dataset tuep          # 2 classes (epilepsy, no_epilepsy)
--dataset siena_scalp   # 2 classes (seizure, normal)
--dataset adftd         # 3 classes (AD, FTD, CN)

# Cognitive / Visual
--dataset things_eeg_2  # 2 classes (non-target, target)
--dataset inria_bci     # 2 classes (wrong, correct)
```

---

## Step 5: Ablation Experiments

Add these flags to the pretrain command:

```bash
# Disable contrastive loss (keep only reconstruction)
python -m lcm.train_pretrain_eegfm --no_contrastive --gpu 0

# Disable reconstruction loss (keep only contrastive)
python -m lcm.train_pretrain_eegfm --no_reconstruction --gpu 0

# Disable channel mapping (use zero-padding instead)
python -m lcm.train_pretrain_eegfm --no_channel_mapping --gpu 0

# SimSiam style (no momentum encoder)
python -m lcm.train_pretrain_eegfm --no_momentum --gpu 0

# Sweep mask ratio
python -m lcm.train_pretrain_eegfm --mask_ratio 0.5 --gpu 0
python -m lcm.train_pretrain_eegfm --mask_ratio 0.6 --gpu 0
python -m lcm.train_pretrain_eegfm --mask_ratio 0.75 --gpu 0  # default
python -m lcm.train_pretrain_eegfm --mask_ratio 0.85 --gpu 0

# Sweep reconstruction weight (lambda)
python -m lcm.train_pretrain_eegfm --recon_weight 0.1 --gpu 0
python -m lcm.train_pretrain_eegfm --recon_weight 0.5 --gpu 0
python -m lcm.train_pretrain_eegfm --recon_weight 1.0 --gpu 0  # default
python -m lcm.train_pretrain_eegfm --recon_weight 2.0 --gpu 0
```

---

## Full Benchmark Run (All 15 Downstream Tasks)

```bash
PRETRAINED=checkpoints/pretrain_eegfm/checkpoint_epoch200.pt
EEGFM=/media/meow/Transcend/time_series_benchmark/EEG-FM-Bench/assets/data/processed/fs_256

for DATASET in bcic_2a bcic_1a motor_mv_img \
               seed seed_iv seed_v seed_vii \
               tuab tuev tusl tuep \
               siena_scalp adftd things_eeg_2 inria_bci; do
    echo "=== Fine-tuning on $DATASET ==="
    python -m lcm.train_finetune_eegfm \
        --dataset $DATASET \
        --pretrained_path $PRETRAINED \
        --eegfm_root $EEGFM \
        --epochs 100 \
        --seeds 42 123 456 \
        --gpu 0 \
        --checkpoint_dir checkpoints/finetune_eegfm/$DATASET
done
```

---

## Output Structure

```
checkpoints/
├── pretrain_eegfm/
│   ├── pretrain.log
│   ├── checkpoint_epoch10.pt
│   ├── checkpoint_epoch20.pt
│   └── checkpoint_epoch200.pt
└── finetune_eegfm/
    ├── bcic_2a/
    │   └── finetune_bcic_2a.log
    ├── seed_iv/
    │   └── finetune_seed_iv.log
    └── ...
```

Logs contain per-seed results with mean ± std for:
- Balanced Accuracy
- Cohen's Kappa
- Weighted F1 (multi-class) or AUROC (binary)

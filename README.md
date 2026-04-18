# LCM (Large Cognition Model) Implementation

Self-supervised EEG foundation model using momentum contrastive learning + masked reconstruction for pretraining, with downstream fine-tuning on EEG classification tasks.

**Paper:** "Large Cognition Model: Towards Pretrained Electroencephalography (EEG) Foundation Model" (Chen et al., 2025, arXiv:2502.17464)

> ⚠️ **Note:** The original code was lost. This is a re-implementation attempted one year later. Due to environment differences and hyperparameter tuning issues, the performance reported in the arXiv paper has not yet been reproduced.

---

## Project Structure

```
LCMv1/
├── requirements.txt
└── lcm/
    ├── config.py                 # ModelConfig, PretrainConfig, FinetuneConfig, DataConfig
    ├── model/
    │   ├── channel_mapping.py    # Learnable cross-montage channel mapping (Eq. 1)
    │   ├── patch_embed.py        # Spatio-temporal patching + embedding (Eq. 2)
    │   ├── encoder.py            # ConvBlock + Transformer encoder backbone
    │   ├── reconstructor.py      # Masked patch reconstruction head
    │   ├── lcm.py                # Full LCM: online encoder + momentum target encoder + losses
    │   └── classifier.py         # Downstream classification head with [CLS] token
    ├── data/
    │   ├── datasets.py           # EEGPretrainDataset, EEGFinetuneDataset, DataLoaders
    │   ├── preprocessing.py      # Resample, bandpass filter, segment, re-reference
    │   └── utils.py              # Channel name mappings, montage info, collate_fn
    ├── train_pretrain.py         # Self-supervised pretraining script
    ├── train_finetune.py         # Downstream fine-tuning script (subject-wise CV)
    ├── evaluate.py               # Balanced accuracy, Cohen's kappa, weighted F1, AUROC
    └── utils.py                  # Seed, logging, gradient monitoring, cosine warmup scheduler
```

---

## Architecture

```
Raw EEG [B, M, T]
    │
    ▼
Channel Mapping (Eq. 1)          W_c: [max_channels] → [M'=22] + channel embedding
    │
    ▼
Patch Embedding (Eq. 2)          Per-channel temporal patches → token sequence [B, N, D]
    │
    ▼
ConvBlock                         Lightweight 1D conv for local temporal features
    │
    ▼
Transformer Encoder (×10)        Pre-norm, multi-head self-attention + FFN
    │
    ▼
Output [B, N, D]
```

### Pretraining (Self-Supervised)

- **Online encoder** `f_θ`: receives masked input, updated by gradient descent
- **Target encoder** `f_ξ`: receives full input, updated by EMA (momentum 0.996 → 1.0)
- **Loss:** `L = L_A + λ · L_R`
  - `L_A` — contrastive alignment loss (MSE on layer-normed features)
  - `L_R` — reconstruction loss (MSE on masked patches)

### Fine-Tuning (Downstream)

- Prepend a learnable `[CLS]` token
- Classification head: `LayerNorm → Linear`
- Subject-wise cross-validation, full fine-tuning or linear probing

---

## Model Configuration

| Parameter | Value |
|-----------|-------|
| `embed_dim` | 512 |
| `num_layers` | 10 |
| `num_heads` | 8 |
| `mlp_ratio` | 4.0 |
| `conv_channels` | [256, 512] |
| `unified_channels (M')` | 22 |
| `patch_size_time` | 64 |
| `mask_ratio` | 0.75 |
| `Total parameters` | **~33.98M** |

---

## Supported Datasets

| Dataset | Paradigm | Channels | Subjects | Classes |
|---------|----------|----------|----------|---------|
| PhysioMI | MI + ME | 64 | 109 | 5 |
| TSU SSVEP | SSVEP | 64 | 35 | 40 |
| SEED | Emotion | 62 | 15 | 3 |
| BCIC-2A | MI | 22 | 9 | 4 |
| BCIC-2B | MI | 3 | 9 | 2 |

---

## Data Preprocessing

1. Resample to 256 Hz
2. Segment into 4-second windows (1024 samples)
3. Average re-referencing
4. Scale to millivolts
5. Zero-pad channels to `max_channels` (128) for batching
6. (MI downstream only) Bandpass filter 0–38 Hz

---

## Usage with EEG-FM-Bench

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Setup Data (Symlink from Elements drive)

```bash
python -m lcm.preprocess_eegfm \
    --elements_root /media/meow/Elements \
    --eegfm_root /media/meow/Transcend/time_series_benchmark/EEG-FM-Bench \
    --link_only
```

Then run EEG-FM-Bench preprocessing:
```bash
cd /media/meow/Transcend/time_series_benchmark/EEG-FM-Bench
python preproc.py --config assets/conf/preproc/preproc_lcm.yaml
```

### Step 3: Pretraining (Self-Supervised)

```bash
python -m lcm.train_pretrain_eegfm \
    --eegfm_root /path/to/EEG-FM-Bench/assets/data/processed/fs_256 \
    --epochs 200 \
    --batch_size 64 \
    --gpu 0
```

The script auto-detects all available preprocessed datasets on disk.

### Step 4: Fine-Tuning

```bash
python -m lcm.train_finetune_eegfm \
    --dataset bcic_2a \
    --pretrained_path checkpoints/pretrain_eegfm/checkpoint_epoch200.pt \
    --epochs 100 \
    --seeds 42 123 456 \
    --gpu 0
```

### Fine-Tuning (From Scratch, No Pretrain)

```bash
python -m lcm.train_finetune_eegfm --dataset bcic_2a --gpu 0
```

### Fine-Tuning (Linear Probing)

```bash
python -m lcm.train_finetune_eegfm \
    --dataset bcic_2a \
    --pretrained_path checkpoints/pretrain_eegfm/checkpoint_epoch200.pt \
    --freeze_encoder --gpu 0
```

---

## Pretrain / Finetune Dataset Split

### Pretrain Datasets (Self-Supervised, 11 datasets)

| Dataset | EEG-FM-Bench ID | Channels | Paradigm |
|---------|-----------------|----------|----------|
| TU EEG Corpus | `tueg` | 22 (TCP) | Clinical (largest) |
| TU Abnormal | `tuab` | 22 | Clinical |
| TU Artifact | `tuar` | 22 | Artifact |
| TU Seizure | `tusz` | 22 | Seizure |
| SPIS Resting State | `spis_resting_state` | 64 | Resting |
| PhysioMI | `motor_mv_img` | 64 | Motor Imagery + Execution |
| Grasp & Lift | `grasp_and_lift` | 32 | Motor Execution |
| EmoBrain | `emobrain` | 64 | Emotion |
| Target vs NonTarget | `target_versus_non` | 32 | ERP/P300 |
| THINGS-EEG | `things_eeg` | 59 | Visual |
| Inner Speech | `inner_speech` | 128 | Lingual |

### Finetune Datasets (14 Downstream Tasks)

| Dataset | EEG-FM-Bench ID | Classes | Task |
|---------|-----------------|---------|------|
| BCIC-2A | `bcic_2a` | 4 | Motor Imagery |
| BCIC-1A | `bcic_1a` | 3 | Motor Imagery |
| PhysioMI | `motor_mv_img` | 4 | Motor Imagery |
| SEED | `seed` | 3 | Emotion |
| SEED-IV | `seed_iv` | 4 | Emotion |
| SEED-V | `seed_v` | 5 | Emotion |
| SEED-VII | `seed_vii` | 7 | Emotion |
| TUAB | `tuab` | 2 | Abnormal Detection |
| TUEV | `tuev` | 6 | Event Classification |
| TUSL | `tusl` | 3 | Slowing Detection |
| TUEP | `tuep` | 2 | Epilepsy Detection |
| Siena Scalp | `siena_scalp` | 2 | Seizure Detection |
| ADFTD | `adftd` | 3 | Dementia Classification |
| THINGS-EEG-2 | `things_eeg_2` | 2 | Visual Target Detection |
| Inria BCI | `inria_bci` | 2 | P300 BCI |

---

## Ablation Experiments

Toggle components via CLI flags:

| Flag | Effect |
|------|--------|
| `--no_contrastive` | Disable L_A, keep only L_R |
| `--no_reconstruction` | Disable L_R, keep only L_A |
| `--no_channel_mapping` | Bypass W_c, use zero-padding |
| `--no_momentum` | SimSiam style (stop-gradient, no EMA) |
| `--mask_ratio 0.6` | Change mask ratio (default 0.75) |
| `--recon_weight 0.5` | Change λ (default 1.0) |

---

## Expected Results (Table 2 from Paper)

| Dataset | Balanced Acc | Cohen's Kappa | Weighted F1 / AUROC |
|---------|-------------|---------------|---------------------|
| BCIC-2A (no pretrain) | 0.5263 ± 0.0027 | 0.3682 ± 0.0361 | 0.5256 ± 0.0267 |
| BCIC-2A (pretrained) | 0.6166 ± 0.0083 | 0.4619 ± 0.0241 | 0.5932 ± 0.0121 |
| BCIC-2B (no pretrain) | 0.6825 ± 0.1024 | 0.3651 ± 0.2047 | 0.6766 ± 0.1079 |
| BCIC-2B (pretrained) | 0.7523 ± 0.0097 | 0.4731 ± 0.0082 | 0.8244 ± 0.0026 |

---

## Actual Reproduction Results

Fine-tuned on **BCIC-2A** (4 classes) with 3 seeds (42, 123, 456), 100 finetune epochs, CrossEntropy loss. Metric: **balanced accuracy** (mean ± std). These numbers are below the paper — see the reproducibility note at the top of this README.

| Configuration | Bal. Acc. | Cohen's κ | Weighted F1 |
|---|---|---|---|
| No pretrain | 0.3212 ± 0.0220 | 0.0949 ± 0.0293 | 0.3105 ± 0.0253 |
| Pretrain v2 (106,970 samples, 50 epochs) | **0.3640 ± 0.0249** | 0.1520 ± 0.0332 | 0.3477 ± 0.0182 |
| Pretrain v3 (865,399 samples, 10 epochs) | 0.3461 ± 0.0104 | 0.1281 ± 0.0139 | 0.3316 ± 0.0129 |

**Pretrain configuration (both versions):** MSE contrastive + MSE reconstruction, λ = 1.0, mask ratio 0.75.

- **v2:** tuab_chunks (15 chunks) + spis_resting_state (train+val) + bcic_2a (train), 50 epochs
- **v3:** tuab_chunks (125 chunks) + spis_resting_state (train+val) + bcic_2a (train), 10 epochs

**Observations**
- Pretraining helps over no-pretrain by ~3-4 balanced-acc points.
- Scaling pretrain data ×8 (v2 → v3) **does not improve** downstream performance; likely under-trained at 10 epochs. v2 (50 epochs on smaller pool) remains best.
- Absolute accuracy is ~20 points below the paper's 0.6166 — ongoing investigation into dataset version, preprocessing, hyperparameters, and channel mapping differences.

---

## Citation

```bibtex
@article{chen2025large,
  title={Large cognition model: Towards pretrained eeg foundation model},
  author={Chen, Chi-Sheng and Chen, Ying-Jung and Tsai, Aidan Hung-Wen},
  journal={arXiv preprint arXiv:2502.17464},
  year={2025}
}
```

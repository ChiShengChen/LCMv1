# LCM (Large Cognition Model) Implementation

Self-supervised EEG foundation model using momentum contrastive learning + masked reconstruction for pretraining, with downstream fine-tuning on EEG classification tasks.

**Paper:** "Large Cognition Model: Towards Pretrained Electroencephalography (EEG) Foundation Model" (Chen et al., 2025, arXiv:2502.17464)

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

## Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Pretraining

```bash
python -m lcm.train_pretrain \
    --data_root /path/to/data \
    --epochs 200 \
    --batch_size 1024 \
    --lr 1.5e-4 \
    --seed 42 \
    --gpu 0
```

### Fine-Tuning

```bash
python -m lcm.train_finetune \
    --dataset bcic2a \
    --pretrained_path checkpoints/pretrain/checkpoint_epoch200.pt \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4 \
    --seeds 42 123 456 \
    --gpu 0
```

### Fine-Tuning (Linear Probing)

```bash
python -m lcm.train_finetune \
    --dataset bcic2a \
    --pretrained_path checkpoints/pretrain/checkpoint_epoch200.pt \
    --freeze_encoder \
    --gpu 0
```

### Fine-Tuning (From Scratch, No Pretrain)

```bash
python -m lcm.train_finetune \
    --dataset bcic2a \
    --gpu 0
```

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

## Data Directory Layout

Preprocessed data should be organized as:

```
data/
├── physio_mi/
│   └── pretrain/
│       ├── segment_000000.npy    # [C, 1024] float32
│       ├── segment_000001.npy
│       └── ...
├── bcic2a/
│   ├── train/
│   │   ├── subject_000_segments.npy   # [N, C, 1024] float32
│   │   └── subject_000_labels.npy     # [N] int
│   └── test/
│       ├── subject_000_segments.npy
│       └── subject_000_labels.npy
└── ...
```

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

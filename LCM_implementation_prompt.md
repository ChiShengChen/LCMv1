# LCM (Large Cognition Model) — Full Implementation Prompt

## Overview

Implement the **Large Cognition Model (LCM)** from scratch in PyTorch. LCM is a self-supervised EEG foundation model that uses momentum contrastive learning + masked reconstruction for pretraining, then fine-tunes on downstream EEG classification tasks. The model handles cross-montage EEG data via a learnable channel mapping.

**Paper reference:** "Large Cognition Model: Towards Pretrained Electroencephalography (EEG) Foundation Model" (Chen et al., 2025, arXiv:2502.17464)

---

## Project Structure

```
lcm/
├── config.py              # All hyperparameters and dataset configs
├── model/
│   ├── __init__.py
│   ├── channel_mapping.py # Learnable cross-montage channel mapping
│   ├── patch_embed.py     # Spatio-temporal patching + embedding
│   ├── encoder.py         # Conv block + Transformer encoder (the LCM backbone)
│   ├── reconstructor.py   # Masked patch reconstruction head
│   ├── lcm.py             # Full LCM model (online encoder + momentum target encoder + losses)
│   └── classifier.py      # Downstream classification head for fine-tuning
├── data/
│   ├── __init__.py
│   ├── datasets.py        # Dataset classes for PhysioMI, TSU, SEED, BCIC-2A, BCIC-2B
│   ├── preprocessing.py   # EEG preprocessing (resample, filter, segment, re-reference)
│   └── utils.py           # Channel name mappings, montage info, data splits
├── train_pretrain.py      # Self-supervised pretraining script
├── train_finetune.py      # Downstream fine-tuning script
├── evaluate.py            # Evaluation (balanced accuracy, Cohen's kappa, weighted F1, AUROC)
└── utils.py               # Logging, EMA update, gradient monitoring, checkpointing
```

---

## 1. Configuration (`config.py`)

Define dataclass-based configs:

```python
@dataclass
class ModelConfig:
    # Channel mapping
    unified_channels: int = 22         # M' — target unified channel dim
    channel_embed_dim: int = 256       # d — channel embedding dimension

    # Patching
    patch_size_time: int = 64          # temporal patch length (samples)
    # Note: paper says p×p patches but EEG channels are discrete;
    # treat each channel independently, patch only along time axis.
    # Each token = one channel's one time-patch → total tokens = M' × (T // patch_size_time)

    # Transformer encoder
    embed_dim: int = 256               # transformer hidden dim
    num_heads: int = 8
    num_layers: int = 8                # depth (paper: 33.9M params total — calibrate depth/width to hit this)
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    # Conv block (before transformer)
    conv_channels: list = field(default_factory=lambda: [64, 128, 256])
    conv_kernel_size: int = 7
    conv_stride: int = 1

    # Masking
    mask_ratio: float = 0.75           # paper doesn't specify; use MAE default

    # Momentum encoder
    ema_momentum_start: float = 0.996
    ema_momentum_end: float = 1.0

    # Loss
    reconstruction_weight: float = 1.0  # λ in L = L_A + λ * L_R

@dataclass
class PretrainConfig:
    epochs: int = 200
    batch_size: int = 1024
    lr: float = 1.5e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    betas: tuple = (0.9, 0.95)
    seed: int = 42

@dataclass
class FinetuneConfig:
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    min_lr: float = 1e-6

@dataclass
class DataConfig:
    sample_rate: int = 256             # Hz, all data resampled to this
    segment_length_sec: float = 4.0    # seconds
    segment_length: int = 1024         # = sample_rate * segment_length_sec
    bandpass_low: float = 0.0          # Hz (for MI downstream)
    bandpass_high: float = 38.0        # Hz (for MI downstream)
```

**Important:** Calibrate `num_layers` and `embed_dim` so total model params ≈ 33.9M. Start with embed_dim=256, num_layers=8, then count params and adjust.

---

## 2. Channel Mapping Module (`channel_mapping.py`)

### Purpose
Map EEG from any montage (M channels) to a unified space (M' channels). This is the cross-montage adapter.

### Implementation (Eq. 1 from paper)

```
x̃ = W_c @ x + bias
```

- Input: `x ∈ [B, M, T]` — raw EEG with M channels (varies per dataset)
- `W_c ∈ [M', M]` — learnable linear mapping (nn.Linear)
- Output: `x̃ ∈ [B, M', T]` — unified channel space
- Additionally output a channel embedding: `channel_emb ∈ [M', d]` via nn.Embedding

**Design notes:**
- Each dataset has different M. Use `nn.Linear(M, M', bias=True)` — instantiate per-dataset or use a padded approach (pad smaller montages to max M with zeros, then apply shared W_c).
- **Recommended approach:** Define max_channels (e.g., 128). Zero-pad all inputs to max_channels. Use a single `nn.Linear(max_channels, M')`. Mask padded channels in the loss if needed.
- Channel embedding is a lookup table: `nn.Embedding(M', embed_dim)` that gets added to each token from that channel.

---

## 3. Patch Embedding (`patch_embed.py`)

### Purpose
Segment the unified EEG into spatio-temporal tokens for the transformer.

### Implementation (Eq. 2)

```
Input:  x̃ ∈ [B, M', T]          e.g., [B, 22, 1024]
Step 1: Reshape to per-channel segments
        → [B, M', num_patches, patch_size]   e.g., [B, 22, 16, 64]
Step 2: Linear projection per patch
        → [B, M', num_patches, embed_dim]    e.g., [B, 22, 16, 256]
Step 3: Add channel embedding (from channel index)
        channel_emb[ch_idx] broadcast over patches
Step 4: Add temporal positional embedding (learnable, over patch index)
Step 5: Flatten spatial and temporal dims into token sequence
        → [B, M' * num_patches, embed_dim]   e.g., [B, 352, 256]
```

Use `nn.Linear(patch_size_time, embed_dim)` or a 1D conv with kernel=stride=patch_size for the projection. Both are equivalent.

---

## 4. Encoder (`encoder.py`)

### Architecture: Conv Block → Transformer

**Conv Block** (processes tokens before transformer):
- Purpose: Extract local temporal features before global attention
- Stack of 1D conv layers: e.g., Conv1d → BatchNorm → GELU → Conv1d → BatchNorm → GELU
- Applied per-token or as a shared MLP-like block on the embed_dim
- Keep it lightweight (2-3 layers)

**Transformer:**
- Standard ViT-style transformer encoder
- Pre-norm (LayerNorm before attention and FFN)
- Multi-head self-attention across all spatio-temporal tokens
- FFN with GELU activation, expansion ratio 4x
- Learnable [CLS] token optional (for downstream classification, prepend a CLS token)

```python
class LCMEncoder(nn.Module):
    def __init__(self, config):
        self.conv_block = ConvBlock(config)       # local feature extraction
        self.transformer = TransformerEncoder(config)  # global attention
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, tokens, mask=None):
        # tokens: [B, N, D]
        x = self.conv_block(tokens)    # [B, N, D]
        x = self.transformer(x)        # [B, N, D]
        x = self.norm(x)               # [B, N, D]
        return x
```

---

## 5. Full LCM Model (`lcm.py`)

### Architecture Overview

Two encoders with identical architecture:
1. **Online encoder** `f_θ`: receives masked input, parameters updated by gradient descent
2. **Target encoder** `f_ξ`: receives full (unmasked) input, parameters updated by EMA

### Forward Pass (Pretraining)

```python
def forward(self, x, channel_counts):
    """
    x: [B, max_channels, T] — zero-padded raw EEG
    channel_counts: [B] — actual number of channels per sample
    """
    # 1. Channel mapping
    x_mapped = self.channel_mapping(x)           # [B, M', T]

    # 2. Patch embedding + positional/channel encoding
    tokens = self.patch_embed(x_mapped)           # [B, N, D]

    # 3. Generate mask
    mask = self.generate_mask(tokens)              # [B, N] boolean

    # 4. Online encoder: masked input
    masked_tokens = tokens.clone()
    masked_tokens[mask] = self.mask_token          # learnable mask token [D]
    z = self.online_encoder(masked_tokens)         # [B, N, D]

    # 5. Target encoder: full input (no grad)
    with torch.no_grad():
        h = self.target_encoder(tokens)            # [B, N, D]

    # 6. Contrastive alignment loss (Eq. 6)
    # CRITICAL: only backprop through z, not h (stop gradient on target)
    L_A = F.mse_loss(
        F.layer_norm(z, [z.size(-1)]),
        F.layer_norm(h, [h.size(-1)]).detach()
    )

    # 7. Reconstruction loss (Eq. 9) — only on masked positions
    x_hat = self.reconstructor(z)                  # [B, N, patch_size]
    # Get original patch values for masked positions
    original_patches = self.get_original_patches(x_mapped)  # [B, N, patch_size]
    L_R = F.mse_loss(x_hat[mask], original_patches[mask])

    # 8. Total loss (Eq. 12)
    loss = L_A + self.config.reconstruction_weight * L_R

    return loss, L_A.item(), L_R.item()
```

### EMA Update (after each step)

```python
@torch.no_grad()
def update_target_encoder(self, current_step, total_steps):
    # Cosine schedule for momentum (from 0.996 → 1.0)
    m = self.config.ema_momentum_end - (self.config.ema_momentum_end - self.config.ema_momentum_start) * \
        (math.cos(math.pi * current_step / total_steps) + 1) / 2
    for param_o, param_t in zip(self.online_encoder.parameters(),
                                 self.target_encoder.parameters()):
        param_t.data = m * param_t.data + (1 - m) * param_o.data
```

### Mask Generation

```python
def generate_mask(self, tokens):
    B, N, D = tokens.shape
    num_mask = int(N * self.config.mask_ratio)
    # Per-sample random masking
    mask = torch.zeros(B, N, dtype=torch.bool, device=tokens.device)
    for i in range(B):
        indices = torch.randperm(N)[:num_mask]
        mask[i, indices] = True
    return mask
```

---

## 6. Downstream Classifier (`classifier.py`)

For fine-tuning:

```python
class LCMClassifier(nn.Module):
    def __init__(self, pretrained_lcm, num_classes):
        self.channel_mapping = pretrained_lcm.channel_mapping
        self.patch_embed = pretrained_lcm.patch_embed
        self.encoder = pretrained_lcm.online_encoder  # use online encoder
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x_mapped = self.channel_mapping(x)
        tokens = self.patch_embed(x_mapped)
        # Prepend CLS token
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        features = self.encoder(tokens)
        cls_out = features[:, 0]  # CLS token output
        return self.classifier(cls_out)
```

Fine-tune the **entire model** (not just the head) with a smaller learning rate.

---

## 7. Data Preprocessing (`preprocessing.py`)

### Common preprocessing for ALL datasets:
1. Resample to 256 Hz
2. Segment into 4-second windows (= 1024 samples)
3. Average re-referencing
4. Scale to millivolts (μV → mV if needed)
5. Zero-pad channels to `max_channels` for batching

### Additional for MI downstream (BCIC-2A, BCIC-2B):
6. Bandpass filter 0–38 Hz (use `mne.filter.filter_data` or `scipy.signal.butter` + `filtfilt`)

### Dataset-specific details:

| Dataset | Paradigm | Channels | Subjects | Classes | Source |
|---------|----------|----------|----------|---------|--------|
| PhysioMI (EEG Motor Movement/Imagery) | MI + ME | 64 | 109 | 5 | physionet.org |
| TSU SSVEP | SSVEP | 64 (use 8-10 occipital) | 35 | 40 | SSVEP benchmark |
| SEED | Emotion | 62 | 15 | 3 | BCMI-SJTU |
| BCIC-2A | MI | 22 | 9 (+1 eval) | 4 | BCI Competition IV |
| BCIC-2B | MI | 3 | 9 (+1 eval) | 2 | BCI Competition IV |

Use `mne` library for loading and preprocessing. For BCIC datasets, load from `.gdf` files with `mne.io.read_raw_gdf()`.

### Data loading for pretraining:
- Combine all pretraining datasets into one `ConcatDataset`
- Each sample returns: `(eeg_segment, channel_count, dataset_id)`
- Use a custom collate function that zero-pads to max_channels in the batch

---

## 8. Training Scripts

### `train_pretrain.py`

```python
# Pseudocode
model = LCM(model_config)
optimizer = AdamW(model.online_encoder.parameters(),  # only optimize online encoder
                  lr=config.lr, weight_decay=config.weight_decay, betas=config.betas)
scheduler = CosineAnnealingLR with warmup:
    - Linear warmup from 0 → lr over first 10 epochs
    - Cosine decay from lr → 1e-6 over remaining epochs

for epoch in range(200):
    for batch in dataloader:
        loss, l_a, l_r = model(batch)
        loss.backward()

        # Gradient logging (Eq. 13-14)
        log_gradient_stats(model)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        # EMA update target encoder
        model.update_target_encoder(global_step, total_steps)

    save_checkpoint(model, epoch)
```

### `train_finetune.py`

```python
# Load pretrained LCM
pretrained = LCM.load_from_checkpoint(path)
model = LCMClassifier(pretrained, num_classes=4)  # e.g., BCIC-2A has 4 classes

# Standard supervised training with cross-entropy loss
# Use subject-wise cross-validation (leave-one-subject-out or k-fold)
# Report: balanced accuracy, Cohen's kappa, weighted F1
```

---

## 9. Evaluation Metrics (`evaluate.py`)

Compute and report these three metrics (as in Table 2):

1. **Balanced Accuracy**: `sklearn.metrics.balanced_accuracy_score`
2. **Cohen's Kappa**: `sklearn.metrics.cohen_kappa_score`
3. **Weighted F1 / AUROC**:
   - For multi-class (BCIC-2A, 4 classes): weighted F1 via `sklearn.metrics.f1_score(average='weighted')`
   - For binary (BCIC-2B, 2 classes): AUROC via `sklearn.metrics.roc_auc_score`

Run **3 random seeds** and report mean ± std (matching paper's format).

---

## 10. Key Implementation Notes

### Things the paper is ambiguous about — make these choices:

1. **Patch strategy:** Paper says patches are p×p but EEG channels are discrete electrodes. → Patch only along time axis per channel. Each token = (1 channel) × (patch_size_time samples).

2. **Mask ratio:** Not specified. Use 0.75 (MAE default). Consider ablating [0.5, 0.6, 0.75, 0.85].

3. **λ (reconstruction weight):** Not specified. Start with 1.0, ablate [0.1, 0.5, 1.0, 2.0].

4. **Mask token:** Use a learnable `[MASK]` embedding vector of size embed_dim.

5. **Whether target encoder sees masked or unmasked input:** Standard BYOL/data2vec practice → target encoder sees the FULL unmasked input. Online encoder sees masked input.

6. **Stop gradient:** Target encoder output h must be `.detach()`'d. Only backprop through online encoder z.

7. **Downstream protocol:** Paper doesn't specify if it's linear probing or full fine-tuning. Given the results, it's likely **full fine-tuning**. Implement both options.

8. **Cross-validation:** BCIC-2A and 2B use **subject-dependent** evaluation (train/test split per subject, then average across subjects).

### Parameter count target: 33.9M

Rough calculation to hit 33.9M:
- embed_dim=256, num_layers=8, num_heads=8, mlp_ratio=4
- Per transformer layer: 4 * D² (attention) + 2 * D * 4D (FFN) = 4*65536 + 2*262144 ≈ 786K
- 8 layers ≈ 6.3M
- Patch embedding + channel mapping ≈ 1-2M
- That's only ~8M. Need to increase.

→ Try **embed_dim=512, num_layers=6, num_heads=8**:
- Per layer: 4*512² + 2*512*2048 ≈ 1.05M + 2.1M = 3.15M
- 6 layers ≈ 18.9M
- Patch embed (64→512): ~33K * 22 channels ≈ 0.7M
- Conv block: ~5M
- Reconstructor: ~2M
- Total: ~27M → still short

→ Try **embed_dim=512, num_layers=8**:
- 8 * 3.15M = 25.2M + 5M (conv) + 2M (reconstruct) + 1M (embed) ≈ 33M ✓

**Use: embed_dim=512, num_layers=8, num_heads=8, mlp_ratio=4.0**

### Dependencies

```
torch >= 2.0
mne >= 1.6
scikit-learn
numpy
scipy
tqdm
wandb (optional, for logging)
```

---

## 11. Expected Results (Table 2 from paper)

After pretraining + fine-tuning, target these numbers:

| Dataset | Balanced Acc | Cohen's Kappa | Weighted F1/AUROC |
|---------|-------------|---------------|-------------------|
| BCIC-2A (no pretrain) | 0.5263±0.0027 | 0.3682±0.0361 | 0.5256±0.0267 |
| BCIC-2A (pretrained) | 0.6166±0.0083 | 0.4619±0.0241 | 0.5932±0.0121 |
| BCIC-2B (no pretrain) | 0.6825±0.1024 | 0.3651±0.2047 | 0.6766±0.1079 |
| BCIC-2B (pretrained) | 0.7523±0.0097 | 0.4731±0.0082 | 0.8244±0.0026 |

---

## 12. Ablation Experiments to Add (not in paper, but implement the infrastructure)

Create an ablation config that toggles:
1. `use_contrastive_loss: bool` — disable L_A, keep only L_R
2. `use_reconstruction_loss: bool` — disable L_R, keep only L_A
3. `use_channel_mapping: bool` — bypass W_c, use zero-padding instead
4. `use_momentum_encoder: bool` — if False, use stop-gradient on a copy (SimSiam style)
5. `mask_ratio: float` — sweep [0.5, 0.6, 0.75, 0.85]
6. `reconstruction_weight: float` — sweep [0.1, 0.5, 1.0, 2.0]

---

Please implement the full codebase following this specification. Start with `config.py` and `model/` modules, then data pipeline, then training scripts. Verify parameter count matches ~33.9M before proceeding to training. Include type hints and docstrings.

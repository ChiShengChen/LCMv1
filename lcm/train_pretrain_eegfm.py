from __future__ import annotations
"""Pretrain LCM using EEG-FM-Bench processed data."""

import argparse
import os
import time

import torch

from lcm.config import ModelConfig, PretrainConfig, DataConfig
from lcm.model import LCM
from lcm.data.eegfm_adapter import get_eegfm_pretrain_loader
from lcm.utils import (
    set_seed,
    setup_logger,
    count_parameters,
    count_all_parameters,
    log_gradient_stats,
    CosineWarmupScheduler,
    save_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="LCM Pretraining with EEG-FM-Bench Data")
    parser.add_argument(
        "--eegfm_root",
        type=str,
        default="/media/meow/Transcend/time_series_benchmark/EEG-FM-Bench/assets/data/processed/fs_256",
        help="Path to EEG-FM-Bench processed data root",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/pretrain_eegfm")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="lcm-pretrain")
    # Ablation
    parser.add_argument("--no_contrastive", action="store_true")
    parser.add_argument("--no_reconstruction", action="store_true")
    parser.add_argument("--no_channel_mapping", action="store_true")
    parser.add_argument("--no_momentum", action="store_true")
    parser.add_argument("--mask_ratio", type=float, default=None)
    parser.add_argument("--recon_weight", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # Configs
    model_config = ModelConfig()
    pretrain_config = PretrainConfig()
    data_config = DataConfig()

    pretrain_config.epochs = args.epochs
    pretrain_config.batch_size = args.batch_size
    pretrain_config.lr = args.lr
    pretrain_config.seed = args.seed
    pretrain_config.save_every = args.save_every
    pretrain_config.log_every = args.log_every

    # Ablation toggles
    if args.no_contrastive:
        model_config.use_contrastive_loss = False
    if args.no_reconstruction:
        model_config.use_reconstruction_loss = False
    if args.no_channel_mapping:
        model_config.use_channel_mapping = False
    if args.no_momentum:
        model_config.use_momentum_encoder = False
    if args.mask_ratio is not None:
        model_config.mask_ratio = args.mask_ratio
    if args.recon_weight is not None:
        model_config.reconstruction_weight = args.recon_weight

    # Setup
    set_seed(pretrain_config.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger = setup_logger(
        "pretrain_eegfm",
        log_file=os.path.join(args.checkpoint_dir, "pretrain.log"),
    )

    logger.info(f"Model config: {model_config}")
    logger.info(f"Device: {device}")
    logger.info(f"EEG-FM-Bench root: {args.eegfm_root}")

    # Optional wandb
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, config={
            "model": model_config.__dict__,
            "pretrain": pretrain_config.__dict__,
        })

    # Model
    model = LCM(model_config, data_config).to(device)
    trainable = count_parameters(model)
    total = count_all_parameters(model)
    logger.info(f"Trainable parameters: {trainable:,}")
    logger.info(f"Total parameters: {total:,}")

    # Data — use all available EEG-FM-Bench datasets for pretraining
    dataset_configs = [
        # Pretrain dataset
        {"name": "spis_resting_state", "config": "pretrain", "splits": ["train", "validation"]},
        # Use finetune datasets' train splits for pretraining too (labels ignored)
        {"name": "bcic_2a", "config": "finetune", "splits": ["train"]},
        {"name": "seed_iv", "config": "finetune", "splits": ["train"]},
    ]

    dataloader = get_eegfm_pretrain_loader(
        eegfm_processed_root=args.eegfm_root,
        dataset_configs=dataset_configs,
        segment_length=data_config.segment_length,
        max_channels=model_config.max_channels,
        batch_size=pretrain_config.batch_size,
        num_workers=args.num_workers,
    )
    logger.info(f"Training samples: {len(dataloader.dataset):,}")
    logger.info(f"Batches per epoch: {len(dataloader)}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=pretrain_config.lr,
        weight_decay=pretrain_config.weight_decay,
        betas=pretrain_config.betas,
    )

    # Scheduler
    steps_per_epoch = len(dataloader)
    total_steps = pretrain_config.epochs * steps_per_epoch
    warmup_steps = pretrain_config.warmup_epochs * steps_per_epoch
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps, total_steps, pretrain_config.lr, pretrain_config.min_lr
    )

    # Training loop
    global_step = 0
    logger.info("Starting pretraining with EEG-FM-Bench data...")

    for epoch in range(pretrain_config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_la = 0.0
        epoch_lr = 0.0
        epoch_start = time.time()

        for batch_idx, (x, channel_counts, _) in enumerate(dataloader):
            x = x.to(device)

            loss, l_a, l_r = model(x)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Gradient logging
            if global_step % pretrain_config.log_every == 0:
                grad_stats = log_gradient_stats(model, logger)
                if args.use_wandb:
                    import wandb
                    wandb.log({
                        "grad_norm": grad_stats.get("total_grad_norm", 0),
                        "step": global_step,
                    })

            optimizer.step()
            scheduler.step()

            # EMA update target encoder
            model.update_target_encoder(global_step, total_steps)

            epoch_loss += loss.item()
            epoch_la += l_a
            epoch_lr += l_r
            global_step += 1

        # Epoch summary
        n_batches = max(1, len(dataloader))
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / n_batches
        avg_la = epoch_la / n_batches
        avg_lr = epoch_lr / n_batches
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch+1}/{pretrain_config.epochs} | "
            f"Loss: {avg_loss:.4f} (L_A: {avg_la:.4f}, L_R: {avg_lr:.4f}) | "
            f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
        )

        if args.use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "loss_alignment": avg_la,
                "loss_reconstruction": avg_lr,
                "lr": current_lr,
            })

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch == pretrain_config.epochs - 1:
            ckpt_path = os.path.join(
                args.checkpoint_dir, f"checkpoint_epoch{epoch+1}.pt"
            )
            save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, global_step)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    logger.info("Pretraining complete.")

    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()

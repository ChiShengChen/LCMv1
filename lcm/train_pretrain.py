from __future__ import annotations
"""Self-supervised pretraining script for LCM."""

import argparse
import os
import time

import torch

from lcm.config import ModelConfig, PretrainConfig, DataConfig
from lcm.model import LCM
from lcm.data import get_pretrain_loader
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
    parser = argparse.ArgumentParser(description="LCM Self-Supervised Pretraining")
    # Override defaults from config
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/pretrain")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="lcm-pretrain")
    # Ablation toggles
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

    # Apply CLI overrides
    if args.epochs is not None:
        pretrain_config.epochs = args.epochs
    if args.batch_size is not None:
        pretrain_config.batch_size = args.batch_size
    if args.lr is not None:
        pretrain_config.lr = args.lr
    pretrain_config.seed = args.seed
    data_config.data_root = args.data_root
    pretrain_config.checkpoint_dir = args.checkpoint_dir
    pretrain_config.num_workers = args.num_workers
    pretrain_config.use_wandb = args.use_wandb
    pretrain_config.wandb_project = args.wandb_project

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
    os.makedirs(pretrain_config.checkpoint_dir, exist_ok=True)
    logger = setup_logger(
        "pretrain",
        log_file=os.path.join(pretrain_config.checkpoint_dir, "pretrain.log"),
    )

    logger.info(f"Model config: {model_config}")
    logger.info(f"Pretrain config: {pretrain_config}")
    logger.info(f"Data config: {data_config}")
    logger.info(f"Device: {device}")

    # Optional wandb
    if pretrain_config.use_wandb:
        import wandb
        wandb.init(project=pretrain_config.wandb_project, config={
            "model": model_config.__dict__,
            "pretrain": pretrain_config.__dict__,
            "data": data_config.__dict__,
        })

    # Model
    model = LCM(model_config, data_config).to(device)
    trainable = count_parameters(model)
    total = count_all_parameters(model)
    logger.info(f"Trainable parameters: {trainable:,}")
    logger.info(f"Total parameters: {total:,}")

    # Data
    dataloader = get_pretrain_loader(
        data_config,
        max_channels=model_config.max_channels,
        batch_size=pretrain_config.batch_size,
        num_workers=pretrain_config.num_workers,
    )
    logger.info(f"Training samples: {len(dataloader.dataset):,}")

    # Optimizer (only optimize online encoder + channel mapping + patch embed + reconstructor)
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
    logger.info("Starting pretraining...")

    for epoch in range(pretrain_config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_la = 0.0
        epoch_lr = 0.0
        epoch_start = time.time()

        for batch_idx, (x, channel_counts, dataset_ids) in enumerate(dataloader):
            x = x.to(device)

            loss, l_a, l_r = model(x)

            optimizer.zero_grad()
            loss.backward()

            # Gradient logging
            if global_step % pretrain_config.log_every == 0:
                log_gradient_stats(model, logger)

            optimizer.step()
            scheduler.step()

            # EMA update target encoder
            model.update_target_encoder(global_step, total_steps)

            epoch_loss += loss.item()
            epoch_la += l_a
            epoch_lr += l_r
            global_step += 1

        # Epoch summary
        n_batches = len(dataloader)
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

        if pretrain_config.use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "loss_alignment": avg_la,
                "loss_reconstruction": avg_lr,
                "lr": current_lr,
            })

        # Save checkpoint
        if (epoch + 1) % pretrain_config.save_every == 0 or epoch == pretrain_config.epochs - 1:
            ckpt_path = os.path.join(
                pretrain_config.checkpoint_dir, f"checkpoint_epoch{epoch+1}.pt"
            )
            save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, global_step)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    logger.info("Pretraining complete.")

    if pretrain_config.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()

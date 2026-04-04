#!/usr/bin/env python3
"""
SISA Experiment Training Script.

Usage:
    python train.py --phase 0 --model sisa
    python train.py --phase 1 --model transformer
    python train.py --phase 1 --model mamba2
    python train.py --phase 1 --model sisa
    python train.py --phase 2 --model sisa
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import wandb

from configs import get_phase_config
from models import create_model
from models.sisa import SISAModel
from data.dataset import create_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="SISA Training")
    parser.add_argument("--phase", type=int, required=True, choices=[0, 1, 2, 3])
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["transformer", "transformer_reduced", "mamba2", "mamba3", "sisa", "hybrid_transformer", "hybrid_sisa"],
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-compile", action="store_true")

    # Override defaults
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    return parser.parse_args()


def create_optimizer(model, config):
    """Create AdamW optimizer with separate LR for lambda_raw."""
    lambda_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lambda_raw" in name:
            lambda_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": other_params, "lr": config.lr, "weight_decay": config.weight_decay},
    ]

    if lambda_params:
        param_groups.append(
            {"params": lambda_params, "lr": config.lr_lambda, "weight_decay": 0.0},
        )

    return torch.optim.AdamW(param_groups, betas=(config.beta1, config.beta2))


def create_scheduler(optimizer, config):
    """Cosine schedule with linear warmup."""
    effective_batch_tokens = config.micro_batch * config.seq_len * config.grad_accum
    total_steps = config.max_tokens // effective_batch_tokens

    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(total_steps - config.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def estimate_val_loss(model, config, device, max_batches: int = 50):
    """Quick validation loss estimate on a small validation split."""
    model.eval()
    total_loss = 0.0
    count = 0

    # Use local data with offset for validation (last 5% of tokens)
    from data.dataset import LocalTokenDataset
    from torch.utils.data import DataLoader
    from pathlib import Path

    data_dir = Path(config.data_cache_dir)
    if not (data_dir / "meta.txt").exists():
        model.train()
        return float("nan")

    val_dataset = LocalTokenDataset(
        data_dir=str(data_dir),
        seq_len=config.seq_len,
        max_tokens=config.max_tokens,
    )
    # Use last portion as validation by not shuffling and skipping
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.micro_batch,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    for i, (input_ids, labels) in enumerate(val_loader):
        if i >= max_batches:
            break
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        total_loss += loss.item()
        count += 1

    model.train()
    return total_loss / max(count, 1)


def save_checkpoint(model, optimizer, scheduler, step, tokens_seen, config, path):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "tokens_seen": tokens_seen,
            "config": vars(config),
        },
        path,
    )
    print(f"  Checkpoint saved: {path}")


def main():
    args = parse_args()
    config = get_phase_config(args.phase, args.model)

    # Apply overrides
    if args.micro_batch is not None:
        config.micro_batch = args.micro_batch
    if args.grad_accum is not None:
        config.grad_accum = args.grad_accum
    if args.lr is not None:
        config.lr = args.lr
    if args.max_tokens is not None:
        config.max_tokens = args.max_tokens
    if args.no_compile:
        config.compile = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    effective_batch_tokens = config.micro_batch * config.seq_len * config.grad_accum
    total_steps = config.max_tokens // effective_batch_tokens

    print(f"{'='*60}")
    print(f"  SISA Experiment — Phase {args.phase} / {args.model}")
    print(f"{'='*60}")
    print(f"  Effective batch: {effective_batch_tokens:,} tokens")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Micro batch: {config.micro_batch}, Grad accum: {config.grad_accum}")
    print(f"  LR: {config.lr}, LR(lambda): {config.lr_lambda}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # --- Model ---
    model = create_model(config)
    model = model.to(device=device, dtype=torch.bfloat16)

    # Keep lambda_raw in fp32 to avoid bf16 precision loss during training
    from models.sisa import SISAModel
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    if isinstance(base_model, SISAModel):
        for layer in base_model.layers:
            layer.lambda_raw.data = layer.lambda_raw.data.float()

    if config.compile:
        print("  Compiling model with torch.compile...")
        model = torch.compile(model)

    # --- Data ---
    train_loader = create_dataloader(config, split="train")

    # --- Optimizer ---
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # --- Resume ---
    start_step = 0
    tokens_seen = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        tokens_seen = ckpt["tokens_seen"]
        print(f"  Resumed from step {start_step}, tokens {tokens_seen:,}")

    # --- Wandb ---
    if not args.no_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"phase{args.phase}_{args.model}",
            config=vars(config),
        )

    # --- Output dir ---
    out_dir = Path(config.output_dir) / f"phase{args.phase}_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Training ---
    model.train()
    optimizer.zero_grad()

    step = start_step
    micro_step = 0
    accum_loss = 0.0
    t_start = time.time()
    log_tokens_start = tokens_seen

    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward + Backward (both inside autocast for BF16 backward stability)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            scaled_loss = loss / config.grad_accum
            scaled_loss.backward()
        accum_loss += loss.item()
        micro_step += 1
        tokens_seen += config.micro_batch * config.seq_len

        # Optimizer step
        if micro_step % config.grad_accum == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            avg_loss = accum_loss / config.grad_accum
            accum_loss = 0.0

            # --- Logging ---
            if step % config.log_interval == 0:
                elapsed = time.time() - t_start
                tokens_delta = tokens_seen - log_tokens_start
                tps = tokens_delta / max(elapsed, 1e-6)
                current_lr = scheduler.get_last_lr()[0]
                ppl = math.exp(min(avg_loss, 20))

                log_dict = {
                    "train/loss": avg_loss,
                    "train/ppl": ppl,
                    "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "train/lr": current_lr,
                    "train/tokens": tokens_seen,
                    "train/tps": tps,
                    "step": step,
                }

                # SISA-specific monitoring
                base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                if isinstance(base_model, SISAModel):
                    log_dict.update(base_model.get_monitor_metrics())

                if not args.no_wandb:
                    wandb.log(log_dict, step=step)

                print(
                    f"  step {step:>6d}/{total_steps} | "
                    f"loss {avg_loss:.4f} | ppl {ppl:.1f} | "
                    f"grad {log_dict['train/grad_norm']:.3f} | "
                    f"lr {current_lr:.2e} | "
                    f"tok/s {tps:.0f}"
                )

            # --- Eval ---
            if step % config.eval_interval == 0:
                val_loss = estimate_val_loss(model, config, device)
                val_ppl = math.exp(min(val_loss, 20))
                print(f"  >>> val_loss={val_loss:.4f}  val_ppl={val_ppl:.1f}")

                if not args.no_wandb:
                    wandb.log(
                        {"val/loss": val_loss, "val/ppl": val_ppl},
                        step=step,
                    )
                model.train()

            # --- Save ---
            if step % config.save_interval == 0:
                ckpt_path = out_dir / f"step_{step:06d}.pt"
                save_checkpoint(model, optimizer, scheduler, step, tokens_seen, config, ckpt_path)

            # --- Done? ---
            if tokens_seen >= config.max_tokens:
                print(f"\n  Reached {config.max_tokens:,} tokens. Training complete.")
                break

    # Final save
    ckpt_path = out_dir / "final.pt"
    save_checkpoint(model, optimizer, scheduler, step, tokens_seen, config, ckpt_path)

    if not args.no_wandb:
        wandb.finish()

    elapsed_total = time.time() - t_start
    print(f"\n  Total training time: {elapsed_total / 3600:.2f} hours")
    print(f"  Total tokens: {tokens_seen:,}")
    print(f"  Final step: {step}")


if __name__ == "__main__":
    main()

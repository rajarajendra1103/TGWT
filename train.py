"""
Training script for TGWT and Baseline Transformer on WikiText-103.
Optimized for Google Colab with NVIDIA Tesla T4 GPU (Mixed Precision fp16).
"""

import os
import sys
import time
import math
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tgwt.model.tgwt_model import TGWTForCausalLM
from tgwt.model.baseline_llama import BaselineTransformerLM
from tgwt.data.dataloader import get_dataloaders
from tgwt.benchmark.metrics import compute_perplexity, compute_prediction_accuracy


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.05):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device, max_batches: int = 25):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0

    with torch.no_grad():
        for i, (input_ids, labels) in enumerate(val_loader):
            if i >= max_batches:
                break
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            out = model(input_ids, labels=labels)
            if out["loss"] is not None:
                total_loss += out["loss"].item()
                total_acc += compute_prediction_accuracy(out["logits"], labels)
                count += 1

    model.train()
    avg_loss = total_loss / max(1, count)
    avg_acc = total_acc / max(1, count)
    ppl = compute_perplexity(avg_loss)
    return avg_loss, ppl, avg_acc


def main():
    parser = argparse.ArgumentParser(description="Train TGWT or Baseline Transformer")
    parser.add_argument("--config", type=str, default="configs/tgwt_colab_t4.yaml", help="Path to YAML config")
    parser.add_argument("--model_type", type=str, default=None, choices=["tgwt", "baseline"], help="Override model type")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--seq_len", type=int, default=None, help="Override sequence length")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Wandb logging")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint file")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    m_cfg = config.get("model", {})
    t_cfg = config.get("training", {})
    d_cfg = config.get("data", {})

    model_type = args.model_type or m_cfg.get("type", "tgwt")
    max_steps = args.max_steps or t_cfg.get("max_steps", 5000)
    batch_size = args.batch_size or t_cfg.get("batch_size", 4)
    seq_len = args.seq_len or t_cfg.get("seq_len", 1024)
    output_dir = args.output_dir or t_cfg.get("output_dir", f"./checkpoints_{model_type}")
    os.makedirs(output_dir, exist_ok=True)

    # Device & Precision setup
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Executing on Device: {device} ({torch.cuda.get_device_name(0) if use_cuda else 'CPU'})")

    use_amp = use_cuda and (t_cfg.get("mixed_precision", "fp16") == "fp16")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Set seed
    seed = t_cfg.get("seed", 42)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

    # Build model
    print(f"Building {model_type.upper()} model...")
    if model_type == "tgwt":
        model = TGWTForCausalLM(
            vocab_size=m_cfg.get("vocab_size", 50257),
            max_seq_len=m_cfg.get("max_seq_len", 4096),
            d_model=m_cfg.get("d_model", 512),
            n_heads=m_cfg.get("n_heads", 8),
            n_layers=m_cfg.get("n_layers", 4),
            d_prop_node=m_cfg.get("d_prop_node", 64),
            d_prop_edge=m_cfg.get("d_prop_edge", 64),
            walk_length=m_cfg.get("walk_length", 32),
            branch_factor=m_cfg.get("branch_factor", 4),
            oversample_ratio=m_cfg.get("oversample_ratio", 1.5),
            dropout=m_cfg.get("dropout", 0.1),
            tie_weights=m_cfg.get("tie_weights", True),
        )
    else:
        model = BaselineTransformerLM(
            vocab_size=m_cfg.get("vocab_size", 50257),
            max_seq_len=m_cfg.get("max_seq_len", 4096),
            d_model=m_cfg.get("d_model", 512),
            n_heads=m_cfg.get("n_heads", 8),
            n_layers=m_cfg.get("n_layers", 4),
            d_ff=m_cfg.get("d_ff", 2048),
            dropout=m_cfg.get("dropout", 0.1),
            tie_weights=m_cfg.get("tie_weights", True),
        )

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Total Parameters: {total_params:,}")

    # DataLoaders
    print("Loading WikiText-103 dataset...")
    train_loader, val_loader = get_dataloaders(
        seq_len=seq_len,
        batch_size=batch_size,
        cache_dir=d_cfg.get("cache_dir", "./data_cache"),
    )

    # Optimizer & LR Scheduler
    lr = float(t_cfg.get("learning_rate", 3e-4))
    weight_decay = float(t_cfg.get("weight_decay", 0.01))
    warmup_steps = int(t_cfg.get("warmup_steps", 200))
    grad_accum_steps = int(t_cfg.get("gradient_accumulation_steps", 2))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=warmup_steps, max_steps=max_steps)

    # Wandb (Optional)
    if args.use_wandb:
        try:
            import wandb
            wandb.init(project="TGWT-Experiments", config=config, name=f"{model_type}-t4")
        except Exception as e:
            print(f"Wandb not initialized: {e}")

    # Training Loop
    print(f"Starting training for {max_steps} steps (SeqLen={seq_len}, Batch={batch_size}, AMP={use_amp})...")
    model.train()
    step = 0
    running_loss = 0.0
    start_time = time.time()
    best_val_loss = float("inf")

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "step" in ckpt:
            step = ckpt["step"]
            for _ in range(step):
                scheduler.step()
        if "val_loss" in ckpt:
            best_val_loss = ckpt["val_loss"]
        print(f"Successfully resumed from step {step} (Best Val Loss: {best_val_loss:.4f})!")

    data_iter = iter(train_loader)

    while step < max_steps:
        try:
            input_ids, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            input_ids, labels = next(data_iter)

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp)
        else:
            autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)

        with autocast_ctx:
            out = model(input_ids, labels=labels)
            loss = out["loss"] / grad_accum_steps

        scaler.scale(loss).backward()
        running_loss += loss.item() * grad_accum_steps

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        step += 1

        # Logging
        log_interval = t_cfg.get("log_interval", 50)
        if step % log_interval == 0:
            avg_train_loss = running_loss / log_interval
            train_ppl = compute_perplexity(avg_train_loss)
            elapsed = time.time() - start_time
            tok_per_sec = (log_interval * batch_size * seq_len) / max(elapsed, 1e-6)
            current_lr = scheduler.get_last_lr()[0]
            vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if use_cuda else 0.0

            print(
                f"Step [{step:5d}/{max_steps}] | Loss: {avg_train_loss:.4f} | PPL: {train_ppl:.2f} | "
                f"Speed: {tok_per_sec:.1f} tok/s | VRAM: {vram_mb:.1f}MB | LR: {current_lr:.2e}"
            )
            running_loss = 0.0
            start_time = time.time()

        # Evaluation
        eval_interval = t_cfg.get("eval_interval", 250)
        if step % eval_interval == 0 or step == max_steps:
            val_loss, val_ppl, val_acc = evaluate(model, val_loader, device=device)
            print(f">>> EVALUATION @ Step {step}: Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Acc: {val_acc:.2f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(output_dir, f"best_model_{model_type}.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "config": config,
                }, ckpt_path)
                print(f">>> Best model saved to {ckpt_path}")

    print(f"Training completed successfully! Best Validation Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

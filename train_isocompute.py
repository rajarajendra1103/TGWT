"""
Fast Iso-Compute Training & Head-to-Head Evaluation Pipeline.
Trains both TGWT and Baseline Transformer for an identical token budget (e.g. 2,000 steps)
using locked seeds, AdamW, and cosine learning rate decay on WikiText-103.
Evaluates final Perplexity, Top-1 Accuracy, Throughput, VRAM, and Needle-in-a-Haystack (NIAH).
"""

import os
import sys
import time
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tgwt.benchmark.compare import create_matched_models
from tgwt.benchmark.metrics import compute_perplexity, compute_prediction_accuracy
from tgwt.benchmark.niah import evaluate_niah_retrieval
from tgwt.data.dataloader import get_dataloaders


def get_cosine_schedule(optimizer, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.05):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_dataset(model: nn.Module, val_loader: DataLoader, device: torch.device, max_batches: int = 30):
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
    return avg_loss, compute_perplexity(avg_loss), avg_acc


def run_training_loop(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_steps: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    lr: float = 3e-4,
    seed: int = 42,
) -> Dict:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule(optimizer, warmup_steps=int(0.1 * max_steps), max_steps=max_steps)

    use_amp = (device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"\n{'='*70}\nStarting Iso-Compute Training: {model_name} ({max_steps} steps)\n{'='*70}")
    model.train()
    step = 0
    running_loss = 0.0
    total_tokens = 0
    start_time = time.time()

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
            loss = out["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        running_loss += loss.item()
        total_tokens += input_ids.numel()
        step += 1

        if step % 200 == 0 or step == max_steps:
            avg_loss = running_loss / (200 if step % 200 == 0 else (step % 200))
            cur_ppl = compute_perplexity(avg_loss)
            vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if use_amp else 0.0
            print(f"[{model_name}] Step {step:4d}/{max_steps} | Loss: {avg_loss:.4f} | PPL: {cur_ppl:.2f} | Peak VRAM: {vram:.1f} MB")
            running_loss = 0.0

    elapsed = time.time() - start_time
    train_throughput = total_tokens / max(1e-6, elapsed)
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if use_amp else 0.0

    # Final Validation
    val_loss, val_ppl, val_acc = evaluate_dataset(model, val_loader, device=device)
    print(f"\nFinal {model_name} Evaluation -> Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Accuracy: {val_acc:.2f}%")

    return {
        "model_name": model_name,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "val_acc": val_acc,
        "throughput": train_throughput,
        "peak_vram": peak_vram,
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Iso-Compute Head-to-Head Training Pipeline")
    parser.add_argument("--steps", type=int, default=2000, help="Number of training steps per model (default: 2000)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per step")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--d_model", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--walk_length", type=int, default=32, help="Walk length for TGWT")
    parser.add_argument("--cache_dir", type=str, default="./data_cache", help="Dataset cache directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Iso-Compute Benchmark on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Data
    print("Loading WikiText-103 dataset...")
    train_loader, val_loader = get_dataloaders(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    )

    # 2. Build Models with exact parameter matching
    tgwt_model, baseline_model = create_matched_models(
        vocab_size=50257,
        max_seq_len=4096,
        d_model=args.d_model,
        n_heads=8,
        n_layers=args.n_layers,
        walk_length=args.walk_length,
    )

    # 3. Train TGWT
    tgwt_stats = run_training_loop(
        model=tgwt_model,
        model_name="TGWT (Ours)",
        train_loader=train_loader,
        val_loader=val_loader,
        max_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device,
    )

    # Evaluate NIAH on TGWT at N=2048 and N=4096
    print("\nEvaluating Needle-in-a-Haystack (NIAH) on TGWT...")
    tgwt_niah_2048 = evaluate_niah_retrieval(tgwt_model, seq_len=2048, device=device)
    tgwt_niah_4096 = evaluate_niah_retrieval(tgwt_model, seq_len=4096, device=device)
    print(f"TGWT NIAH @ N=2048: {tgwt_niah_2048['average_niah_acc']:.2f}% | @ N=4096: {tgwt_niah_4096['average_niah_acc']:.2f}%")

    # Save TGWT checkpoint
    os.makedirs("./checkpoints_tgwt", exist_ok=True)
    torch.save({
        "step": args.steps,
        "model_state_dict": tgwt_model.state_dict(),
        "val_loss": tgwt_stats["val_loss"],
        "val_ppl": tgwt_stats["val_ppl"],
    }, "./checkpoints_tgwt/best_model_tgwt.pt")

    # Free TGWT memory
    del tgwt_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Train Baseline Standard Transformer
    base_stats = run_training_loop(
        model=baseline_model,
        model_name="Baseline Transformer (Llama-style)",
        train_loader=train_loader,
        val_loader=val_loader,
        max_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device,
    )

    # Evaluate NIAH on Baseline at N=2048 and N=4096
    print("\nEvaluating Needle-in-a-Haystack (NIAH) on Baseline Transformer...")
    base_niah_2048 = evaluate_niah_retrieval(baseline_model, seq_len=2048, device=device)
    base_niah_4096 = evaluate_niah_retrieval(baseline_model, seq_len=4096, device=device)
    print(f"Baseline NIAH @ N=2048: {base_niah_2048['average_niah_acc']:.2f}% | @ N=4096: {base_niah_4096['average_niah_acc']:.2f}%")

    # Free Baseline memory
    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Final Comparison Table
    print("\n\n" + "="*85)
    print("                    ISO-COMPUTE 2,000-STEP FINAL COMPARISON                     ")
    print("="*85)
    print(f"{'Metric':<35} | {'Baseline Transformer':<22} | {'TGWT (Ours)':<22}")
    print("-" * 85)
    print(f"{'Validation Perplexity (PPL)':<35} | {base_stats['val_ppl']:<22.2f} | {tgwt_stats['val_ppl']:<22.2f}")
    print(f"{'Next-Token Top-1 Accuracy (%)':<35} | {base_stats['val_acc']:<22.2f}% | {tgwt_stats['val_acc']:<22.2f}%")
    print(f"{'Training Throughput (tok/s)':<35} | {base_stats['throughput']:<22,.1f} | {tgwt_stats['throughput']:<22,.1f}")
    print(f"{'Peak VRAM Usage (MB)':<35} | {base_stats['peak_vram']:<22.1f} | {tgwt_stats['peak_vram']:<22.1f}")
    print(f"{'NIAH Retrieval Acc @ N=2048':<35} | {base_niah_2048['average_niah_acc']:<22.2f}% | {tgwt_niah_2048['average_niah_acc']:<22.2f}%")
    print(f"{'NIAH Retrieval Acc @ N=4096':<35} | {base_niah_4096['average_niah_acc']:<22.2f}% | {tgwt_niah_4096['average_niah_acc']:<22.2f}%")
    print("="*85)


if __name__ == "__main__":
    main()

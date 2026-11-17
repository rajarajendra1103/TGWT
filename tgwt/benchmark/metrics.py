"""
The 5 Benchmark Metrics for TGWT:
1. Validation Loss / Perplexity (PPL)
2. Throughput (Tokens / Second)
3. Peak VRAM / Memory Usage (MB)
4. Theoretical FLOPs per Token
5. Prediction Accuracy (%)
"""

from typing import Dict, Optional, Tuple, Union
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_perplexity(loss: float) -> float:
    """
    Computes perplexity from cross-entropy loss.
    Clamps loss to prevent overflow.
    """
    try:
        return math.exp(min(loss, 20.0))
    except OverflowError:
        return float("inf")


def compute_flops_per_token(
    model_type: str,
    seq_len: int,
    d_model: int,
    n_layers: int,
    walk_length: int = 32,
    oversample_ratio: float = 1.5,
) -> float:
    """
    Computes theoretical FLOPs per token per layer.
    Standard Transformer:
      - QKV projections: 3 * 2 * D^2 = 6 D^2
      - Attention matrix & score: 2 * N * D (scales quadratically with N!)
      - Out projection: 2 * D^2
      - FFN (2 layers, 4D intermediate): 2 * 2 * 4 * D^2 = 16 D^2
      Total standard per token = (24 D^2 + 2 N D) * n_layers

    TGWT:
      - Walks process c * N tokens with fixed attention window W
      Total TGWT per token = (24 * c * D^2 + 2 * c * W * D + fusion_flops) * n_layers
      Notice: W is constant (e.g. 32), so FLOPs per token does NOT scale with N!
    """
    c = oversample_ratio
    W = walk_length
    D = d_model

    if model_type.lower() in ("standard", "baseline", "llama", "transformer"):
        # Flops per token per layer: 24 D^2 + 2 N D
        layer_flops = 24 * (D ** 2) + 2 * seq_len * D
    else:  # TGWT
        # Attention on M walks: 24 * c * D^2 + 2 * c * W * D
        # Fusion projection: ~4 * c * D^2
        layer_flops = 28 * c * (D ** 2) + 2 * c * W * D

    return float(layer_flops * n_layers)


def measure_throughput_and_vram(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    num_warmup: int = 3,
    num_steps: int = 10,
    backward: bool = True,
) -> Tuple[float, float]:
    """
    Measures:
    1. Throughput in tokens/sec
    2. Peak allocated VRAM in MB (or 0.0 if running on CPU)
    """
    device = input_ids.device
    is_cuda = device.type == "cuda"

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    model.eval() if not backward else model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) if backward else None

    # Use FP16 autocast on CUDA to match training speed and prevent OOM
    if is_cuda and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16)
    else:
        autocast_ctx = torch.no_grad() if not backward else torch.enable_grad()

    grad_context = torch.no_grad() if not backward else torch.enable_grad()

    with grad_context:
        # Warmup
        for _ in range(num_warmup):
            if optimizer:
                optimizer.zero_grad()
            with autocast_ctx:
                out = model(input_ids, labels=labels)
            if backward and out["loss"] is not None:
                out["loss"].backward()
                optimizer.step()

        if is_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        # Timed run
        start_time = time.perf_counter()
        total_tokens = 0

        for _ in range(num_steps):
            if optimizer:
                optimizer.zero_grad()
            with autocast_ctx:
                out = model(input_ids, labels=labels)
            if backward and out["loss"] is not None:
                out["loss"].backward()
                optimizer.step()
            total_tokens += input_ids.numel()

        if is_cuda:
            torch.cuda.synchronize(device)
            peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        else:
            peak_vram_mb = 0.0

    elapsed = time.perf_counter() - start_time
    throughput = total_tokens / max(elapsed, 1e-6)

    return throughput, peak_vram_mb


def compute_prediction_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Computes top-1 next-token prediction accuracy.
    """
    # Shift logits and targets
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    preds = shift_logits.argmax(dim=-1)
    mask = shift_labels != -100

    correct = (preds == shift_labels) & mask
    total = mask.sum().item()
    if total == 0:
        return 0.0
    return float(correct.sum().item() / total) * 100.0


def benchmark_model(
    model: nn.Module,
    model_name: str,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    n_layers: int,
    d_model: int,
    walk_length: int = 32,
) -> Dict[str, Union[float, str]]:
    """
    Evaluates all 5 metrics for a model.
    """
    B, N = input_ids.shape
    device = input_ids.device

    model.eval()
    if device.type == "cuda" and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        eval_autocast = torch.amp.autocast("cuda", dtype=torch.float16)
    else:
        eval_autocast = torch.no_grad()

    with torch.no_grad():
        with eval_autocast:
            out = model(input_ids, labels=labels)
        val_loss = float(out["loss"].item()) if out["loss"] is not None else 0.0
        ppl = compute_perplexity(val_loss)
        acc = compute_prediction_accuracy(out["logits"], labels)

    # Measure throughput and memory
    throughput, peak_vram = measure_throughput_and_vram(
        model=model,
        input_ids=input_ids,
        labels=labels,
        num_warmup=2,
        num_steps=5,
        backward=False,
    )

    # Compute theoretical FLOPs per token
    flops = compute_flops_per_token(
        model_type=model_name,
        seq_len=N,
        d_model=d_model,
        n_layers=n_layers,
        walk_length=walk_length,
    )

    return {
        "model": model_name,
        "val_loss": val_loss,
        "perplexity": ppl,
        "throughput_tokens_per_sec": throughput,
        "peak_vram_mb": peak_vram,
        "flops_per_token": flops,
        "accuracy_pct": acc,
        "params": sum(p.numel() for p in model.parameters()),
    }

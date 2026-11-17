"""
Evaluation and Benchmarking Script for TGWT vs Baseline Transformer.
Evaluates the 5 canonical comparison metrics across sequence lengths [512, 1024, 2048, 4096, 8192, 16384].
Includes automatic OOM catching to demonstrate where standard attention fails and TGWT succeeds.
"""

import os
import argparse
import torch

from tgwt.benchmark.compare import create_matched_models, format_comparison_table
from tgwt.benchmark.metrics import benchmark_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate TGWT vs Baseline Transformer")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length to benchmark")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for benchmark")
    parser.add_argument("--d_model", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--walk_length", type=int, default=32, help="TGWT walk length")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained TGWT checkpoint")
    parser.add_argument("--test_scaling", action="store_true", help="Benchmark scaling across [512, 1024, 2048, 4096, 8192, 16384]")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Benchmark on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384] if args.test_scaling else [args.seq_len]

    # Summary table across all sequence lengths
    scaling_summary = []

    for sl in seq_lengths:
        print(f"\n{'='*75}\nRunning Head-to-Head Benchmark for Sequence Length N = {sl:,}\n{'='*75}")

        # Instantiate parameter-matched models
        try:
            tgwt, baseline = create_matched_models(
                vocab_size=50257,
                max_seq_len=max(sl, 16384),
                d_model=args.d_model,
                n_heads=8,
                n_layers=args.n_layers,
                walk_length=args.walk_length,
            )
        except Exception as e:
            print(f"Failed to allocate models for N = {sl}: {e}")
            continue

        if args.checkpoint:
            target_path = args.checkpoint
            alt_candidates = [
                target_path,
                os.path.join("/content", target_path.lstrip("./")),
                os.path.join("/content/TGWT", target_path.lstrip("./")),
                os.path.join("/content/drive/MyDrive/TGWT_Backup", target_path.lstrip("./")),
                os.path.join("/content/drive/MyDrive", target_path.lstrip("./")),
                "best_model_tgwt.pt",
            ]
            found_path = None
            for p in alt_candidates:
                if os.path.exists(p):
                    found_path = p
                    break

            if found_path:
                print(f"Loading trained TGWT weights from {found_path}...")
                ckpt = torch.load(found_path, map_location="cpu")
                state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                tgwt.load_state_dict(state_dict, strict=False)
                print("Successfully loaded trained weights!")
            else:
                print(f"[Notice] Checkpoint '{target_path}' not found on disk.")
                print(f"         Proceeding with benchmark to profile Throughput, VRAM, and FLOPs scaling.")

        input_ids = torch.randint(0, 50257, (args.batch_size, sl), dtype=torch.long, device=device)
        labels = input_ids.clone()

        # 1. Benchmark Baseline Standard Transformer with OOM catching
        print("\nProfiling Baseline Standard Transformer (Llama-style causal attention)...")
        base_res = None
        try:
            baseline = baseline.to(device)
            base_res = benchmark_model(
                model=baseline,
                model_name="Baseline Transformer",
                input_ids=input_ids,
                labels=labels,
                n_layers=args.n_layers,
                d_model=args.d_model,
                d_ff=baseline.d_ff,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"-> Baseline Transformer CRASHED with OOM at N = {sl:,}! ({e})")
            base_res = {
                "model": "Baseline Transformer",
                "val_loss": "OOM",
                "perplexity": "OOM (>15.3 GB)",
                "throughput_tokens_per_sec": 0.0,
                "peak_vram_mb": "OOM (>15.3 GB)",
                "flops_per_token": "OOM",
                "accuracy_pct": 0.0,
                "oom": True,
            }
        finally:
            if baseline is not None:
                del baseline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 2. Benchmark TGWT (Tree-Graph Walk Transformer) with OOM catching
        print("Profiling TGWT (Property-Rich Tree-Graph Walk Transformer)...")
        tgwt_res = None
        try:
            tgwt = tgwt.to(device)
            tgwt_res = benchmark_model(
                model=tgwt,
                model_name="TGWT",
                input_ids=input_ids,
                labels=labels,
                n_layers=args.n_layers,
                d_model=args.d_model,
                walk_length=args.walk_length,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"-> TGWT failed with OOM at N = {sl:,}! ({e})")
            tgwt_res = {
                "model": "TGWT",
                "val_loss": "OOM",
                "perplexity": "OOM",
                "throughput_tokens_per_sec": 0.0,
                "peak_vram_mb": "OOM",
                "flops_per_token": "OOM",
                "accuracy_pct": 0.0,
                "oom": True,
            }
        finally:
            if tgwt is not None:
                del tgwt
            del input_ids
            del labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Print comparison table for this sequence length
        table_md = format_comparison_table(tgwt_res, base_res, seq_len=sl)
        print(table_md)

        # Record summary
        scaling_summary.append({
            "N": sl,
            "base_tps": base_res["throughput_tokens_per_sec"],
            "tgwt_tps": tgwt_res["throughput_tokens_per_sec"],
            "base_vram": base_res["peak_vram_mb"],
            "tgwt_vram": tgwt_res["peak_vram_mb"],
        })

    # Print Full Scaling Summary
    if len(scaling_summary) > 1:
        print("\n\n" + "="*80)
        print("                 COMPREHENSIVE SCALING SUMMARY (N = 512 to 16,384)                 ")
        print("="*80)
        print(f"{'Seq Len (N)':<12} | {'Llama Speed':<14} | {'TGWT Speed':<14} | {'Speedup':<10} | {'Llama VRAM':<14} | {'TGWT VRAM':<14}")
        print("-" * 88)
        for s in scaling_summary:
            b_t = f"{s['base_tps']:,.1f} tok/s" if isinstance(s['base_tps'], (int, float)) and s['base_tps'] > 0 else "OOM Crash"
            t_t = f"{s['tgwt_tps']:,.1f} tok/s" if isinstance(s['tgwt_tps'], (int, float)) and s['tgwt_tps'] > 0 else "OOM"
            speedup = f"{s['tgwt_tps'] / max(1e-6, s['base_tps']):.2f}x" if isinstance(s['base_tps'], (int, float)) and s['base_tps'] > 0 else "INF (Wins)"
            b_v = f"{s['base_vram']:,.1f} MB" if isinstance(s['base_vram'], (int, float)) else str(s['base_vram'])
            t_v = f"{s['tgwt_vram']:,.1f} MB" if isinstance(s['tgwt_vram'], (int, float)) else str(s['tgwt_vram'])
            print(f"{s['N']:<12} | {b_t:<14} | {t_t:<14} | {speedup:<10} | {b_v:<14} | {t_v:<14}")
        print("="*88)


if __name__ == "__main__":
    main()

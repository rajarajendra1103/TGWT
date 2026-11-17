"""
Evaluation and Benchmarking Script for TGWT vs Baseline Transformer.
Evaluates the 5 canonical comparison metrics across configurable sequence lengths.
"""

import argparse
import torch

from tgwt.benchmark.compare import create_matched_models, format_comparison_table
from tgwt.benchmark.metrics import benchmark_model
from tgwt.data.dataloader import get_dataloaders


def main():
    parser = argparse.ArgumentParser(description="Evaluate TGWT vs Baseline Transformer")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length to benchmark")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for benchmark")
    parser.add_argument("--d_model", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--walk_length", type=int, default=32, help="TGWT walk length")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained TGWT checkpoint to evaluate real PPL")
    parser.add_argument("--test_scaling", action="store_true", help="Benchmark scaling across [512, 1024, 2048, 4096]")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Benchmark on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    seq_lengths = [512, 1024, 2048, 4096] if args.test_scaling else [args.seq_len]

    for sl in seq_lengths:
        print(f"\n{'='*70}\nRunning Head-to-Head Benchmark for Sequence Length N = {sl}\n{'='*70}")

        # Instantiate parameter-matched models
        tgwt, baseline = create_matched_models(
            vocab_size=50257,
            max_seq_len=max(sl, 4096),
            d_model=args.d_model,
            n_heads=8,
            n_layers=args.n_layers,
            walk_length=args.walk_length,
        )

        if args.checkpoint:
            print(f"Loading trained TGWT weights from {args.checkpoint}...")
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            tgwt.load_state_dict(state_dict, strict=False)
            print("Successfully loaded trained weights!")

        tgwt = tgwt.to(device)
        baseline = baseline.to(device)

        # Generate sample batch
        input_ids = torch.randint(0, 50257, (args.batch_size, sl), dtype=torch.long, device=device)
        labels = input_ids.clone()

        # Benchmark Baseline
        print("\nProfiling Baseline Standard Transformer...")
        base_res = benchmark_model(
            model=baseline,
            model_name="Baseline Transformer",
            input_ids=input_ids,
            labels=labels,
            n_layers=args.n_layers,
            d_model=args.d_model,
        )

        # Benchmark TGWT
        print("Profiling TGWT (Property-Rich Tree-Graph Walk Transformer)...")
        tgwt_res = benchmark_model(
            model=tgwt,
            model_name="TGWT",
            input_ids=input_ids,
            labels=labels,
            n_layers=args.n_layers,
            d_model=args.d_model,
            walk_length=args.walk_length,
        )

        # Print comparison table
        table_md = format_comparison_table(tgwt_res, base_res, seq_len=sl)
        print(table_md)

        # Explicitly clean up memory to prevent OOM across scaling loop
        del tgwt
        del baseline
        del input_ids
        del labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

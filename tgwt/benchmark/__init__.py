"""
Benchmarking and fair evaluation harness for TGWT.
"""

from .metrics import (
    compute_perplexity,
    compute_flops_per_token,
    measure_throughput_and_vram,
    compute_prediction_accuracy,
    benchmark_model,
)
from .compare import create_matched_models, format_comparison_table
from .niah import evaluate_niah_retrieval

__all__ = [
    "compute_perplexity",
    "compute_flops_per_token",
    "measure_throughput_and_vram",
    "compute_prediction_accuracy",
    "benchmark_model",
    "create_matched_models",
    "format_comparison_table",
    "evaluate_niah_retrieval",
]

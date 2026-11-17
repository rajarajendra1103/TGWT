"""
Needle-in-a-Haystack (NIAH) and Multi-Hop Associative Recall Benchmark.
Tests long-context information retrieval across N = 2048 and N = 4096.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def evaluate_niah_retrieval(
    model: nn.Module,
    seq_len: int = 2048,
    needle_depths: Tuple[float, ...] = (0.1, 0.5, 0.9),
    vocab_size: int = 50257,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluates retrieval accuracy when a key-value fact is inserted at varying depths
    within distractor context of sequence length N.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    depth_accuracies = {}

    # Target key and value token IDs (arbitrary distinct tokens)
    key_tokens = torch.tensor([1243, 8921, 334], device=device, dtype=torch.long) # Key pattern
    val_tokens = torch.tensor([4242, 9999], device=device, dtype=torch.long)       # Target answer

    query_trigger = torch.tensor([1243, 8921, 334, 25], device=device, dtype=torch.long) # "Key:"

    distractor_vocab_start = 50
    distractor_vocab_end = 1000

    for depth in needle_depths:
        # Construct distractor sequence
        input_tokens = torch.randint(
            distractor_vocab_start, distractor_vocab_end, (1, seq_len), device=device, dtype=torch.long
        )

        # Insert needle at specified depth
        insert_idx = int(depth * (seq_len - 50))
        needle = torch.cat([key_tokens, val_tokens])
        input_tokens[0, insert_idx : insert_idx + len(needle)] = needle

        # Append query trigger at the end of the sequence
        query_len = len(query_trigger) + len(val_tokens)
        input_tokens[0, -query_len : -len(val_tokens)] = query_trigger
        input_tokens[0, -len(val_tokens) :] = val_tokens

        # Targets: -100 everywhere except the final target answer tokens
        labels = torch.full_like(input_tokens, -100)
        labels[0, -len(val_tokens) :] = val_tokens

        with torch.no_grad():
            out = model(input_tokens, labels=labels)
            logits = out["logits"]

            # Shift logits to check predictions for target tokens
            pred_tokens = logits[0, -len(val_tokens) - 1 : -1, :].argmax(dim=-1)
            correct = (pred_tokens == val_tokens).float().mean().item() * 100.0
            depth_accuracies[f"depth_{int(depth * 100)}%"] = correct

    avg_acc = sum(depth_accuracies.values()) / max(1, len(depth_accuracies))
    depth_accuracies["average_niah_acc"] = avg_acc
    return depth_accuracies

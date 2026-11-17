"""
Parameter-Matched Baseline Standard Transformer (Llama/GPT style).
Serves as the rigorous benchmark baseline for fair comparison with TGWT.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import StandardAttention, StandardFeedForward


class BaselineTransformerBlock(nn.Module):
    """
    Standard full-sequence causal Transformer block.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = StandardAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = StandardFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Full causal attention over entire sequence length N
        normed = self.norm1(x)
        x = x + self.dropout1(self.attn(normed, is_causal=True))
        normed2 = self.norm2(x)
        x = x + self.dropout2(self.ffn(normed2))
        return x


class BaselineTransformerLM(nn.Module):
    """
    Standard Causal Transformer Language Model.
    Computes full O(N^2) quadratic attention over the entire sequence.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 4096,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_layers = n_layers

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            BaselineTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def get_num_params(self, non_embedding: bool = False) -> int:
        if non_embedding:
            return sum(
                p.numel() for n, p in self.named_parameters() if "embedding" not in n
            )
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,               # [B, N]
        labels: Optional[torch.Tensor] = None, # [B, N]
    ) -> Dict[str, torch.Tensor]:
        B, N = input_ids.shape
        device = input_ids.device

        positions = torch.arange(0, N, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(positions)
        h = self.emb_dropout(tok_emb + pos_emb)

        for block in self.blocks:
            h = block(h)

        h = self.final_norm(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            flat_logits = shift_logits.view(-1, self.vocab_size)
            flat_labels = shift_labels.view(-1)

            chunk_size = 2048
            if flat_logits.shape[0] <= chunk_size:
                loss = F.cross_entropy(
                    flat_logits,
                    flat_labels,
                    ignore_index=-100,
                )
            else:
                total_loss = torch.tensor(0.0, device=device, dtype=flat_logits.dtype)
                total_tokens = torch.tensor(0, device=device, dtype=torch.long)
                for c_start in range(0, flat_logits.shape[0], chunk_size):
                    c_end = min(c_start + chunk_size, flat_logits.shape[0])
                    c_loss = F.cross_entropy(
                        flat_logits[c_start:c_end],
                        flat_labels[c_start:c_end],
                        ignore_index=-100,
                        reduction="sum",
                    )
                    valid = (flat_labels[c_start:c_end] != -100).sum()
                    total_loss = total_loss + c_loss
                    total_tokens = total_tokens + valid
                loss = total_loss / total_tokens.clamp(min=1)

        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": h,
        }

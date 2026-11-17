"""
Untouched Standard Transformer Block for TGWT.
Processes fixed-length serialized walks [B * M, W, D] with 100% standard attention.
Uses PyTorch F.scaled_dot_product_attention for native FlashAttention speed on GPU.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardAttention(nn.Module):
    """
    Standard multi-head attention module using native PyTorch SDPA.
    Zero custom CUDA code, fully hardware-accelerated.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,                      # [Batch, SeqLen, D]
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # PyTorch SDPA automatically chooses FlashAttention, Memory-Efficient Attention, or Math
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(attn_out)


class StandardFeedForward(nn.Module):
    """
    Standard Transformer Feed-Forward Network (GELU or SwiGLU).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.linear1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.act(self.linear1(x))))


class UntouchedTransformerBlock(nn.Module):
    """
    Standard Pre-LayerNorm Transformer Block.
    Processes walk sequences without any modification to standard Transformer mechanics.
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

    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Input: [B * M, W, D]
        Output: [B * M, W, D]
        """
        # Self-attention with residual
        normed = self.norm1(x)
        x = x + self.dropout1(self.attn(normed, is_causal=is_causal))

        # FFN with residual
        normed2 = self.norm2(x)
        x = x + self.dropout2(self.ffn(normed2))
        return x

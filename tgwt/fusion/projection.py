"""
Property Fusion Projection Layer for TGWT.
Solves New Problem 3 (Input Vector Crowding) and New Problem 4 (Positional Signal Confusion).
Fuses content vectors and metadata properties into size D_model.
"""

import torch
import torch.nn as nn


class PropertyFusionLayer(nn.Module):
    """
    Fuses node content vectors, node property embeddings, and edge property embeddings
    into a unified vector of size D_model for standard Transformer consumption.
    """

    def __init__(
        self,
        d_model: int,
        d_prop_node: int = 64,
        d_prop_edge: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        total_dim = d_model + d_prop_node + d_prop_edge

        # Direct linear projection + LayerNorm for maximum GPU throughput
        self.proj = nn.Linear(total_dim, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        content_vectors: torch.Tensor,   # [B, M, W, D_model]
        node_prop_vectors: torch.Tensor, # [B, M, W, D_prop_node]
        edge_prop_vectors: torch.Tensor, # [B, M, W, D_prop_edge]
    ) -> torch.Tensor:
        """
        Fuses representations and projects to [B, M, W, D_model].
        """
        concatenated = torch.cat(
            [content_vectors, node_prop_vectors, edge_prop_vectors], dim=-1
        )
        projected = self.proj(concatenated)
        fused = self.layer_norm(content_vectors + self.dropout(projected))
        return fused

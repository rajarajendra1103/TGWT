"""
Edge Properties tracking and embedding for TGWT.
"""

import torch
import torch.nn as nn


class EdgePropertyEmbedder(nn.Module):
    """
    Embeds edge properties:
    - edge_type: 0=self, 1=seq_next, 2=seq_prev, 3=tree_up, 4=tree_down, 5=semantic_bridge
    - jump_distance: signed log-scaled sequence distance
    - depth_delta: hierarchy level change
    """

    def __init__(
        self,
        d_prop: int = 64,
        num_edge_types: int = 6,
    ):
        super().__init__()
        self.d_prop = d_prop
        self.type_embedding = nn.Embedding(num_edge_types, d_prop // 2)
        self.continuous_proj = nn.Linear(2, d_prop - (d_prop // 2))
        self.out_norm = nn.LayerNorm(d_prop)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_attr: [E, 3] where [:, 0] is edge_type, [:, 1] is log_dist, [:, 2] is delta_lvl.

        Returns:
            edge_embeddings: [E, d_prop]
        """
        edge_types = edge_attr[:, 0].long()
        cont_features = edge_attr[:, 1:3]  # [E, 2]

        type_emb = self.type_embedding(edge_types)          # [E, d_prop // 2]
        cont_emb = self.continuous_proj(cont_features)      # [E, d_prop - (d_prop // 2)]

        emb = torch.cat([type_emb, cont_emb], dim=-1)       # [E, d_prop]
        return self.out_norm(emb)

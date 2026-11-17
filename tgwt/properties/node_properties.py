"""
Node Properties tracking and embedding for TGWT.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn


class NodePropertyTracker(nn.Module):
    """
    Manages and embeds dynamic node properties:
    - depth: hierarchical level in the tree
    - visit_count: normalized walker visitation count
    - surprise_score: prediction error / entropy proxy
    - node_type: categorical type (0=leaf, 1=phrase, 2=section, 3=root)
    """

    def __init__(
        self,
        d_prop: int = 64,
        num_node_types: int = 4,
    ):
        super().__init__()
        self.d_prop = d_prop
        self.num_node_types = num_node_types

        # Embedding for discrete node types
        self.type_embedding = nn.Embedding(num_node_types, d_prop // 2)

        # Projection for continuous properties: [depth, visit_count, surprise_score]
        self.continuous_proj = nn.Linear(3, d_prop - (d_prop // 2))
        self.out_norm = nn.LayerNorm(d_prop)

    def initialize_properties(
        self,
        tree_meta: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Initializes node properties for a new batch.
        """
        device = tree_meta["level_ids"].device
        K = int(tree_meta["total_nodes"].item())
        max_lvl = float(tree_meta["max_level"].item())
        max_lvl = max(1.0, max_lvl)

        level_ids = tree_meta["level_ids"]  # [K]
        norm_depth = level_ids.float() / max_lvl

        # Categorize node types: 0=leaf, 1=phrase, 2=section, 3=root
        node_types = torch.zeros(K, dtype=torch.long, device=device)
        is_leaf = tree_meta["is_leaf"]
        node_types[~is_leaf] = 1
        # Root is node with max_level
        node_types[level_ids == int(max_lvl)] = 3

        # Expand for batch
        # visit_count starts at 0.0
        # surprise_score starts at a uniform prior 0.5
        properties = {
            "depth": norm_depth.unsqueeze(0).expand(batch_size, -1),             # [B, K]
            "visit_count": torch.zeros((batch_size, K), dtype=torch.float32, device=device),  # [B, K]
            "surprise_score": torch.full((batch_size, K), 0.5, dtype=torch.float32, device=device), # [B, K]
            "node_type": node_types.unsqueeze(0).expand(batch_size, -1),         # [B, K]
        }
        return properties

    def forward(
        self,
        properties: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Embeds node properties into dense vectors of shape [B, K, d_prop].
        """
        depth = properties["depth"].unsqueeze(-1)           # [B, K, 1]
        visits = properties["visit_count"].unsqueeze(-1)    # [B, K, 1]
        surprise = properties["surprise_score"].unsqueeze(-1) # [B, K, 1]
        node_types = properties["node_type"]                # [B, K]

        continuous = torch.cat([depth, visits, surprise], dim=-1)  # [B, K, 3]
        cont_emb = self.continuous_proj(continuous)                # [B, K, d_prop // 2]
        type_emb = self.type_embedding(node_types)                 # [B, K, d_prop // 2]

        prop_vec = torch.cat([cont_emb, type_emb], dim=-1)         # [B, K, d_prop]
        return self.out_norm(prop_vec)

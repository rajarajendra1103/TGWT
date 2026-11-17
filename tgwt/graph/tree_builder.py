"""
Hierarchical Tree Builder for TGWT.
Builds multi-scale vertical tree nodes (leaf tokens -> phrase summaries -> paragraph/section -> root).
"""

from typing import Dict, List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TreeBuilder(nn.Module):
    """
    Constructs a multi-scale hierarchical tree over a sequence of leaf tokens.
    Branching factor k groups adjacent nodes bottom-up.
    """

    def __init__(
        self,
        d_model: int,
        branch_factor: int = 4,
        aggregation: str = "linear",  # "linear" or "mean"
    ):
        super().__init__()
        self.d_model = d_model
        self.branch_factor = branch_factor
        self.aggregation = aggregation

        if aggregation == "linear":
            self.pool_proj = nn.Linear(d_model * branch_factor, d_model)
            self.layer_norm = nn.LayerNorm(d_model)
        else:
            self.pool_proj = None
            self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self, leaf_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            leaf_embeddings: [B, N, D] leaf token representations.

        Returns:
            all_node_features: [B, K, D] where K is total nodes across all levels.
            tree_meta: Dictionary containing:
                - level_ids: [K] level of each node (0 = leaf, 1 = phrase, ..., L = root)
                - parent_indices: [K] parent node index for each node (-1 for root)
                - original_positions: [K] original center sequence position [0..N-1]
                - is_leaf: [K] boolean mask
                - total_nodes: int (K)
        """
        B, N, D = leaf_embeddings.shape
        device = leaf_embeddings.device

        levels_features: List[torch.Tensor] = [leaf_embeddings]
        level_ids_list: List[torch.Tensor] = [
            torch.zeros(N, dtype=torch.long, device=device)
        ]
        orig_pos_list: List[torch.Tensor] = [
            torch.arange(N, dtype=torch.float32, device=device)
        ]

        # Track parent connections
        # For each level, we will compute which node in the next level it connects to
        parent_ptrs: List[torch.Tensor] = []

        curr_features = leaf_embeddings
        curr_len = N
        curr_level = 0
        running_node_offset = N

        while curr_len > 1:
            next_len = math.ceil(curr_len / self.branch_factor)
            pad_needed = (next_len * self.branch_factor) - curr_len

            if pad_needed > 0:
                padded = F.pad(curr_features, (0, 0, 0, pad_needed), mode="constant", value=0.0)
            else:
                padded = curr_features

            # Shape: [B, next_len, branch_factor, D]
            grouped = padded.view(B, next_len, self.branch_factor, D)

            if self.aggregation == "linear":
                # Flatten [B, next_len, branch_factor * D] and project
                flat_grouped = grouped.view(B, next_len, self.branch_factor * D)
                next_features = self.layer_norm(self.pool_proj(flat_grouped))
            else:
                # Mean pool
                next_features = self.layer_norm(grouped.mean(dim=2))

            curr_level += 1
            levels_features.append(next_features)
            level_ids_list.append(
                torch.full((next_len,), curr_level, dtype=torch.long, device=device)
            )

            # Center position of pooled group
            prev_positions = orig_pos_list[-1]
            if pad_needed > 0:
                padded_pos = F.pad(prev_positions.unsqueeze(0), (0, pad_needed), mode="replicate").squeeze(0)
            else:
                padded_pos = prev_positions
            grouped_pos = padded_pos.view(next_len, self.branch_factor)
            next_positions = grouped_pos.mean(dim=1)
            orig_pos_list.append(next_positions)

            # Mapping from current level nodes to next level nodes
            # Each node i in current level maps to running_node_offset + (i // branch_factor)
            node_parents = running_node_offset + (
                torch.arange(curr_len, device=device) // self.branch_factor
            )
            parent_ptrs.append(node_parents)

            running_node_offset += next_len
            curr_features = next_features
            curr_len = next_len

        # The root node (top level) has no parent (-1)
        parent_ptrs.append(torch.tensor([-1], dtype=torch.long, device=device))

        # Concatenate all node features across all levels: [B, K, D]
        all_node_features = torch.cat(levels_features, dim=1)
        all_level_ids = torch.cat(level_ids_list, dim=0)
        all_parent_ptrs = torch.cat(parent_ptrs, dim=0)
        all_orig_pos = torch.cat(orig_pos_list, dim=0)

        K = all_node_features.shape[1]
        is_leaf = torch.zeros(K, dtype=torch.bool, device=device)
        is_leaf[:N] = True

        tree_meta = {
            "level_ids": all_level_ids,
            "parent_indices": all_parent_ptrs,
            "original_positions": all_orig_pos,
            "is_leaf": is_leaf,
            "total_nodes": torch.tensor(K, device=device),
            "num_leaves": torch.tensor(N, device=device),
            "max_level": torch.tensor(curr_level, device=device),
        }

        return all_node_features, tree_meta

    def pool_features(
        self, leaf_embeddings: torch.Tensor, tree_meta: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Fast feature pooling path using pre-computed tree topology.
        """
        B, N, D = leaf_embeddings.shape
        levels_features = [leaf_embeddings]
        curr_features = leaf_embeddings
        curr_len = N

        while curr_len > 1:
            next_len = math.ceil(curr_len / self.branch_factor)
            pad_needed = (next_len * self.branch_factor) - curr_len

            if pad_needed > 0:
                padded = F.pad(curr_features, (0, 0, 0, pad_needed), mode="constant", value=0.0)
            else:
                padded = curr_features

            grouped = padded.view(B, next_len, self.branch_factor, D)

            if self.aggregation == "linear":
                flat_grouped = grouped.view(B, next_len, self.branch_factor * D)
                next_features = self.layer_norm(self.pool_proj(flat_grouped))
            else:
                next_features = self.layer_norm(grouped.mean(dim=2))

            levels_features.append(next_features)
            curr_features = next_features
            curr_len = next_len

        return torch.cat(levels_features, dim=1)

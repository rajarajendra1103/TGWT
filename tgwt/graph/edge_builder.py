"""
Edge Builder for TGWT.
Builds vertical tree edges (up/down), horizontal sequential edges (prev/next),
and cross-branch semantic skip edges.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn


# Edge Type Constants
EDGE_TYPE_SELF = 0
EDGE_TYPE_SEQ_NEXT = 1
EDGE_TYPE_SEQ_PREV = 2
EDGE_TYPE_TREE_UP = 3
EDGE_TYPE_TREE_DOWN = 4
EDGE_TYPE_SEMANTIC_BRIDGE = 5


class EdgeBuilder(nn.Module):
    """
    Constructs edge connectivity and static edge properties between nodes in the Tree-Graph.
    """

    def __init__(self, semantic_skip_strides: Tuple[int, ...] = (8, 32)):
        super().__init__()
        self.semantic_skip_strides = semantic_skip_strides

    def forward(
        self, tree_meta: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tree_meta: Dictionary from TreeBuilder containing:
                - level_ids: [K]
                - parent_indices: [K]
                - original_positions: [K]
                - is_leaf: [K]
                - num_leaves: int
                - total_nodes: int

        Returns:
            edge_index: [2, E] source and target node indices.
            edge_attr: [E, 3] raw edge features:
                [:, 0] = edge_type_id (0..5)
                [:, 1] = signed jump distance (log-scaled)
                [:, 2] = delta tree level
        """
        device = tree_meta["level_ids"].device
        K = int(tree_meta["total_nodes"].item())
        N = int(tree_meta["num_leaves"].item())
        parent_indices = tree_meta["parent_indices"]
        positions = tree_meta["original_positions"]
        level_ids = tree_meta["level_ids"]
        causal_positions = tree_meta.get("causal_positions", positions.long())

        src_list = []
        dst_list = []
        type_list = []

        # 1. Self loops for all nodes (always causal: u -> u)
        all_nodes = torch.arange(K, device=device)
        src_list.append(all_nodes)
        dst_list.append(all_nodes)
        type_list.append(
            torch.full((K,), EDGE_TYPE_SELF, dtype=torch.long, device=device)
        )

        # 2. Horizontal sequential edges: strictly backward (i -> i - 1)
        if N > 1:
            leaf_curr = torch.arange(1, N, device=device)
            leaf_prev = torch.arange(0, N - 1, device=device)

            # Prev: i -> i - 1
            src_list.append(leaf_curr)
            dst_list.append(leaf_prev)
            type_list.append(
                torch.full((N - 1,), EDGE_TYPE_SEQ_PREV, dtype=torch.long, device=device)
            )

        # 3. Vertical Tree Edges (Only when causal: causal_pos[dst] <= causal_pos[src])
        has_parent_mask = parent_indices >= 0
        child_nodes = torch.arange(K, device=device)[has_parent_mask]
        parent_nodes = parent_indices[has_parent_mask]

        if len(child_nodes) > 0:
            # Tree Up: child -> parent (allowed only if parent causal_pos <= child causal_pos)
            valid_up = causal_positions[parent_nodes] <= causal_positions[child_nodes]
            if valid_up.any():
                src_list.append(child_nodes[valid_up])
                dst_list.append(parent_nodes[valid_up])
                type_list.append(
                    torch.full(
                        (int(valid_up.sum().item()),),
                        EDGE_TYPE_TREE_UP,
                        dtype=torch.long,
                        device=device,
                    )
                )

            # Tree Down: parent -> child (allowed only if child causal_pos <= parent causal_pos)
            valid_down = causal_positions[child_nodes] <= causal_positions[parent_nodes]
            if valid_down.any():
                src_list.append(parent_nodes[valid_down])
                dst_list.append(child_nodes[valid_down])
                type_list.append(
                    torch.full(
                        (int(valid_down.sum().item()),),
                        EDGE_TYPE_TREE_DOWN,
                        dtype=torch.long,
                        device=device,
                    )
                )

        # 4. Semantic Skip Bridges: strictly backward (i -> i - stride)
        for stride in self.semantic_skip_strides:
            if N > stride:
                skip_curr = torch.arange(stride, N, device=device)
                skip_past = torch.arange(0, N - stride, device=device)

                # Backward skip: i -> i - stride
                src_list.append(skip_curr)
                dst_list.append(skip_past)
                type_list.append(
                    torch.full(
                        (len(skip_curr),),
                        EDGE_TYPE_SEMANTIC_BRIDGE,
                        dtype=torch.long,
                        device=device,
                    )
                )

        src_all = torch.cat(src_list, dim=0)
        dst_all = torch.cat(dst_list, dim=0)
        edge_types = torch.cat(type_list, dim=0)

        # Enforce strict causal filter: destination must never be in the future of source
        valid_causal_mask = causal_positions[dst_all] <= causal_positions[src_all]
        src_all = src_all[valid_causal_mask]
        dst_all = dst_all[valid_causal_mask]
        edge_types = edge_types[valid_causal_mask]

        edge_index = torch.stack([src_all, dst_all], dim=0)  # [2, E]

        # Compute continuous jump distance and level deltas
        pos_src = positions[src_all]
        pos_dst = positions[dst_all]
        delta_pos = pos_dst - pos_src
        # Signed log-scale distance: sign(delta) * log1p(|delta|)
        sign_pos = torch.sign(delta_pos)
        log_dist = sign_pos * torch.log1p(torch.abs(delta_pos))

        lvl_src = level_ids[src_all].float()
        lvl_dst = level_ids[dst_all].float()
        delta_lvl = lvl_dst - lvl_src

        # Shape [E, 3]
        edge_attr = torch.stack(
            [edge_types.float(), log_dist, delta_lvl], dim=-1
        )

        return edge_index, edge_attr

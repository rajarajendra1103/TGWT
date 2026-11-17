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

        src_list = []
        dst_list = []
        type_list = []

        # 1. Self loops for all nodes
        all_nodes = torch.arange(K, device=device)
        src_list.append(all_nodes)
        dst_list.append(all_nodes)
        type_list.append(
            torch.full((K,), EDGE_TYPE_SELF, dtype=torch.long, device=device)
        )

        # 2. Horizontal sequential edges between adjacent leaf tokens
        if N > 1:
            leaf_src = torch.arange(N - 1, device=device)
            leaf_dst = torch.arange(1, N, device=device)

            # Next: i -> i + 1
            src_list.append(leaf_src)
            dst_list.append(leaf_dst)
            type_list.append(
                torch.full((N - 1,), EDGE_TYPE_SEQ_NEXT, dtype=torch.long, device=device)
            )

            # Prev: i + 1 -> i
            src_list.append(leaf_dst)
            dst_list.append(leaf_src)
            type_list.append(
                torch.full((N - 1,), EDGE_TYPE_SEQ_PREV, dtype=torch.long, device=device)
            )

        # 3. Vertical Tree Edges (Child -> Parent and Parent -> Child)
        has_parent_mask = parent_indices >= 0
        child_nodes = torch.arange(K, device=device)[has_parent_mask]
        parent_nodes = parent_indices[has_parent_mask]

        if len(child_nodes) > 0:
            # Tree Up: child -> parent
            src_list.append(child_nodes)
            dst_list.append(parent_nodes)
            type_list.append(
                torch.full(
                    (len(child_nodes),),
                    EDGE_TYPE_TREE_UP,
                    dtype=torch.long,
                    device=device,
                )
            )

            # Tree Down: parent -> child
            src_list.append(parent_nodes)
            dst_list.append(child_nodes)
            type_list.append(
                torch.full(
                    (len(child_nodes),),
                    EDGE_TYPE_TREE_DOWN,
                    dtype=torch.long,
                    device=device,
                )
            )

        # 4. Semantic / Dilated Skip Bridges across leaf branches
        for stride in self.semantic_skip_strides:
            if N > stride:
                skip_src = torch.arange(N - stride, device=device)
                skip_dst = torch.arange(stride, N, device=device)

                # Forward skip
                src_list.append(skip_src)
                dst_list.append(skip_dst)
                type_list.append(
                    torch.full(
                        (len(skip_src),),
                        EDGE_TYPE_SEMANTIC_BRIDGE,
                        dtype=torch.long,
                        device=device,
                    )
                )

                # Backward skip
                src_list.append(skip_dst)
                dst_list.append(skip_src)
                type_list.append(
                    torch.full(
                        (len(skip_src),),
                        EDGE_TYPE_SEMANTIC_BRIDGE,
                        dtype=torch.long,
                        device=device,
                    )
                )

        src_all = torch.cat(src_list, dim=0)
        dst_all = torch.cat(dst_list, dim=0)
        edge_types = torch.cat(type_list, dim=0)

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

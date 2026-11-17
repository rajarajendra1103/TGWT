"""
TreeGraph data structure and builder module for TGWT.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from .tree_builder import TreeBuilder
from .edge_builder import EdgeBuilder


class TreeGraphData:
    """
    Container for the constructed Tree-Graph state in a single batch.
    """

    def __init__(
        self,
        node_features: torch.Tensor,       # [B, K, D]
        edge_index: torch.Tensor,          # [2, E]
        edge_attr: torch.Tensor,           # [E, 3]
        tree_meta: Dict[str, torch.Tensor],
        adjacency_list: Optional[List[List[int]]] = None,
    ):
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.tree_meta = tree_meta
        self.adjacency_list = adjacency_list

    @property
    def total_nodes(self) -> int:
        return self.node_features.shape[1]

    @property
    def num_leaves(self) -> int:
        return int(self.tree_meta["num_leaves"].item())

    @property
    def device(self) -> torch.device:
        return self.node_features.device


class TreeGraphConstructor(nn.Module):
    """
    Complete module to construct a property-rich Tree-Graph from leaf token embeddings.
    """

    def __init__(
        self,
        d_model: int,
        branch_factor: int = 4,
        aggregation: str = "linear",
        semantic_skip_strides: Tuple[int, ...] = (8, 32),
    ):
        super().__init__()
        self.tree_builder = TreeBuilder(
            d_model=d_model,
            branch_factor=branch_factor,
            aggregation=aggregation,
        )
        self.edge_builder = EdgeBuilder(
            semantic_skip_strides=semantic_skip_strides,
        )

    def forward(self, leaf_embeddings: torch.Tensor) -> TreeGraphData:
        """
        Args:
            leaf_embeddings: [B, N, D]

        Returns:
            TreeGraphData container with node features, edges, and metadata.
        """
        all_node_features, tree_meta = self.tree_builder(leaf_embeddings)
        edge_index, edge_attr = self.edge_builder(tree_meta)

        return TreeGraphData(
            node_features=all_node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            tree_meta=tree_meta,
        )

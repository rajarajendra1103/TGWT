"""
Unit tests for Tree-Graph construction.
"""

import unittest
import torch
from tgwt.graph.tree_builder import TreeBuilder
from tgwt.graph.edge_builder import EdgeBuilder
from tgwt.graph.tree_graph import TreeGraphConstructor


class TestTreeGraph(unittest.TestCase):

    def test_tree_builder(self):
        B, N, D = 2, 64, 128
        leaves = torch.randn(B, N, D)
        builder = TreeBuilder(d_model=D, branch_factor=4)

        all_nodes, meta = builder(leaves)
        K = meta["total_nodes"].item()

        # For N=64 with branch_factor 4: Level 0=64, Level 1=16, Level 2=4, Level 3=1 (Total = 85)
        self.assertEqual(K, 85)
        self.assertEqual(all_nodes.shape, (B, K, D))
        self.assertEqual(meta["level_ids"].shape[0], K)
        self.assertEqual(meta["parent_indices"].shape[0], K)
        self.assertEqual(meta["is_leaf"].sum().item(), N)

    def test_edge_builder(self):
        B, N, D = 2, 64, 128
        leaves = torch.randn(B, N, D)
        builder = TreeBuilder(d_model=D, branch_factor=4)
        all_nodes, meta = builder(leaves)

        edge_builder = EdgeBuilder(semantic_skip_strides=(8, 16))
        edge_index, edge_attr = edge_builder(meta)

        self.assertEqual(edge_index.dim(), 2)
        self.assertEqual(edge_index.shape[0], 2)
        self.assertEqual(edge_attr.shape[0], edge_index.shape[1])
        self.assertEqual(edge_attr.shape[1], 3)

    def test_tree_graph_constructor(self):
        B, N, D = 2, 32, 64
        leaves = torch.randn(B, N, D)
        constructor = TreeGraphConstructor(d_model=D, branch_factor=4)
        tg_data = constructor(leaves)

        self.assertEqual(tg_data.num_leaves, N)
        self.assertTrue(tg_data.total_nodes > N)
        self.assertEqual(tg_data.edge_index.shape[0], 2)


if __name__ == "__main__":
    unittest.main()

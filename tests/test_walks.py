"""
Unit tests for deterministic walk routing and serialization.
"""

import unittest
import torch
from tgwt.graph.tree_graph import TreeGraphConstructor
from tgwt.properties.node_properties import NodePropertyTracker
from tgwt.walks.serializer import WalkSerializer


class TestWalkSerialization(unittest.TestCase):

    def test_walk_serializer(self):
        B, N, D = 2, 64, 64
        W = 16
        leaves = torch.randn(B, N, D)
        constructor = TreeGraphConstructor(d_model=D, branch_factor=4)
        tg_data = constructor(leaves)

        tracker = NodePropertyTracker(d_prop=32)
        properties = tracker.initialize_properties(tg_data.tree_meta, batch_size=B)

        serializer = WalkSerializer(walk_length=W, oversample_ratio=1.5)
        walk_nodes, walk_eattrs = serializer(tg_data, properties)

        M = walk_nodes.shape[1]
        self.assertEqual(walk_nodes.shape, (B, M, W))
        self.assertEqual(walk_eattrs.shape, (B, M, W, 3))

        # Check all node indices are in valid range [0..K-1]
        K = tg_data.total_nodes
        self.assertTrue((walk_nodes >= 0).all())
        self.assertTrue((walk_nodes < K).all())


if __name__ == "__main__":
    unittest.main()

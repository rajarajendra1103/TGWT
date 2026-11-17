"""
Scatter Pooling Layer for TGWT.
Aggregates representations from multiple walks visiting the same graph nodes
and applies gated updates to maintain stability across layers.
"""

from typing import Tuple
import torch
import torch.nn as nn

from ..properties.gated_updater import GatedNodeUpdater


class ScatterPoolingLayer(nn.Module):
    """
    Vectorized Scatter Pooling Layer.
    Merges updated walk vectors [B, M, W, D] back into graph nodes [B, K, D]
    via native PyTorch C++ scatter_reduce followed by gated updates.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.gated_updater = GatedNodeUpdater(d_model=d_model)

    def forward(
        self,
        current_node_features: torch.Tensor, # [B, K, D]
        walk_outputs: torch.Tensor,          # [B, M, W, D]
        walk_node_indices: torch.Tensor,     # [B, M, W]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            updated_node_features: [B, K, D]
            active_mask: [B, K] boolean tensor of visited nodes
        """
        B, M, W, D = walk_outputs.shape
        K = current_node_features.shape[1]
        device = walk_outputs.device
        dtype = walk_outputs.dtype

        flat_indices = walk_node_indices.view(B, M * W)
        flat_outputs = walk_outputs.view(B, M * W, D)

        # 1. Native PyTorch C++ scatter_reduce (O(1) memory allocation)
        expanded_indices = flat_indices.unsqueeze(-1).expand(-1, -1, D)
        mean_pooled = torch.zeros(B, K, D, device=device, dtype=dtype)
        mean_pooled = mean_pooled.scatter_reduce(
            1, expanded_indices, flat_outputs, reduce="mean", include_self=False
        )

        # 2. Fast active mask computation
        visited_counts = torch.zeros(B, K, device=device, dtype=torch.float32)
        visited_counts.scatter_add_(1, flat_indices, torch.ones_like(flat_indices, dtype=torch.float32))
        active_mask = visited_counts > 0

        # 3. Gated blending update
        updated_features = self.gated_updater(
            current_features=current_node_features,
            pooled_updates=mean_pooled,
            active_mask=active_mask,
        )

        return updated_features, active_mask

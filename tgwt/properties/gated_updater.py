"""
Gated Property & Node Representation Updater for TGWT.
Solves New Problem 5 (Stale or Noisy Property Drift) using an LSTM-style
gating mechanism to update node states and dynamic properties.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class GatedNodeUpdater(nn.Module):
    """
    Applies LSTM-style input and forget gates to update node representations
    from pooled walk outputs, preventing catastrophic drift.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Gating projections
        self.gate_proj = nn.Linear(2 * d_model, d_model)
        self.candidate_proj = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        current_features: torch.Tensor,   # [B, K, D]
        pooled_updates: torch.Tensor,     # [B, K, D]
        active_mask: torch.Tensor,        # [B, K] or [K] nodes that received visits
    ) -> torch.Tensor:
        """
        Updates node features with gated blending.
        """
        # Concatenate current and pooled features
        combined = torch.cat([current_features, pooled_updates], dim=-1) # [B, K, 2D]

        # Gate in [0, 1]
        gate = torch.sigmoid(self.gate_proj(combined))                    # [B, K, D]
        # Candidate state in [-1, 1]
        candidate = torch.tanh(self.candidate_proj(combined))             # [B, K, D]

        # Gated blend
        updated = (1.0 - gate) * current_features + gate * candidate
        updated = self.norm(updated)

        # Only apply update to nodes that were actually visited
        if active_mask.dim() == 1:
            mask = active_mask.unsqueeze(0).unsqueeze(-1) # [1, K, 1]
        else:
            mask = active_mask.unsqueeze(-1)             # [B, K, 1]

        final_features = torch.where(mask, updated, current_features)
        return final_features

    def update_dynamic_properties(
        self,
        properties: Dict[str, torch.Tensor],
        walk_node_indices: torch.Tensor, # [B, M, W]
        step_surprises: Optional[torch.Tensor] = None, # [B, M, W]
        momentum: float = 0.2,
    ) -> Dict[str, torch.Tensor]:
        """
        Updates visit_counts and surprise_scores based on executed walks.
        """
        B, M, W = walk_node_indices.shape
        K = properties["visit_count"].shape[1]
        device = walk_node_indices.device

        # Compute visit count delta for each node
        flat_indices = walk_node_indices.view(B, -1) # [B, M*W]
        ones = torch.ones_like(flat_indices, dtype=torch.float32)

        visit_deltas = torch.zeros((B, K), dtype=torch.float32, device=device)
        visit_deltas.scatter_add_(1, flat_indices, ones)

        # Normalize visits by walk length
        norm_deltas = visit_deltas / float(W)
        new_visits = properties["visit_count"] + norm_deltas

        # Update surprise scores if provided
        new_surprise = properties["surprise_score"].clone()
        if step_surprises is not None:
            flat_surprises = step_surprises.view(B, -1) # [B, M*W]
            sum_surprise = torch.zeros((B, K), dtype=torch.float32, device=device)
            sum_surprise.scatter_add_(1, flat_indices, flat_surprises)

            # Average surprise per node
            counts = visit_deltas.clamp(min=1.0)
            avg_surprise = sum_surprise / counts

            visited_mask = visit_deltas > 0
            # Exponential moving update
            blended = (1.0 - momentum) * properties["surprise_score"] + momentum * avg_surprise
            new_surprise = torch.where(visited_mask, blended, properties["surprise_score"])

        updated_properties = dict(properties)
        updated_properties["visit_count"] = new_visits
        updated_properties["surprise_score"] = new_surprise
        return updated_properties

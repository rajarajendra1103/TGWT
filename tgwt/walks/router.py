"""
Deterministic Property-Guided Walk Router for TGWT.
Solves New Problem 2 (Non-Differentiable Walk Routing) using deterministic
coverage rules based on node properties (lowest visit count + highest surprise)
without requiring reinforcement learning.
"""

from typing import Dict, List, Optional, Tuple
import torch


class PropertyGuidedRouter:
    """
    Selects the next hop in a graph walk using deterministic property scoring.
    Score = alpha * surprise - beta * visits + gamma * edge_type_priority
    """

    def __init__(
        self,
        alpha_surprise: float = 1.0,
        beta_visits: float = 1.5,
        gamma_forward: float = 0.5,
    ):
        self.alpha = alpha_surprise
        self.beta = beta_visits
        self.gamma = gamma_forward

    def select_next_node(
        self,
        curr_node: int,
        neighbors: List[int],
        edge_types: List[int],
        edge_indices: List[int],
        visit_counts: torch.Tensor,     # [K]
        surprise_scores: torch.Tensor,  # [K]
        prev_node: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Selects next neighbor deterministically.

        Returns:
            next_node: chosen neighbor node index
            chosen_edge_idx: corresponding edge index
        """
        if not neighbors:
            return curr_node, -1

        best_score = -1e9
        best_candidate = neighbors[0]
        best_edge_idx = edge_indices[0]

        for nbr, etype, eidx in zip(neighbors, edge_types, edge_indices):
            # Penalize immediate 1-step back-tracking if alternatives exist
            backtrack_penalty = 0.0
            if prev_node is not None and nbr == prev_node and len(neighbors) > 1:
                backtrack_penalty = 2.0

            surp = float(surprise_scores[nbr].item())
            vis = float(visit_counts[nbr].item())

            # Forward bias gives preference to advancing through sequence
            forward_bonus = 0.0
            if etype in (1, 5):  # SEQ_NEXT or SEMANTIC_BRIDGE forward
                forward_bonus = self.gamma

            score = (
                self.alpha * surp
                - self.beta * vis
                + forward_bonus
                - backtrack_penalty
            )

            # Deterministic tie breaking: higher score, then lower node index
            if score > best_score:
                best_score = score
                best_candidate = nbr
                best_edge_idx = eidx

        return best_candidate, best_edge_idx

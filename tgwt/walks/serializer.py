"""
GPU-Vectorized Walk Serializer for TGWT.
Converts the 2D/hierarchical Tree-Graph into a fixed-length 1D walk batch
of shape [B, M, W] with linear complexity O(N).
Fully vectorized with pure PyTorch tensor operations on CUDA and cached static adjacency.
"""

from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn

from ..graph.tree_graph import TreeGraphData


class WalkSerializer(nn.Module):
    """
    High-Performance GPU-Vectorized Walk Serializer.
    Executes walk routing in parallel across all batch elements and walks simultaneously.
    Caches graph adjacency per (K, device) to eliminate dynamic allocation overhead.
    """

    def __init__(
        self,
        walk_length: int = 32,
        oversample_ratio: float = 1.5,
        alpha_surprise: float = 1.0,
        beta_visits: float = 1.5,
        gamma_forward: float = 0.5,
        max_degree: int = 16,
    ):
        super().__init__()
        self.walk_length = walk_length
        self.oversample_ratio = oversample_ratio
        self.alpha = alpha_surprise
        self.beta = beta_visits
        self.gamma = gamma_forward
        self.max_degree = max_degree

        # Cache for static GPU adjacency tables
        self._adj_cache: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _get_gpu_adjacency(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        total_nodes: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_key = (total_nodes, str(device))
        if cache_key in self._adj_cache:
            return self._adj_cache[cache_key]

        K = total_nodes
        max_deg = self.max_degree

        src = edge_index[0]
        dst = edge_index[1]
        etypes = edge_attr[:, 0].long()
        E = edge_index.shape[1]
        eindices = torch.arange(E, device=device)

        sorted_src, perm = torch.sort(src)
        sorted_dst = dst[perm]
        sorted_etypes = etypes[perm]
        sorted_eindices = eindices[perm]

        boundaries = torch.cat([
            torch.tensor([0], device=device),
            torch.nonzero(sorted_src[1:] != sorted_src[:-1], as_tuple=False).squeeze(-1) + 1
        ])
        run_starts = torch.repeat_interleave(
            boundaries,
            torch.diff(torch.cat([boundaries, torch.tensor([E], device=device)]))
        )
        local_slot = torch.arange(E, device=device) - run_starts

        valid_edge_mask = local_slot < max_deg
        filt_src = sorted_src[valid_edge_mask]
        filt_slot = local_slot[valid_edge_mask]
        filt_dst = sorted_dst[valid_edge_mask]
        filt_etypes = sorted_etypes[valid_edge_mask]
        filt_eindices = sorted_eindices[valid_edge_mask]

        adj_nbrs = torch.zeros((K, max_deg), dtype=torch.long, device=device)
        adj_etypes = torch.zeros((K, max_deg), dtype=torch.long, device=device)
        adj_eindices = torch.zeros((K, max_deg), dtype=torch.long, device=device)
        adj_mask = torch.zeros((K, max_deg), dtype=torch.bool, device=device)

        flat_dest_idx = filt_src * max_deg + filt_slot
        adj_nbrs.view(-1).scatter_(0, flat_dest_idx, filt_dst)
        adj_etypes.view(-1).scatter_(0, flat_dest_idx, filt_etypes)
        adj_eindices.view(-1).scatter_(0, flat_dest_idx, filt_eindices)
        adj_mask.view(-1).scatter_(0, flat_dest_idx, True)

        self._adj_cache[cache_key] = (adj_nbrs, adj_etypes, adj_eindices, adj_mask)
        return self._adj_cache[cache_key]

    def forward(
        self,
        tree_graph: TreeGraphData,
        properties: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, K, _ = tree_graph.node_features.shape
        N = tree_graph.num_leaves
        W = self.walk_length
        device = tree_graph.device

        # Linear scaling M
        M = max(1, math.ceil((N * self.oversample_ratio) / W))
        D_deg = self.max_degree

        adj_nbrs, adj_etypes, adj_eindices, adj_mask = self._get_gpu_adjacency(
            tree_graph.edge_index,
            tree_graph.edge_attr,
            K,
            device,
        )

        # Seed nodes evenly spaced across leaves
        step_stride = max(1, N // M)
        seed_indices = torch.arange(M, device=device) * step_stride % N
        curr_nodes = seed_indices.unsqueeze(0).expand(B, -1).clone()
        prev_nodes = torch.full_like(curr_nodes, -1)

        walk_nodes = torch.zeros((B, M, W), dtype=torch.long, device=device)
        walk_eindices = torch.zeros((B, M, W), dtype=torch.long, device=device)
        walk_nodes[:, :, 0] = curr_nodes

        b_visits = properties["visit_count"].clone()
        b_surprise = properties["surprise_score"]

        # Pre-allocate deltas
        seed_delta = torch.full_like(curr_nodes, 0.5, dtype=torch.float32)
        step_delta = torch.full_like(curr_nodes, 0.2, dtype=torch.float32)
        b_visits.scatter_add_(1, curr_nodes, seed_delta)

        flat_shape = (B, M * D_deg)

        for step in range(1, W):
            cand_nbrs = adj_nbrs[curr_nodes]         # [B, M, D_deg]
            cand_etypes = adj_etypes[curr_nodes]     # [B, M, D_deg]
            cand_eindices = adj_eindices[curr_nodes] # [B, M, D_deg]
            cand_mask = adj_mask[curr_nodes]         # [B, M, D_deg]

            # Fast 2D gather without allocating 3D [B, M, K] tensors
            flat_cand_nbrs = cand_nbrs.view(flat_shape)
            nbr_visits = torch.gather(b_visits, 1, flat_cand_nbrs).view(B, M, D_deg)
            nbr_surprise = torch.gather(b_surprise, 1, flat_cand_nbrs).view(B, M, D_deg)

            forward_bonus = torch.where(
                (cand_etypes == 1) | (cand_etypes == 5),
                self.gamma,
                0.0
            )
            backtrack_penalty = torch.where(
                cand_nbrs == prev_nodes.unsqueeze(-1),
                2.0,
                0.0
            )

            scores = (
                self.alpha * nbr_surprise
                - self.beta * nbr_visits
                + forward_bonus
                - backtrack_penalty
            )
            scores = scores.masked_fill(~cand_mask, -1e9)

            best_choice = scores.argmax(dim=-1, keepdim=True)
            nxt_nodes = torch.gather(cand_nbrs, 2, best_choice).squeeze(-1)
            nxt_eindices = torch.gather(cand_eindices, 2, best_choice).squeeze(-1)

            prev_nodes = curr_nodes
            curr_nodes = nxt_nodes

            walk_nodes[:, :, step] = curr_nodes
            walk_eindices[:, :, step] = nxt_eindices
            b_visits.scatter_add_(1, curr_nodes, step_delta)

        walk_edge_attrs = tree_graph.edge_attr[walk_eindices]
        return walk_nodes, walk_edge_attrs

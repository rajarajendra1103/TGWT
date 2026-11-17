"""
GPU-Vectorized Walk Serializer for TGWT.
Converts the 2D/hierarchical Tree-Graph into a fixed-length 1D walk batch
of shape [B, M, W] with linear complexity O(N).
Fully vectorized with pure PyTorch tensor operations on CUDA (Zero Python loops over walkers).
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

    def _build_gpu_adjacency(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        total_nodes: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Builds fixed-width neighbor tensors on GPU for parallel gather operations:
        - adj_nbrs: [K, max_deg]
        - adj_etypes: [K, max_deg]
        - adj_eindices: [K, max_deg]
        - adj_mask: [K, max_deg] (bool)
        """
        K = total_nodes
        max_deg = self.max_degree

        src = edge_index[0]
        dst = edge_index[1]
        etypes = edge_attr[:, 0].long()
        E = edge_index.shape[1]
        eindices = torch.arange(E, device=device)

        # Count degree per source node
        degrees = torch.zeros(K, dtype=torch.long, device=device)
        degrees.scatter_add_(0, src, torch.ones_like(src))

        # Assign a local neighbor slot [0..deg-1] for each edge
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

        # Keep only edges that fit in max_deg
        valid_edge_mask = local_slot < max_deg
        filt_src = sorted_src[valid_edge_mask]
        filt_slot = local_slot[valid_edge_mask]
        filt_dst = sorted_dst[valid_edge_mask]
        filt_etypes = sorted_etypes[valid_edge_mask]
        filt_eindices = sorted_eindices[valid_edge_mask]

        # Allocate fixed table
        adj_nbrs = torch.zeros((K, max_deg), dtype=torch.long, device=device)
        adj_etypes = torch.zeros((K, max_deg), dtype=torch.long, device=device)
        adj_eindices = torch.zeros((K, max_deg), dtype=torch.long, device=device)
        adj_mask = torch.zeros((K, max_deg), dtype=torch.bool, device=device)

        flat_dest_idx = filt_src * max_deg + filt_slot
        adj_nbrs.view(-1).scatter_(0, flat_dest_idx, filt_dst)
        adj_etypes.view(-1).scatter_(0, flat_dest_idx, filt_etypes)
        adj_eindices.view(-1).scatter_(0, flat_dest_idx, filt_eindices)
        adj_mask.view(-1).scatter_(0, flat_dest_idx, True)

        return adj_nbrs, adj_etypes, adj_eindices, adj_mask

    def forward(
        self,
        tree_graph: TreeGraphData,
        properties: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized walk generation on GPU.

        Returns:
            walk_nodes: [B, M, W] node indices along walks
            walk_edge_attrs: [B, M, W, 3] edge attributes of each transition
        """
        B, K, _ = tree_graph.node_features.shape
        N = tree_graph.num_leaves
        W = self.walk_length
        device = tree_graph.device

        # Number of walks M scaled linearly with N
        M = max(1, math.ceil((N * self.oversample_ratio) / W))

        # Build GPU neighbor table
        adj_nbrs, adj_etypes, adj_eindices, adj_mask = self._build_gpu_adjacency(
            tree_graph.edge_index,
            tree_graph.edge_attr,
            K,
            device,
        )
        D_deg = self.max_degree

        # Seed nodes evenly spaced across leaves
        step_stride = max(1, N // M)
        seed_indices = torch.arange(M, device=device) * step_stride % N # [M]
        curr_nodes = seed_indices.unsqueeze(0).expand(B, -1).clone()     # [B, M]
        prev_nodes = torch.full_like(curr_nodes, -1)                     # [B, M]

        # Preallocate result tensors
        walk_nodes = torch.zeros((B, M, W), dtype=torch.long, device=device)
        walk_eindices = torch.zeros((B, M, W), dtype=torch.long, device=device)

        walk_nodes[:, :, 0] = curr_nodes

        # Dynamic visits state clone
        b_visits = properties["visit_count"].clone()       # [B, K]
        b_surprise = properties["surprise_score"]          # [B, K]

        # Initial seed visit bump
        b_visits.scatter_add_(1, curr_nodes, torch.full_like(curr_nodes, 0.5, dtype=torch.float32))

        # Vectorized step loop along W (Only 31 GPU iterations!)
        for step in range(1, W):
            # 1. Lookup candidates for all B x M walkers in one operation
            cand_nbrs = adj_nbrs[curr_nodes]         # [B, M, D_deg]
            cand_etypes = adj_etypes[curr_nodes]     # [B, M, D_deg]
            cand_eindices = adj_eindices[curr_nodes] # [B, M, D_deg]
            cand_mask = adj_mask[curr_nodes]         # [B, M, D_deg]

            # 2. Gather visits and surprise scores for all candidates
            exp_visits = b_visits.unsqueeze(1).expand(-1, M, -1)
            exp_surprise = b_surprise.unsqueeze(1).expand(-1, M, -1)

            nbr_visits = torch.gather(exp_visits, 2, cand_nbrs)     # [B, M, D_deg]
            nbr_surprise = torch.gather(exp_surprise, 2, cand_nbrs) # [B, M, D_deg]

            # 3. Vectorized deterministic scoring
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

            # 4. Argmax over candidates for all B x M walkers in parallel
            best_choice = scores.argmax(dim=-1, keepdim=True) # [B, M, 1]
            nxt_nodes = torch.gather(cand_nbrs, 2, best_choice).squeeze(-1)       # [B, M]
            nxt_eindices = torch.gather(cand_eindices, 2, best_choice).squeeze(-1) # [B, M]

            # 5. Update state
            prev_nodes = curr_nodes
            curr_nodes = nxt_nodes

            walk_nodes[:, :, step] = curr_nodes
            walk_eindices[:, :, step] = nxt_eindices

            # Update visits
            b_visits.scatter_add_(1, curr_nodes, torch.full_like(curr_nodes, 0.2, dtype=torch.float32))

        # Vectorized gather of all edge attributes in one GPU indexing operation
        walk_edge_attrs = tree_graph.edge_attr[walk_eindices] # [B, M, W, 3]

        return walk_nodes, walk_edge_attrs

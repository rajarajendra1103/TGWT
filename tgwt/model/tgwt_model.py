"""
Complete TGWT (Tree-Graph Walk Transformer) Model.
Implements the 5-stage pipeline with fused QKV, GPU-vectorized walk serialization,
and high-throughput linear complexity O(N).
"""

from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graph.tree_graph import TreeGraphConstructor, TreeGraphData
from ..properties.node_properties import NodePropertyTracker
from ..properties.edge_properties import EdgePropertyEmbedder
from ..walks.serializer import WalkSerializer
from ..fusion.projection import PropertyFusionLayer
from .transformer import UntouchedTransformerBlock
from .pooling import ScatterPoolingLayer


class TGWTBlock(nn.Module):
    """
    Modular TGWT processing block.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_prop_node: int = 64,
        d_prop_edge: int = 64,
        walk_length: int = 32,
        oversample_ratio: float = 1.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.transformer = UntouchedTransformerBlock(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )

    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        return self.transformer(x, is_causal=is_causal)


class TGWTForCausalLM(nn.Module):
    """
    Complete Property-Rich Tree-Graph Walk Transformer for Language Modeling.
    Runs with linear O(N) memory and compute complexity.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 16384,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        d_prop_node: int = 64,
        d_prop_edge: int = 64,
        walk_length: int = 32,
        branch_factor: int = 4,
        oversample_ratio: float = 1.5,
        dropout: float = 0.1,
        tie_weights: bool = True,
        n_macro_blocks: int = 1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.walk_length = walk_length
        self.oversample_ratio = oversample_ratio
        self.n_macro_blocks = n_macro_blocks

        # Stage 1: Base Token Embedding and Positional Encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # Stage 2: Tree-Graph Constructor with topology caching
        self.graph_constructor = TreeGraphConstructor(
            d_model=d_model,
            branch_factor=branch_factor,
        )

        # Stage 3: Property Trackers & Embedders
        self.node_prop_tracker = NodePropertyTracker(d_prop=d_prop_node)
        self.edge_prop_embedder = EdgePropertyEmbedder(d_prop=d_prop_edge)

        # Stage 4: High-Performance GPU Walk Serializer
        self.serializer = WalkSerializer(
            walk_length=walk_length,
            oversample_ratio=oversample_ratio,
        )

        # Stage 5: Property Fusion Projection Layer
        self.fusion = PropertyFusionLayer(
            d_model=d_model,
            d_prop_node=d_prop_node,
            d_prop_edge=d_prop_edge,
            dropout=dropout,
        )

        # Stage 5: Stack of Untouched Transformer Blocks (All standard PyTorch layers)
        self.transformer_layers = nn.ModuleList([
            UntouchedTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Stage 5: Vectorized Scatter-Pooling & Gated Update
        self.scatter_pooling = ScatterPoolingLayer(d_model=d_model)

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_num_params(self, non_embedding: bool = False) -> int:
        if non_embedding:
            return sum(
                p.numel() for n, p in self.named_parameters() if "embedding" not in n
            )
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,               # [B, N]
        labels: Optional[torch.Tensor] = None, # [B, N]
    ) -> Dict[str, torch.Tensor]:
        B, N = input_ids.shape
        device = input_ids.device
        D = self.d_model
        W = self.walk_length

        # Stage 1: Base Token Embedding + Positional Encoding
        positions = torch.arange(0, N, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(positions)
        leaf_features = self.emb_dropout(tok_emb + pos_emb) # [B, N, D]

        # Stage 2: Property-Rich Tree-Graph Construction
        tree_graph = self.graph_constructor(leaf_features)
        K = tree_graph.total_nodes

        # Stage 3: Dynamic Node & Edge Properties
        properties = self.node_prop_tracker.initialize_properties(
            tree_meta=tree_graph.tree_meta,
            batch_size=B,
        )
        node_prop_vecs = self.node_prop_tracker(properties) # [B, K, d_prop_node]

        # Stage 4: GPU-Vectorized Walk Serialization
        walk_nodes, walk_eattrs = self.serializer(tree_graph, properties)
        M = walk_nodes.shape[1]

        # Gather node content along walks
        expanded_walk_nodes = walk_nodes.view(B, M * W).unsqueeze(-1).expand(-1, -1, D)
        walk_content = torch.gather(
            tree_graph.node_features, dim=1, index=expanded_walk_nodes
        ).view(B, M, W, D)

        # Gather node properties along walks
        d_pnode = node_prop_vecs.shape[-1]
        expanded_pnodes = walk_nodes.view(B, M * W).unsqueeze(-1).expand(-1, -1, d_pnode)
        walk_node_props = torch.gather(
            node_prop_vecs, dim=1, index=expanded_pnodes
        ).view(B, M, W, d_pnode)

        # Embed edge properties for transitions along walks
        flat_eattrs = walk_eattrs.view(B * M * W, 3)
        walk_edge_props = self.edge_prop_embedder(flat_eattrs).view(B, M, W, -1)

        # Stage 5: Property Fusion Projection Layer
        fused_walks = self.fusion(
            content_vectors=walk_content,
            node_prop_vectors=walk_node_props,
            edge_prop_vectors=walk_edge_props,
        ) # [B, M, W, D]

        # Stage 5: Pass through stack of untouched Transformer blocks
        flat_walk_input = fused_walks.view(B * M, W, D)
        for layer in self.transformer_layers:
            flat_walk_input = layer(flat_walk_input, is_causal=False)

        walk_output = flat_walk_input.view(B, M, W, D)

        # Stage 5: Scatter-Pooling back to graph nodes & Gated Updates
        updated_features, active_mask = self.scatter_pooling(
            current_node_features=tree_graph.node_features,
            walk_outputs=walk_output,
            walk_node_indices=walk_nodes,
        )

        # Update dynamic properties
        updated_properties = self.scatter_pooling.gated_updater.update_dynamic_properties(
            properties=properties,
            walk_node_indices=walk_nodes,
        )

        # Extract updated leaf representations: [B, N, D]
        leaf_out = updated_features[:, :N, :]
        leaf_out = self.final_norm(leaf_out)

        # Language modeling head
        logits = self.lm_head(leaf_out) # [B, N, Vocab]

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            flat_logits = shift_logits.view(-1, self.vocab_size)
            flat_labels = shift_labels.view(-1)

            # Chunked cross entropy to eliminate OOM on long sequences
            chunk_size = 2048
            if flat_logits.shape[0] <= chunk_size:
                loss = F.cross_entropy(
                    flat_logits,
                    flat_labels,
                    ignore_index=-100,
                )
            else:
                total_loss = torch.tensor(0.0, device=device, dtype=flat_logits.dtype)
                total_tokens = torch.tensor(0, device=device, dtype=torch.long)
                for c_start in range(0, flat_logits.shape[0], chunk_size):
                    c_end = min(c_start + chunk_size, flat_logits.shape[0])
                    c_loss = F.cross_entropy(
                        flat_logits[c_start:c_end],
                        flat_labels[c_start:c_end],
                        ignore_index=-100,
                        reduction="sum",
                    )
                    valid = (flat_labels[c_start:c_end] != -100).sum()
                    total_loss = total_loss + c_loss
                    total_tokens = total_tokens + valid
                loss = total_loss / total_tokens.clamp(min=1)

        return {
            "logits": logits,
            "loss": loss,
            "leaf_hidden_states": leaf_out,
        }

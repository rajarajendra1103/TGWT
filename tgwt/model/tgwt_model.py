"""
Complete TGWT (Tree-Graph Walk Transformer) Model.
Implements the full 5-stage pipeline with stacked blocks and causal language modeling.
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
    A single TGWT processing block.
    1. Reads current node features & properties.
    2. Serializes graph into fixed-length walks.
    3. Fuses content and properties.
    4. Passes walks through untouched standard Transformer.
    5. Scatter-pools updates back to graph nodes with gated updating.
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
        self.d_model = d_model
        self.walk_length = walk_length

        self.node_prop_tracker = NodePropertyTracker(d_prop=d_prop_node)
        self.edge_prop_embedder = EdgePropertyEmbedder(d_prop=d_prop_edge)
        self.serializer = WalkSerializer(
            walk_length=walk_length,
            oversample_ratio=oversample_ratio,
        )
        self.fusion = PropertyFusionLayer(
            d_model=d_model,
            d_prop_node=d_prop_node,
            d_prop_edge=d_prop_edge,
            dropout=dropout,
        )
        self.transformer = UntouchedTransformerBlock(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.scatter_pooling = ScatterPoolingLayer(d_model=d_model)

    def forward(
        self,
        tree_graph: TreeGraphData,
        properties: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            tree_graph: TreeGraphData containing node_features [B, K, D], edge_index, edge_attr
            properties: Dict with 'depth', 'visit_count', 'surprise_score', 'node_type'

        Returns:
            updated_node_features: [B, K, D]
            updated_properties: Dict of updated properties
        """
        B, K, D = tree_graph.node_features.shape

        # 1. Embed node and edge properties
        node_prop_vecs = self.node_prop_tracker(properties)         # [B, K, d_prop_node]
        edge_prop_vecs = self.edge_prop_embedder(tree_graph.edge_attr) # [E, d_prop_edge]

        # 2. Serialize graph into walks: [B, M, W] and [B, M, W, 3]
        walk_nodes, walk_eattrs = self.serializer(tree_graph, properties)
        M = walk_nodes.shape[1]
        W = self.walk_length

        # 3. Gather content and property vectors for all walk steps
        # Content vectors: [B, M, W, D]
        expanded_walk_nodes = walk_nodes.view(B, M * W).unsqueeze(-1).expand(-1, -1, D)
        walk_content = torch.gather(
            tree_graph.node_features, dim=1, index=expanded_walk_nodes
        ).view(B, M, W, D)

        # Node properties: [B, M, W, d_prop_node]
        d_pnode = node_prop_vecs.shape[-1]
        expanded_pnodes = (
            walk_nodes.view(B, M * W).unsqueeze(-1).expand(-1, -1, d_pnode)
        )
        walk_node_props = torch.gather(
            node_prop_vecs, dim=1, index=expanded_pnodes
        ).view(B, M, W, d_pnode)

        # Edge properties: embed edge attributes for each step
        flat_eattrs = walk_eattrs.view(B * M * W, 3)
        walk_edge_props = self.edge_prop_embedder(flat_eattrs).view(
            B, M, W, -1
        )

        # 4. Property Fusion
        fused_walks = self.fusion(
            content_vectors=walk_content,
            node_prop_vectors=walk_node_props,
            edge_prop_vectors=walk_edge_props,
        ) # [B, M, W, D]

        # 5. Untouched Transformer execution
        # Flatten batch and walks: [B * M, W, D]
        flat_walk_input = fused_walks.view(B * M, W, D)
        flat_walk_output = self.transformer(flat_walk_input, is_causal=False)
        walk_output = flat_walk_output.view(B, M, W, D)

        # 6. Scatter-Pooling back to graph nodes & Gated Updates
        updated_features, active_mask = self.scatter_pooling(
            current_node_features=tree_graph.node_features,
            walk_outputs=walk_output,
            walk_node_indices=walk_nodes,
        )

        # 7. Update dynamic graph properties (visits and surprise score)
        updated_properties = self.scatter_pooling.gated_updater.update_dynamic_properties(
            properties=properties,
            walk_node_indices=walk_nodes,
        )

        return updated_features, updated_properties


class TGWTForCausalLM(nn.Module):
    """
    Complete Property-Rich Tree-Graph Walk Transformer for Language Modeling.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 4096,
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
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_layers = n_layers

        # Stage 1: Base Token Embedding and Positional Encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # Stage 2: Tree-Graph Constructor
        self.graph_constructor = TreeGraphConstructor(
            d_model=d_model,
            branch_factor=branch_factor,
        )

        # Stage 3-5: Stacked TGWT Blocks
        self.blocks = nn.ModuleList([
            TGWTBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_prop_node=d_prop_node,
                d_prop_edge=d_prop_edge,
                walk_length=walk_length,
                oversample_ratio=oversample_ratio,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

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

        # Stage 1: Embeddings + 1D Positional Encodings
        positions = torch.arange(0, N, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(positions)
        leaf_features = self.emb_dropout(tok_emb + pos_emb) # [B, N, D]

        # Stage 2: Tree-Graph Construction
        tree_graph = self.graph_constructor(leaf_features)

        # Stage 3: Dynamic Property Initialization
        # Node property tracker from first block can initialize properties
        properties = self.blocks[0].node_prop_tracker.initialize_properties(
            tree_meta=tree_graph.tree_meta,
            batch_size=B,
        )

        # Stages 4 & 5: Pass through stacked TGWT blocks
        curr_features = tree_graph.node_features
        for block in self.blocks:
            tree_graph.node_features = curr_features
            curr_features, properties = block(tree_graph, properties)

        # Extract leaf representations: [B, N, D]
        leaf_out = curr_features[:, :N, :]
        leaf_out = self.final_norm(leaf_out)

        # Language modeling head
        logits = self.lm_head(leaf_out) # [B, N, Vocab]

        loss = None
        if labels is not None:
            # Shift tokens for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            flat_logits = shift_logits.view(-1, self.vocab_size)
            flat_labels = shift_labels.view(-1)

            # Chunked cross entropy to prevent OOM on long context lengths
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

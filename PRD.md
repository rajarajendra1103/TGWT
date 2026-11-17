# Product Requirements Document (PRD)
## Property-Rich Tree-Graph Walk Transformer (TGWT)

---

### 1. Executive Summary & Vision
The **Property-Rich Tree-Graph Walk Transformer (TGWT)** is a hybrid AI architecture designed to process long sequential text with strictly **linear memory efficiency \(O(N)\)** while keeping the core Transformer compute engine **100% untouched**. 

Traditional Transformers incur an \(O(N^2)\) quadratic computational and memory complexity, bottlenecking long-context scaling. Graph Transformers attempt to incorporate structural relations by tokenizing both nodes and edges, inflating sequence lengths by up to \(3\times\) and attention compute by \(9\times\) (\(O((3N)^2)\)). Pure trees suffer from "branch blindness" where cross-branch tokens cannot interact locally, and pure graphs suffer from slow long-range message travel across high diameters.

TGWT solves all four bottlenecks by introducing a lightweight **Property-Rich Tree-Graph front-end** coupled with a **Deterministic Property-Guided Walk Serializer**. Sequential data is converted into a multi-scale hierarchical tree with horizontal semantic/local graph shortcuts. Fixed-length 1D walks (e.g., $W=32$ or $W=64$) are extracted from this graph using deterministic coverage rules prioritizing unexplored regions (lowest visit count) and high-information nodes (highest surprise score). These walks are projected through a property-fusion layer into a standard batch of shape `[M, W, D]` and fed into a completely standard, off-the-shelf Transformer block (leveraging FlashAttention and native PyTorch SDPA without custom CUDA kernels). Finally, an LSTM-style gated scatter pooling layer recombines representations and updates graph states across stacked blocks.

---

### 2. Architecture Specification (The 5 Stages)

```
Raw Sequence (Tokens x_1, ..., x_N)
               │
               ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Base Embedding & 1D Positional Encoding      │
│  - Embedding Lookup: R^{N x D_in}                      │
│  - Standard 1D Positional Encoding (locks word order)  │
└────────────────────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: Property-Rich Tree-Graph Construction         │
│  - Vertical Tree Nodes: Hierarchical multi-level chunks│
│    (Leaf tokens -> Phrase nodes -> Paragraph -> Root)  │
│  - Horizontal Graph Edges: 1-hop sequential neighbors  │
│    + Semantic long-range skip edges                    │
└────────────────────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: Dynamic Node & Edge Properties                │
│  - Node State: depth, visit_count, surprise_score, vec │
│  - Edge State: edge_type, hop_distance, delta_pos      │
└────────────────────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────┐
│ Stage 4: Property-Guided Walk Serialization            │
│  - Deterministic router: min(visit_count) + max(surp)  │
│  - Batch of M fixed-length 1D walks: [M, W, Node_IDs]  │
│  - Guarantees 100% node coverage without RL            │
└────────────────────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────┐
│ Stage 5: Fusion Projection & Untouched Transformer     │
│  - Property Fusion: [Content || Properties] -> R^D     │
│  - Untouched Transformer (MHA + MLP, standard PyTorch) │
│  - Scatter-Pooling & Gated LSTM Updates to Graph Nodes │
└────────────────────────────────────────────────────────┘
               │
               ▼
Language Modeling Head / Downstream Prediction (CrossEntropy)
```

#### Stage 1: Base Token, Embedding, and Positional Encoding
- **Input**: Token IDs sequence $\mathbf{x} = (x_1, \dots, x_N) \in \mathbb{N}^N$.
- **Embedding**: $\mathbf{E} = \text{Embedding}(\mathbf{x}) \in \mathbb{R}^{N \times D_{\text{model}}}$.
- **1D Positional Encoding**: $\mathbf{H}_0 = \mathbf{E} + \mathbf{PE}_{1\text{D}}$, where $\mathbf{PE}_{1\text{D}}(i)$ encodes original sequential position $i \in \{1, \dots, N\}$.
- **Purpose**: Preserves exact chronological order prior to any hierarchical grouping.

#### Stage 2: Property-Rich Tree-Graph Construction
- **Vertical Tree Nodes**:
  - Level 0 (Leaves): The $N$ base token vectors $\mathbf{H}_0$.
  - Level $k \ge 1$: Adjacent tokens or lower-level clusters are pooled bottom-up using chunked pooling (branching factor $k_{\text{tree}}$, e.g. $k_{\text{tree}}=4$), culminating in a single root node at level $L_{\text{tree}} = \lceil \log_{k_{\text{tree}}} N \rceil$.
  - Total tree nodes $\approx N \frac{k_{\text{tree}}}{k_{\text{tree}} - 1} \approx 1.33N$ for $k_{\text{tree}}=4$.
- **Horizontal Graph Edges**:
  - Sequential edges: Bidirectional edge $(i, i+1)$ between sequential tokens to retain local sentence context.
  - Semantic skip edges: Distance-bounded or lexical/attention-guided bridges connecting distant correlated tokens across separate tree branches in $O(1)$ hops.

#### Stage 3: Dynamic Node and Edge Properties
- **Node Property Vector** $\mathbf{p}_v \in \mathbb{R}^{D_v}$:
  - `depth`: Normalized tree depth $l(v) / L_{\text{max}}$.
  - `visit_count`: Number of times node $v$ has been traversed by walkers (normalized).
  - `surprise_score`: Running prediction entropy or loss signal $\sigma(v) \in [0, 1]$.
  - `node_type`: One-hot or learned embedding (Leaf, Phrase, Root).
- **Edge Property Vector** $\mathbf{q}_{uv} \in \mathbb{R}^{D_e}$:
  - `edge_type`: Categorical embedding (Parent-to-Child, Child-to-Parent, Sequential-Neighbor, Semantic-Bridge).
  - `jump_distance`: Signed original token offset $|pos(u) - pos(v)|$, log-scaled.
  - `hop_distance`: Graph structural distance.

#### Stage 4: Property-Guided Walk Serialization
- **Walk Generation**: Generates $M = \lceil \frac{N \cdot \text{oversample}}{W} \rceil$ walks, each of length $W$ (e.g., $W=32$).
- **Routing Rule**:
  At current node $u$, select next node $v^* \in \mathcal{N}(u)$ via deterministic score:
  $$\text{Score}(u, v) = \alpha \cdot \text{Surprise}(v) - \beta \cdot \text{VisitCount}(v) + \gamma \cdot \text{EdgeWeight}(u, v)$$
  Ties are broken deterministically by node index. This guarantees uniform graph coverage, prioritizes surprise, and eliminates non-differentiable RL routing.
- **Batched Serialization**: Produces node sequence tensor $\mathbf{W}_{\text{nodes}} \in \mathbb{N}^{M \times W}$ and edge transition tensor $\mathbf{W}_{\text{edges}} \in \mathbb{N}^{M \times W \times D_e}$.

#### Stage 5: Untouched Transformer and Scatter Pooling
- **Property Fusion Projection**:
  $$\mathbf{Z}_{\text{in}} = \text{LayerNorm}(\mathbf{W}_{\text{content}} \mathbf{W}_c + \mathbf{W}_{\text{prop}} \mathbf{W}_p) \in \mathbb{R}^{M \times W \times D_{\text{model}}}$$
- **Untouched Transformer**:
  $$\mathbf{Z}_{\text{out}} = \text{TransformerBlock}(\mathbf{Z}_{\text{in}}) \in \mathbb{R}^{M \times W \times D_{\text{model}}}$$
  Uses standard PyTorch `nn.TransformerEncoderLayer` or custom standard causal attention with `F.scaled_dot_product_attention`. No custom CUDA required.
- **Scatter Pooling & Gated Node Updates**:
  Multiple walks visit node $v$. We scatter-aggregate representations:
  $$\tilde{\mathbf{h}}_v = \text{ScatterMean}(\{\mathbf{Z}_{\text{out}}[m, w] : \mathbf{W}_{\text{nodes}}[m, w] = v\})$$
  $$\mathbf{g}_v = \text{sigmoid}(\mathbf{W}_g [\mathbf{h}_v; \tilde{\mathbf{h}}_v] + \mathbf{b}_g) \quad (\text{Update Gate})$$
  $$\mathbf{h}_v^{(t+1)} = (1 - \mathbf{g}_v) \odot \mathbf{h}_v^{(t)} + \mathbf{g}_v \odot \tanh(\mathbf{W}_u \tilde{\mathbf{h}}_v + \mathbf{b}_u)$$
  Node properties `visit_count` and `surprise_score` are updated deterministically.

---

### 3. Solutions to Key Engineering Challenges

| Problem | Root Cause | TGWT Engineering Fix |
|---|---|---|
| **Cross-Walk Tunnel Vision** | Independent batch dimensions prevent inter-walk communication in single block | Shared node property whiteboard + scatter-pooling update + multi-block stacking. |
| **Non-Differentiable Walk Routing** | Discrete step selection on graphs breaks gradients | Deterministic coverage heuristic ($\min \text{visits} + \max \text{surprise}$) with complete coverage. |
| **Input Vector Crowding** | Merging content, depth, and hops into size $D$ causes feature interference | Dedicated Property-Fusion Linear Projection + LayerNorm with residual balancing. |
| **Positional Signal Confusion** | Large jumps in graph walks mimic adjacent words to standard attention | Explicit hop-distance and edge-type embeddings appended to step properties. |
| **Stale / Noisy Property Drift** | Multi-pass property updates accumulate divergence | LSTM-style input/forget gating with relevance threshold. |
| **Text Generation Latency** | Graph re-generation per token is costly | Chunked sliding-window generation: immediate raw window + periodic graph folding. |

---

### 4. Target Hardware & Deployment Environment
- **Primary Training & Testing Target**: Google Colab Free Tier with NVIDIA Tesla T4 GPU (16 GB VRAM).
- **Precision**: Mixed Precision (`torch.cuda.amp.autocast(dtype=torch.float16)`).
- **Target Context Length**: Long sequences ($N = 2048$ to $8192$) where standard Transformers hit $O(N^2)$ VRAM limits, demonstrating TGWT's $O(N \cdot W)$ linear footprint.
- **Dataset**: WikiText-103 (103M tokens, standard benchmark for long-range language modeling).

---

### 5. Benchmark Metrics & Fair Comparison Protocol
To validate architectural superiority, TGWT is benchmarked directly against a parameter-matched standard Llama/GPT-style Transformer:

1. **Validation Loss & Perplexity (PPL)**: Lower is better. Measured on WikiText-103 test/validation splits at identical token budgets.
2. **Throughput (Tokens / Second)**: Higher is better. Wall-clock training and inference throughput measured on NVIDIA T4.
3. **Peak VRAM (MB)**: Lower is better. Monitored via `torch.cuda.max_memory_allocated()` across context lengths $N \in [512, 1024, 2048, 4096, 8192]$.
4. **FLOPs per Token**: Lower is better. Theoretical floating point operations computed via analytical profiling / FLOP counters.
5. **Zero-Shot Downstream Accuracy**: Task evaluation across standard language modeling benchmarks.

**Fairness Guardrails**:
- Parameter parity within $\pm 2\%$.
- Identical BPE / WordPiece tokenizer and vocabulary size ($V = 50,257$ or $32,000$).
- Identical optimizer (AdamW, $\beta_1=0.9, \beta_2=0.95$), cosine learning rate decay with warm-up, and random seed.

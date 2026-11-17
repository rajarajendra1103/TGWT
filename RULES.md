# Architectural Rules & Standards (RULES.md)
## Property-Rich Tree-Graph Walk Transformer (TGWT)

These rules govern the implementation, tensor conventions, memory bounds, training stability, and Colab T4 hardware compliance for the TGWT codebase.

---

### 1. Codebase & Tensor Conventions

1. **Standard Transformer Invariance (Rule of Zero Modification)**:
   - The core Transformer attention and feedforward layers (`tgwt/model/transformer.py`) must remain **100% standard PyTorch modules**.
   - No custom CUDA kernels (`.cu`), Triton code, or non-standard attention operations are permitted inside the Transformer core.
   - Any modern off-the-shelf Transformer block (including PyTorch `F.scaled_dot_product_attention`, HuggingFace `GPT2Block`, or Llama Decoder Layer) must be plug-and-play compatible.

2. **Tensor Shapes & Dimensions**:
   - Leaf Sequence: `[B, N, D]` where $B$ is batch size, $N$ is sequence length, $D$ is model dimension.
   - Total Graph Nodes: $K \approx N + \frac{N}{k} + \frac{N}{k^2} + \dots + 1 \approx 1.33 N$ (for branching factor $k=4$).
   - Node Features Tensor: `[B, K, D]`.
   - Node Properties Tensor: `[B, K, P_node]` containing:
     - `depth_norm`: normalized level in $[0, 1]$.
     - `visit_count`: normalized scalar frequency.
     - `surprise_score`: running loss / entropy scalar.
     - `node_type_id`: integer enum (0=leaf, 1=phrase, 2=section, 3=root).
   - Walk Sequence: `[B, M, W]` where $M$ is number of parallel walks and $W$ is walk length (e.g., $W=32$ or $64$).
   - Transformer Input: `[B * M, W, D]` — processed in standard batched parallel mode.
   - Walk Edge Properties Tensor: `[B * M, W, P_edge]` containing:
     - `edge_type_id`: integer enum (0=self, 1=sequential_next, 2=sequential_prev, 3=tree_child, 4=tree_parent, 5=semantic_bridge).
     - `hop_distance`: signed log-scaled jump distance.
     - `depth_delta`: difference in tree hierarchy depth.

3. **Scatter Pooling Vectorization**:
   - Scatter pooling from `[B * M, W, D]` back to `[B, K, D]` must use vectorized operations (`torch.scatter_add_` or PyTorch `scatter_reduce` with `reduce="mean"`).
   - Python loops over walks or nodes are forbidden in the forward/backward critical path to maintain high throughput on GPU.

---

### 2. Algorithmic Rules

1. **Deterministic Coverage Guarantee**:
   - Walk serialization must follow the score:
     $$\text{Score}(u, v) = \alpha \cdot \text{Surprise}(v) - \beta \cdot \text{VisitCount}(v) + \gamma \cdot \text{Weight}(u, v)$$
   - Deterministic tie-breaking (e.g., by node index) ensures 100% reproducible execution for a given seed.
   - Every leaf node must be visited by at least one walk (coverage threshold = 1.0) to ensure every token receives contextual attention.

2. **Gated Node Updating**:
   - When pooling multiple walk outputs $\tilde{\mathbf{h}}_v$ for node $v$, updates must pass through an LSTM-style input gate $\mathbf{g}_v$:
     $$\mathbf{g}_v = \sigma(\mathbf{W}_g [\mathbf{h}_v; \tilde{\mathbf{h}}_v] + \mathbf{b}_g)$$
     $$\mathbf{h}_v \leftarrow (1 - \mathbf{g}_v) \odot \mathbf{h}_v + \mathbf{g}_v \odot \tanh(\mathbf{W}_u \tilde{\mathbf{h}}_v + \mathbf{b}_u)$$
   - This prevents property drift and numerical explosion over stacked layers.

3. **Linear Complexity Enforcement**:
   - The number of walks $M$ scales linearly with sequence length $N$:
     $$M = \left\lceil \frac{c \cdot N}{W} \right\rceil$$
     where $c$ is the walk coverage oversample factor ($1.2 \le c \le 2.0$) and $W$ is constant.
   - Attention computational complexity is $O(M \cdot W^2) = O\left(\frac{c N}{W} \cdot W^2\right) = O(c \cdot W \cdot N)$, which is strictly **$O(N)$ linear**.

---

### 3. Google Colab & NVIDIA T4 Hardware Rules

1. **VRAM Budget (16 GB T4)**:
   - Peak allocated memory during training must stay below 14.5 GB to prevent Colab out-of-memory (OOM) crashes.
   - Use `torch.cuda.amp.autocast(dtype=torch.float16)` by default on T4.
   - Enable PyTorch SDPA (`torch.nn.functional.scaled_dot_product_attention`) which automatically uses memory-efficient attention or FlashAttention on CUDA.

2. **Self-Contained & Resilient Execution**:
   - Any external dependency (like HuggingFace `transformers` or `datasets`) must have clean, graceful local fallbacks or automated installation scripts.
   - Colab scripts must support resume from checkpoint in case of session timeouts.

---

### 4. Benchmarking & Fair Comparison Rules

1. **Parameter Parity**:
   - Both the TGWT model and the baseline standard Transformer must match total trainable parameters within $\pm 2\%$.
   - Any auxiliary parameters in TGWT (fusion projection, gated updater, property embeddings) must be accounted for by adjusting the hidden dimension or MLP dimension of the baseline.

2. **Uniform Training Hyperparameters**:
   - Same tokenizer vocabulary.
   - Same random seed across runs.
   - Same batch size in total tokens per step.
   - Same optimizer (AdamW), weight decay ($0.01$), learning rate ($3\times 10^{-4}$), and cosine decay schedule with 10% warmup steps.

3. **Metrics Monitored**:
   - Test Perplexity (PPL) $\exp(\mathcal{L})$.
   - Throughput (Tokens/second).
   - Peak VRAM (MB via `torch.cuda.max_memory_allocated()`).
   - FLOPs per token.
   - Downstream zero-shot evaluation accuracy.

# KromCanon — Complete Technical Reference

This document contains everything needed to implement the project without web access.

---

## Table of Contents

1. [Canon Layers](#1-canon-layers)
2. [Hyper-Connections (HC)](#2-hyper-connections)
3. [KromHC](#3-kromhc)
4. [Reference Implementations](#4-reference-implementations)
5. [Training Infrastructure](#5-training-infrastructure)
6. [Interpretability Background](#6-interpretability-background)
7. [Paper Index](#7-paper-index)
8. [Code Repositories](#8-code-repositories)
9. [Pretrained Models](#9-pretrained-models)

---

## 1. Canon Layers

**Paper**: "Physics of Language Models: Part 4.1 — Architecture Design and the Magic of Canon Layers" (Allen-Zhu, 2025)
**ArXiv**: 2512.17351
**Code**: github.com/facebookresearch/PhysicsLM4 (Apache 2.0)

### What Canon Layers Are

Canon layers are **trainable 1-D causal convolutions** (kernel size 4) that promote local horizontal information flow between neighboring tokens. Named after the musical form where a melody is imitated by successive voices.

### Architecture

Canon layers are inserted at up to **four points** per transformer block:

- **Canon-A** (pre-attention): applied to normalized hidden states before self-attention
- **Canon-B** (attention projections): applied to concatenated Q, K, V after linear projection, before RoPE
- **Canon-C** (pre-MLP): applied after attention residual, before feedforward
- **Canon-D** (MLP internal): applied to concatenated gate and up projections inside MLP

The standard useful set is **A+B** (referred to as "AB" in configs). Full set is "ABCD".

### Core Implementation

Canon is a `nn.Conv1d` wrapper operating on the feature dimension:

```python
class ShortConvolution(nn.Module):
    """1-D causal convolution for local token mixing."""

    def __init__(self, d_model: int, kernel_size: int = 4, bias: bool = False, activation: str | None = None):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,        # depthwise — each channel independently
            padding=kernel_size-1, # causal: pad left only
            bias=bias,
        )
        self.activation = nn.SiLU() if activation == "silu" else None

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq_len, d_model)
        x = x.transpose(1, 2)              # (batch, d_model, seq_len)
        x = self.conv(x)[..., :seq_len]    # causal: trim right padding
        x = x.transpose(1, 2)              # back to (batch, seq_len, d_model)
        if self.activation:
            x = self.activation(x)
        return x
```

### Integration into Transformer Block

```python
# Standard block:
x = x + attn(norm1(x))
x = x + ffn(norm2(x))

# With Canon-AB:
h = norm1(x)
h = canon_a(h) + h  # Canon-A with residual (if canon_residual=True)
x = x + attn(h)     # attention gets locally-mixed input
# Inside attn: q,k,v = proj(h); qkv = canon_b(cat(q,k,v)) + cat(q,k,v); q,k,v = split(qkv)
h = norm2(x)
x = x + ffn(h)
```

### Key Properties

- **Depthwise**: each feature channel has its own kernel — no cross-channel mixing
- **Causal**: left-padded, no future information leakage
- **Kernel size 4**: looks at current + 3 previous tokens
- **Parameter overhead**: ~0.5% of total model parameters
- **Residual**: Canon output is added to input (when `canon_residual=True`)

### Config Parameters

```python
canon_set: str = ""       # Which canon layers: "", "A", "AB", "ABCD"
canon_bias: bool = False  # Bias in conv
canon_activation: bool = False  # SiLU activation
canon_kernel: int = 4     # Kernel size
canon_residual: bool = True  # Residual connection around canon
```

### Results (from Part 4.1/4.2)

- Reasoning depth: 2-4x improvement
- Reasoning breadth: +30%
- Knowledge manipulation length: +30%
- Knowledge capacity: +10-15%
- Works across: Transformers, linear attention (GLA, GDN), SSMs (Mamba2)
- Pretrained models: 1B, 3B, 8B Llama/LlamaCanon on HuggingFace

### Inference Caching

For autoregressive generation, Canon maintains a rolling cache of the last `kernel_size-1` hidden states per layer. The `step()` method processes one token at a time:

```python
def step(self, x: Tensor, cache: Tensor) -> tuple[Tensor, Tensor]:
    # x: (batch, 1, d_model), cache: (batch, d_model, kernel_size-1)
    cache = torch.cat([cache, x.transpose(1,2)], dim=-1)
    y = (cache * self.conv.weight.squeeze()).sum(dim=-1, keepdim=True)
    return y.transpose(1,2), cache[..., 1:]  # shift cache
```

---

## 2. Hyper-Connections (HC)

**Paper**: "Hyper-Connections" (2024)
**ArXiv**: 2409.19606

### The Seesaw Problem

Pre-Norm fixes gradient vanishing but causes representation collapse (deeper layers become similar).
Post-Norm fixes representation collapse but reintroduces gradient vanishing.
HC learns the optimal balance.

### Core Formulation

Standard residual: `x_{l+1} = x_l + F(x_l)`

HC expands the residual stream to n copies:

```
H^0 = (h^0, h^0, ..., h^0)^T  ∈ R^{n×d}
```

At each layer, **width connection** mixes the n streams:

```
(h_input, H') = WC^T · H
where WC = [A_m  A_r] ∈ R^{n×(n+1)}
```

- `A_m ∈ R^{n×1}`: weights for producing the single branch input from n streams
- `A_r ∈ R^{n×n}`: residual pathway mixing matrix

Then **depth connection** combines branch output back:

```
H_next = B^T · F(h_input) + H'
```

- `B ∈ R^{1×n}`: output distribution weights

### Dynamic HC

Parameters become input-dependent:

```
A_m(H) = s_α · tanh(norm(H) · W_m) + A_m_static
A_r(H) = s_α · tanh(norm(H) · W_r) + A_r_static
B(H)   = s_β · tanh(norm(H) · W_β) + B_static
```

Where `s_α`, `s_β` are learnable scales, and `W_m`, `W_r`, `W_β` are initialized to zero (so it starts as static HC).

### Initialization

Layer k:
```
HC = [0       1 1 ... 1  ]     (1×(n+1))
     [e_{k%n}   I_{n×n}  ]     (n×(n+1))
```

Where `e_i` is the i-th basis vector. Effect: equivalent to Pre-Norm when n=1.

### Optimal n

n=4 consistently best across model sizes (1B, 7B, MoE).

### Special Cases

- n=1, static: equivalent to Pre-Norm
- n=2, specific initialization: equivalent to sequential or parallel transformer blocks
- The architecture can learn to rearrange layers dynamically per token

### Results

- 1B model: val loss improvement of 0.034 vs baseline
- 7B model: downstream accuracy 71.0% vs 70.1%
- MoE: 1.8x faster convergence, +6 points on ARC-Challenge

---

## 3. KromHC

**Paper**: "KromHC: Manifold-Constrained Hyper-Connections with Kronecker-Product Residual Matrices" (Wang et al., 2025)
**ArXiv**: 2601.21579
**Code**: github.com/wz1119/KromHC

### Problem with HC

The residual mixing matrix `A_r` is unconstrained → can cause gradient explosion and training instability.

### mHC: Manifold-Constrained HC

Constrains `H^res` (the residual mixing matrix) to be **doubly stochastic** (rows and columns sum to 1, all entries non-negative).

Uses Sinkhorn-Knopp algorithm (20 iterations) to project onto the Birkhoff polytope.

**Equations:**
```
x'_l = RMSNorm(x_l)                               # flatten to (1, nC)
H^pre_l = σ(α^pre_l · x'_l · W^pre_l + b^pre_l)   # sigmoid, ∈ R^{1×n}
H^post_l = 2σ(α^post_l · x'_l · W^post_l + b^post_l) # sigmoid×2, ∈ R^{1×n}
H^res_l = SK(α^res_l · mat(x'_l · W^res_l) + b^res_l) # Sinkhorn, ∈ R^{n×n}
```

**Problem**: SK doesn't guarantee exact double stochasticity. MAE ≈ 0.05 on column sums.
**Complexity**: O(n³C) from `W^res ∈ R^{nC × n²}`

### mHC-lite: Exact via Birkhoff-von-Neumann

Uses the theorem that any doubly stochastic matrix is a convex combination of permutation matrices:

```
H^res_l = Σ_{k=1}^{n!} a_l(k) · P_k
where a_l = softmax(α^res_l · x'_l · W^res_l + b^res_l)
```

**Problem**: n! permutation matrices → factorial parameter explosion O(nC · n!)

### KromHC: Kronecker Product Decomposition

**Key insight**: tensorize the n residual streams into an order-(K+1) tensor where n = ∏_{k=1}^K i_k.

For n=4: factor as 2×2, giving K=2 Kronecker factors.
For n=8: factor as 2×2×2, giving K=3 factors.

Each factor U_l^k is a small doubly stochastic matrix parametrized via Birkhoff-von-Neumann:

```
a_l^k = softmax(α^res_l · x'_l · W^{res,k}_l + b^{res,k}_l)
U_l^k = Σ_{m=1}^{i_k!} a_l^k(m) · P_m
```

The full residual matrix is their Kronecker product:

```
H^res_l = U_l^1 ⊗ U_l^2 ⊗ ... ⊗ U_l^K
```

**Theorem 4.2 (Kronecker Closure)**: The Kronecker product of doubly stochastic matrices is doubly stochastic.

Proof: Non-negativity from products. Row/column sums: (A⊗B)(1_{i1}⊗1_{i2}) = (A·1_{i1})⊗(B·1_{i2}) = 1⊗1 = 1.

### For 2×2 Factors (Optimized Path)

When i_k = 2, each factor has only 2 permutations (identity and swap):

```
U_l^k = p_k · I + (1-p_k) · [[0,1],[1,0]]
      = [[p_k, 1-p_k], [1-p_k, p_k]]
```

Where `p_k = softmax(coefficients)[0]`. This is the fast path in the implementation.

### Complexity

| Method   | Complexity  |
|----------|------------|
| mHC      | O(n³C)    |
| mHC-lite | O(nC·n!)  |
| KromHC   | O(n²C)    |

For n=4, C=768: mHC needs 1844K extra params, KromHC needs 959K (48% reduction).

### Initialization

- All W matrices initialized to zero
- `b^pre, b^post = [-1,...,-1, 1, -1,...,-1]` (single 1 at layer_index % n)
- `α^pre = α^post = α^res = 0.01`
- For i_k=2: `b^{res,k} = [0, -8]` → softmax gives [~1, ~0] → U ≈ Identity
- **Effect**: at initialization, H^res ≈ I, H^pre selects one stream, H^post writes to one stream

### Training Configuration (from paper)

- Model: Nanochat architecture, 60M (D=6) and 186M (D=12) params
- Residual streams: n ∈ {4, 8, 16} (n=4 is sweet spot)
- Dataset: FineWeb-Edu, token:param ratio ≈ 20
- Optimizer: Muon (LR=0.02, WD=0.2) for main; AdamW (LR=0.005, β1=0.8, β2=0.95) for HC params
- Batch: 524,288 tokens, seq_len=2048
- Warmup: 40% of total steps
- Hardware: 4-8 NVIDIA RTX PRO 6000

### Results (D=12, n=4)

| Method   | Δ Params (K) | Train Loss | Val BPB | CORE Score |
|----------|-------------|-----------|---------|-----------|
| Residual | —           | 2.971     | 0.864   | 14.774    |
| mHC      | 1844        | 2.964     | 0.861   | 16.023    |
| mHC-lite | 2433        | 2.972     | 0.864   | 13.217    |
| KromHC   | 959         | 2.966     | 0.862   | 16.872    |

### Gradient Stability

KromHC has lowest gradient norm across training (Figure 5 in paper). The doubly stochastic constraint ensures spectral norm ≤ 1, preventing gradient explosion through residual connections.

---

## 4. Reference Implementations

### KromHC (PyTorch) — Full Source

From github.com/wz1119/KromHC/blob/main/hyper_conn/Kromhc.py

Key classes and their roles:

**`KromHC(Module)`** — main class
- `__init__`: sets up Kronecker factors, static/dynamic alpha/beta, scales
- `_build_kronecker_hres(dynamic_coeffs, static_coeffs, device)`: constructs H^res via Kronecker product
- `width_connection(residuals)`: produces branch_input and updated residuals
- `depth_connection(branch_output, residuals, beta)`: combines branch output back
- `forward(residuals)`: returns (branch_input, add_residual_fn) or runs branch

**Einops translations needed for MLX port:**
```
rearrange('(b s) ... d -> b ... s d', s=n)  →  reshape + transpose
rearrange('b ... s d -> (b s) ... d')        →  reshape
repeat('b ... -> (b s) ...', s=n)            →  tile/repeat
einsum('... f1 s f2 t, ... f1 s d -> ... f2 t d')  →  mx.einsum or manual contraction
```

**Key dimensions:**
- Input: `(batch * n_streams, seq_len, dim)` (streams packed into batch)
- Internal: `(batch, seq_len, n_streams, dim)` (unpacked for mixing)
- H^res: `(batch, seq_len, n_streams, n_streams)` (per-token mixing matrix)

### Canon (PyTorch) — Core

From github.com/facebookresearch/PhysicsLM4/huggingface/canon_helper.py

**`ShortConvolution(Module)`** — causal conv wrapper
- `nn.Conv1d` with `groups=d_model` (depthwise)
- Left-padding of `kernel_size - 1`
- Optional SiLU activation
- `step()` method for autoregressive caching

**`create_canon(config, d_model, layer_idx)`** — factory
**`apply_canon(name, canon_module, hidden_states, cache, layer_idx, mask)`** — integration helper with residual

### Integration Points in LlamaCanon

```python
# In LlamaCanonDecoderLayer:
if "A" in config.canon_set:
    self.canonA = create_canon(config, hidden_size, layer_idx)
if "C" in config.canon_set:
    self.canonC = create_canon(config, hidden_size, layer_idx)

# In LlamaCanonAttention:
if "B" in config.canon_set:
    self.canonB = create_canon(config, num_heads * head_dim * 3, layer_idx)  # Q+K+V

# In LlamaCanonMLP:
if "D" in config.canon_set:
    self.canonD = create_canon(config, intermediate_size * 2, layer_idx)  # gate+up
```

---

## 5. Training Infrastructure

### nanochat-mlx

**Repo**: github.com/scasella/nanochat-mlx
**Framework**: Pure MLX (no PyTorch dependency)
**License**: Check repo

#### Project Structure
```
nanochat_mlx/
  gpt.py              # Transformer architecture
  optim.py            # Muon + AdamW optimizers
  engine.py           # KV-cache inference
  train.py            # Training loop
  sft.py              # Supervised fine-tuning
  eval.py             # Bits-per-byte evaluation
  dataloader.py       # BOS-aligned packing
  sft_dataloader.py   # Conversation packing
  dataset.py          # Data download
  tokenizer.py        # BPE tokenizer
scripts/
  train.py            # Training CLI
  sft.py              # SFT CLI
  chat.py             # Inference CLI
```

#### Depth Scaling

| Depth | Params | Time (M3 Pro) | Time (M4 Pro est.) |
|-------|--------|---------------|-------------------|
| 4     | ~5M    | ~1 min        | ~30 sec           |
| 12    | ~125M  | ~1 hour       | ~30-40 min        |
| 20    | ~350M  | ~8 hours      | ~4-5 hours        |
| 26    | ~600M  | ~24 hours     | ~12-15 hours      |

#### Training Pipeline
```bash
# 1. Download data
python -m nanochat_mlx.dataset -n 8

# 2. Train tokenizer
python -m scripts.tok_train

# 3. Pretrain
python -m scripts.train --depth=12

# 4. SFT
python -m scripts.sft --depth=12

# 5. Chat
python -m scripts.chat --depth=12 --source=sft --interactive
```

#### Key Architecture Details

- Standard GPT-2 decoder-only transformer
- Muon optimizer (primary) + AdamW (fallback)
- BOS-aligned best-fit bin packing for data
- BPE tokenizer with 32,768 vocabulary

### Safety Fine-Tuning Datasets

**Anthropic HH-RLHF**:
- HuggingFace: `Anthropic/hh-rlhf`
- Format: chosen/rejected response pairs
- Use: train model to refuse harmful, comply with helpful

**BeaverTails**:
- HuggingFace: `PKU-Alignment/BeaverTails`
- Format: safety-labeled responses across 14 harm categories
- Use: diverse safety training signal

**Format for SFT**: convert to conversation pairs:
```json
{"messages": [
  {"role": "user", "content": "<harmful prompt>"},
  {"role": "assistant", "content": "I can't help with that..."}
]}
{"messages": [
  {"role": "user", "content": "<helpful prompt>"},
  {"role": "assistant", "content": "<helpful response>"}
]}
```

---

## 6. Interpretability Background

### Abliteration (Arditi et al., 2024)

**Paper**: "Refusal in Language Models Is Mediated by a Single Direction" — arxiv.org/abs/2406.11717

Core finding: refusal behavior in LLMs is controlled by a single linear direction in activation space.

**Direction extraction (mean-diff)**:
```python
# Collect activations on harmful prompts
harmful_acts = [model.get_activations(p) for p in harmful_prompts]
# Collect activations on harmless prompts
harmless_acts = [model.get_activations(p) for p in harmless_prompts]
# Direction = difference of means
refusal_dir = mean(harmful_acts) - mean(harmless_acts)
refusal_dir = refusal_dir / norm(refusal_dir)
```

**Direction removal (abliteration)**:
```python
# Project out the refusal direction from weight matrices
for layer in model.layers:
    W = layer.self_attn.o_proj.weight
    proj = outer(refusal_dir, refusal_dir)
    W.data = W.data - W.data @ proj
```

**Key questions for non-standard architectures**:
1. Is the refusal direction still rank-1 (single direction)?
2. Does it live in the same layer range?
3. Can it be extracted from the pre-mixing activations or only post-mixing?
4. For multi-stream (KromHC): is it in one stream or spread across streams?
5. For Canon: does local mixing concentrate or disperse the direction?

### Direction Extraction Methods

1. **Mean-diff**: direction = mean(harmful) - mean(harmless). Simplest, works well.
2. **SVD (subspace)**: take top-k singular vectors of the difference matrix. Captures multi-dimensional effects.
3. **PCA on contrast pairs**: PCA on the difference vectors, top component is the direction.

### Steering

Add/subtract a scaled direction from activations during forward pass:
```python
# During forward pass at layer l:
x = x + alpha * refusal_direction  # increase refusal
x = x - alpha * refusal_direction  # decrease refusal (steer toward compliance)
```

### What Changes with KromHC

In vanilla transformers: single residual stream `x ∈ R^d`
In KromHC (n=4): four coupled streams `X ∈ R^{4×d}` mixed at every layer

Direction extraction options:
1. **Per-stream**: extract direction from each stream independently → 4 directions
2. **Joint**: concatenate all streams → direction in R^{4d}
3. **Post-reduction**: extract after attention pooling reduces to single stream
4. **Per-stream-then-combine**: extract per-stream, study correlation

The doubly stochastic mixing means information flows between streams at every layer. A direction that starts in stream 0 will partially spread to streams 1-3 over depth.

### What Changes with Canon

Canon adds local mixing (kernel=4) before attention. This means:
- Token t's representation is influenced by tokens t-1, t-2, t-3 before attention
- The refusal direction at token t may partially reflect neighboring tokens' state
- Extraction should account for this: compare direction at position t with/without Canon

---

## 7. Paper Index

### Primary Papers (implement these)

| Paper | ArXiv | Key Contribution |
|-------|-------|-----------------|
| Canon Layers (Allen-Zhu, Part 4.1) | 2512.17351 | 1-D causal conv for local token mixing |
| Canon at Scale (Part 4.2) | physics.allen-zhu.com | Real-world validation |
| KromHC (Wang et al.) | 2601.21579 | Kronecker doubly stochastic residuals |
| Hyper-Connections | 2409.19606 | Multi-stream residual foundation |
| Abliteration (Arditi et al.) | 2406.11717 | Refusal = single linear direction |

### Physics of LMs Series (context)

| Paper | ArXiv | Topic |
|-------|-------|-------|
| Part 1 | 2305.13673 | Hierarchical language structures |
| Part 2.1 | 2407.20311 | Grade-school math reasoning |
| Part 2.2 | 2408.16293 | Learning from mistakes |
| Part 3.1 | 2309.14316 | Knowledge storage and extraction |
| Part 3.2 | 2309.14402 | Knowledge manipulation |
| Part 3.3 | 2404.05405 | Knowledge capacity scaling laws |

Key finding from Part 3.1: knowledge must be augmented (paraphrased) during pretraining to be extractable. Without augmentation → memorized but not extractable. Relevant: safety behaviors relying on knowledge manipulation may be inherently fragile (Part 3.2).

Key finding from Part 3.3: models store exactly 2 bits of knowledge per parameter.

### Steering/Defense Papers (context)

| Paper | ArXiv | Relevance |
|-------|-------|-----------|
| CAST | 2409.05907 | Conditional activation steering (defense) |
| RepBend | 2504.01550 | Loss-based defense dual of abliteration |
| AdaSteer | 2504.09466 | Separate detect/steer directions |
| TRYLOCK | 2601.03300 | Non-monotonic danger in fixed-alpha steering |
| AlphaSteer | 2506.07022 | Adaptive alpha per-token |
| SVF (Li et al.) | 2602.01654 | Differentiable boundary MLP for steering |

---

## 8. Code Repositories

| Repo | URL | What | License |
|------|-----|------|---------|
| KromHC | github.com/wz1119/KromHC | KromHC + mHC PyTorch impl | Check repo |
| PhysicsLM4 | github.com/facebookresearch/PhysicsLM4 | Canon layers + data generators | Apache 2.0 |
| nanochat-mlx | github.com/scasella/nanochat-mlx | MLX GPT-2 training pipeline | Check repo |
| iGSM | github.com/facebookresearch/iGSM | Grade-school math data gen | Check repo |

---

## 9. Pretrained Models

### LlamaCanon (HuggingFace)

Collection: `zhuzeyuan/physics-of-language-models-series`
Part 4.2 models: `zhuzeyuan/physics-of-language-models-part-42`

16 Llama/LlamaCanon checkpoints across 1B, 3B, 8B sizes.

**WARNING**: These are Llama-based, not GPT-2. They're useful for:
- Verifying Canon integration works
- Comparing interp results with our custom models
- NOT directly comparable to our GPT-2 variants

### Allen-Zhu Synthetic Datasets

Available in PhysicsLM4 repo under `data-synthetic-pretrain/`:

| Dataset | Tests | Origin |
|---------|-------|--------|
| Lano | Structural reasoning, dynamic programming | Part 1 |
| Capo | Knowledge encoding (synthetic biographies) | Parts 3.1, 3.3 |
| Mano | Hierarchical computation over stored knowledge | Part 4.1 |
| Depo | Synthetic pretraining task | Part 4.1 |
| Brevo | Synthetic pretraining task | Part 4.1 |

---

## Appendix: MLX Porting Notes

### Einops → MLX Native

```python
# rearrange('(b s) t d -> b t s d', s=4)
x = x.reshape(b, s, t, d).transpose(0, 2, 1, 3)  # adjust axes as needed

# rearrange('b t s d -> (b s) t d')
x = x.transpose(0, 2, 1, 3).reshape(b*s, t, d)

# repeat('b t d -> (b s) t d', s=4)
x = mx.repeat(x[:, None], repeats=s, axis=1).reshape(b*s, t, d)

# einsum('... i j, ... j d -> ... i d')
x = mx.einsum('...ij,...jd->...id', A, B)

# reduce('(b s) t d -> b t d', 'sum', s=4)
x = x.reshape(b, s, t, d).sum(axis=1)
```

### PyTorch → MLX Mapping

```
torch.nn.Module        → mlx.nn.Module
torch.nn.Linear        → mlx.nn.Linear
torch.nn.Conv1d        → mlx.nn.Conv1d (note: MLX uses channels-last by default)
F.softmax(x, dim=-1)   → mx.softmax(x, axis=-1)
F.normalize(x, dim=-1) → x / mx.linalg.norm(x, axis=-1, keepdims=True)
torch.cat              → mx.concatenate
torch.stack            → mx.stack
nn.Parameter           → just an mx.array attribute (MLX uses functional updates)
x.sigmoid()            → mx.sigmoid(x)
x.device               → not needed (MLX unified memory)
.to(device)            → not needed
```

### MLX Conv1d Specifics

MLX `nn.Conv1d` expects input shape `(batch, seq_len, channels)` (channels-last), unlike PyTorch's `(batch, channels, seq_len)`. No transpose needed if your tensors are already in this format.

```python
# MLX causal conv1d
class CausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            # MLX doesn't have groups= for depthwise directly
            # Use separate convolutions per channel or implement manually
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, seq_len, channels)
        # Pad left for causal
        x = mx.pad(x, [(0,0), (self.pad, 0), (0,0)])
        return self.conv(x)
```

**NOTE**: MLX's `nn.Conv1d` may not support `groups=d_model` for depthwise conv as of early 2026. Check the version. If not supported, implement depthwise as element-wise multiplication with a learned kernel:

```python
class DepthwiseCausalConv(nn.Module):
    """Depthwise causal 1-D convolution for MLX."""
    def __init__(self, d_model: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = mx.zeros((d_model, kernel_size))  # (channels, kernel)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, seq_len, d_model)
        b, t, d = x.shape
        # Pad left for causal
        x_padded = mx.pad(x, [(0,0), (self.kernel_size-1, 0), (0,0)])
        # Sliding window: gather (b, t, d, kernel_size)
        windows = mx.stack([x_padded[:, i:i+t, :] for i in range(self.kernel_size)], axis=-1)
        # Element-wise multiply and sum over kernel dimension
        return (windows * self.weight).sum(axis=-1)
```

# KromCanon

## What This Is

A research project studying how steering vectors and linear direction extraction behave in non-standard transformer architectures. We train three GPT-2 124M variants from scratch on Apple Silicon (M4 Pro, 24GB) and compare interpretability properties across them.

### The Three Variants

1. **Vanilla** — standard GPT-2 124M (baseline)
2. **Canon** — GPT-2 124M + Canon layers (1-D causal conv, kernel=4, before attention and on Q/K/V)
3. **KromCanon** — GPT-2 124M + Canon layers + KromHC residual connections (Kronecker-product doubly stochastic multi-stream mixing, n=4 streams)

### The Research Question

> Do abliteration-style techniques (finding and removing linear behavioral directions from activations) generalize to architectures with multi-stream residual coupling and local token mixing?

Specifically:
- Are refusal/behavioral directions still linear in KromHC's multi-stream residual space?
- Does Canon's local convolution disperse or concentrate directional information?
- How does direction extraction (mean-diff, SVD) behave when the residual stream is n=4 coupled streams?

## Architecture Details

### Canon Layers (Allen-Zhu, Part 4.1 — arxiv.org/abs/2512.17351)

Trainable 1-D causal convolutions (kernel size 4) inserted at two points per transformer block:
- **Canon-A**: before attention — local token mixing in the residual stream
- **Canon-B**: applied to Q, K, V projections inside attention

~0.5% parameter overhead. Improves reasoning depth 2-4x, knowledge capacity +10-15%.

Key property: **architecture-agnostic** — works with transformers, linear attention, SSMs.

Reference implementation: github.com/facebookresearch/PhysicsLM4 (Apache 2.0)
Paper: physics.allen-zhu.com

### KromHC (Wang et al. — arxiv.org/abs/2601.21579)

Replaces standard residual connections (`x = x + F(x)`) with multi-stream mixing:

```
X_{l+1} = H^res · X_l + H^post · F(H^pre · X_l)
```

Where `H^res` is a doubly stochastic matrix built as a Kronecker product of small (2×2) factor matrices:

```
H^res = U_1 ⊗ U_2 ⊗ ... ⊗ U_K
```

Each `U_k` is a learned convex combination of permutation matrices → guaranteed doubly stochastic.

- n=4 streams → 2 Kronecker factors of 2×2 each
- O(n²C) parameter complexity
- Norm-preserving by construction (spectral norm ≤ 1)

Reference implementation: github.com/wz1119/KromHC (PyTorch, needs MLX port)

## Training Plan

### Phase 1: Pretrain (FineWeb-Edu, ~1.2B tokens)

Base: **nanochat-mlx** (github.com/scasella/nanochat-mlx) — pure MLX, depth-scaled GPT-2 training.

Use `--depth=12` (~125M params). Train all three variants on identical data with identical hyperparams.

Expected time per variant: ~30-40 min on M4 Pro.

### Phase 2: Safety Fine-Tune

Fine-tune each variant on helpful/harmful contrast pairs to create measurable refusal behavior:
- Anthropic HH-RLHF (helpful vs harmful)
- BeaverTails (safety-annotated)
- Format: SFT on "refuse harmful, comply helpful" examples

This gives us models with a direction to extract.

### Phase 3: Interpretability

For each variant, extract and compare:
1. **Refusal direction** via mean-diff (harmful vs helpful activations)
2. **Subspace** via SVD of activation differences
3. **Per-layer projection profiles** — where does the direction live?
4. **Abliteration** — remove the direction, measure refusal rate change
5. **Steering** — add/subtract direction, measure behavioral shift

Key comparisons:
- Vanilla vs Canon: does local conv mixing affect direction linearity?
- Vanilla vs KromCanon: does multi-stream coupling spread the direction across streams?
- Canon vs KromCanon: marginal effect of KromHC on top of Canon

## Implementation Plan

### Step 1: Set Up nanochat-mlx Base

```bash
git clone https://github.com/scasella/nanochat-mlx.git
# Understand the architecture in nanochat_mlx/gpt.py
# Understand the training loop in nanochat_mlx/train.py
```

The transformer block in nanochat-mlx follows standard structure:
```python
x = x + attn(norm1(x))
x = x + ffn(norm2(x))
```

### Step 2: Add Canon Layers (MLX)

Port Canon-A and Canon-B as MLX modules:

```python
class CanonConv(nn.Module):
    """1-D causal convolution, kernel_size=4"""
    # Conv over sequence dimension with causal padding
    # Canon-A: applied to residual before attention
    # Canon-B: applied to Q, K, V after projection
```

Modified block:
```python
x = canon_a(x)                    # local mixing before attention
q, k, v = proj(x)                 # standard projections
q, k, v = canon_b_q(q), canon_b_k(k), canon_b_v(v)  # Canon-B
x = x + attn(q, k, v)
x = x + ffn(norm(x))
```

Reference: PhysicsLM4/huggingface/ for the Llama implementation of Canon.

### Step 3: Add KromHC Residual Connections (MLX)

Port KromHC from PyTorch to MLX. Core components:
- `_build_kronecker_hres()` — Kronecker product of 2×2 doubly stochastic factors
- `width_connection()` — mix residual streams before branch (attn/ffn)
- `depth_connection()` — mix branch output back into streams

Key: the residual stream expands from `(batch, seq, dim)` to `(batch * n_streams, seq, dim)` with learned inter-stream mixing at every layer.

The full KromHC PyTorch implementation is ~400 lines. The MLX port should be straightforward — replace `torch.nn` with `mlx.nn`, `einsum` with `mx.einsum`, `F.softmax` with `mx.softmax`.

**IMPORTANT**: The KromHC reference code uses `einops` heavily. MLX does not have einops. You must translate all `rearrange`, `repeat`, `reduce`, `einsum` calls to native `mx.reshape`, `mx.tile`, `mx.sum`, `mx.einsum` equivalents.

### Step 4: Training Runs

```bash
# Vanilla (baseline)
python -m scripts.train --depth=12

# Canon (Canon layers only)
python -m scripts.train --depth=12 --arch=canon

# KromCanon (Canon + KromHC)
python -m scripts.train --depth=12 --arch=kromcanon
```

Implement `--arch` flag or separate config files for each variant.

### Step 5: Safety Fine-Tuning

Use nanochat-mlx's SFT pipeline (`scripts/sft.py`) with safety contrast data:

```bash
python -m scripts.sft --depth=12 --arch=vanilla --data=safety
python -m scripts.sft --depth=12 --arch=canon --data=safety
python -m scripts.sft --depth=12 --arch=kromcanon --data=safety
```

### Step 6: Interp Tooling

Build extraction and comparison scripts:

```python
# extract_directions.py — mean-diff and SVD direction extraction
# compare_architectures.py — cross-variant direction analysis
# steer.py — activation steering with extracted directions
# abliterate.py — direction removal and refusal rate measurement
```

For KromCanon, direction extraction must account for n=4 streams:
- Extract per-stream directions
- Extract joint direction across concatenated streams
- Compare: is the direction concentrated in one stream or distributed?

## Dependencies

- **mlx**, **mlx-lm** — Apple ML framework
- **nanochat-mlx** — base training infrastructure
- **datasets** (HuggingFace) — for FineWeb-Edu and safety datasets
- **numpy** — numerical ops

## Key Reference Code

- KromHC PyTorch: github.com/wz1119/KromHC/blob/main/hyper_conn/Kromhc.py
- Canon (LlamaCanon): github.com/facebookresearch/PhysicsLM4/tree/main/huggingface
- nanochat-mlx: github.com/scasella/nanochat-mlx

## Papers

- KromHC: arxiv.org/abs/2601.21579
- Canon Layers (Physics of LMs Part 4.1): arxiv.org/abs/2512.17351
- Canon at Scale (Part 4.1): physics.allen-zhu.com
- Abliteration: arxiv.org/abs/2406.11717 (Arditi et al.)
- Hyper-Connections (original): arxiv.org/abs/2409.19606

## Hardware

- Apple M4 Pro, 24 GB unified memory, 12 cores
- Training budget: ~30-40 min per 125M variant
- Total: ~2 hours for all three variants + safety fine-tuning

## Typing Rules

- All code must be fully typed. Every function parameter, return type, and variable.
- No `Any`. No untyped collections. Use `X | Y` over `Union[X, Y]`.
- Use modern Python 3.12+ syntax.

## Code Style

- Use `ruff` for linting.
- Minimal dependencies — prefer MLX native ops over external libraries.
- No einops — translate to native reshape/transpose/tile ops.
- Clear docstrings on all public functions.

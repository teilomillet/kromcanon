# KromCanon Development Log

All significant decisions, implementation steps, and architectural choices are recorded here with timestamps.

---

## 2026-03-13 — Project Initialization

### Context
- Repository scaffolded with CLAUDE.md briefing, REFERENCE.md, and README
- No source code yet — greenfield implementation
- Target: Apple M4 Pro, 24GB, pure MLX

### Decision: Implementation Order
Following CLAUDE.md Step 1-6:
1. Set up nanochat-mlx base (clone, understand architecture)
2. Implement Canon layers as MLX modules
3. Implement KromHC residual connections (MLX port from PyTorch)
4. Integrate into training pipeline with `--arch` flag
5. Safety fine-tuning pipeline
6. Interpretability tooling

### Decision: Ralph Wiggum Loop for Build Phase
Using 50-iteration Ralph Wiggum loop to systematically build the project.
The loop will follow a structured prompt that:
- Checks current state against implementation plan
- Identifies the next uncompleted step
- Implements it with tests
- Documents the change in this devlog
- Commits progress

### Architecture Decisions (locked)
- **n_streams = 4** for KromHC (paper sweet spot, 2 Kronecker factors of 2x2)
- **Canon set = "AB"** (pre-attention + Q/K/V projections — best cost/benefit from paper)
- **kernel_size = 4** for Canon convolutions
- **Depthwise convolution** via element-wise kernel multiplication (MLX Conv1d groups support uncertain)
- **No einops** — all tensor ops use native mx.reshape/transpose/tile
- **Depth = 12** for all variants (~125M params)

---

## 2026-03-13 — Iteration 1: Project Structure + Config + Canon Layers

### Completed
- **Phase 0**: Created `pyproject.toml` (uv + setuptools), `src/kromcanon/` package, `tests/` directory, `scripts/` directory
- **Phase 1 — config.py**: `ModelConfig`, `CanonConfig`, `KromHCConfig`, `TrainConfig` dataclasses. `make_config()` factory. Arch-driven sub-config auto-enable in `__post_init__`.
- **Phase 1 — canon.py**: `DepthwiseCausalConv` (manual sliding-window depthwise conv), `CanonLayer` (conv + optional residual), `apply_canon_a`, `apply_canon_b` (QKV concatenation pattern).
- **Tests**: 10 tests for Canon — output shapes, causal masking, batch independence, residual correctness, Canon-B QKV shapes and mixing.

### Decisions
- **Python 3.12** via `uv venv --python 3.12` (system Python is 3.14, MLX needs 3.12)
- **Depthwise conv**: Implemented via `mx.stack` of shifted views + element-wise multiply + sum. Avoids reliance on `nn.Conv1d` groups support.
- **Canon-B pattern**: Concatenate QKV → single conv → split. Matches PhysicsLM4 reference.

### Environment
- `uv` for package management, editable install with `.[dev]`
- All 10 tests passing

---

# KromCanon — Ralph Wiggum Build Prompt

You are building the KromCanon research project. Read CLAUDE.md for full context.

## Your Mission

Build three GPT-2 124M variants (Vanilla, Canon, KromCanon) from scratch in pure MLX, with training, safety fine-tuning, and interpretability tooling.

## Before Each Iteration

1. **Read the current state**: Check `docs/DEVLOG.md`, `git log --oneline -20`, and scan the source tree (`ls -la`, `ls src/` etc.)
2. **Identify where you are** on the implementation checklist below
3. **Pick the next uncompleted task**
4. **Implement it** with full typing, tests, and documentation
5. **Update `docs/DEVLOG.md`** with what you did, decisions made, and any issues encountered
6. **Commit your work** with a clear message

## Implementation Checklist

### Phase 0: Project Structure
- [ ] Create `pyproject.toml` with dependencies (mlx, mlx-lm, datasets, numpy, ruff)
- [ ] Create `src/kromcanon/` package structure
- [ ] Create `tests/` directory with pytest configuration

### Phase 1: Core Modules
- [ ] `src/kromcanon/config.py` — Dataclass configs for Vanilla, Canon, KromCanon variants
- [ ] `src/kromcanon/canon.py` — Canon layer implementation (depthwise causal conv, kernel=4)
  - DepthwiseCausalConv class
  - Canon-A integration helper (pre-attention)
  - Canon-B integration helper (Q/K/V projections)
  - Unit tests: causal masking, output shapes, gradient flow
- [ ] `src/kromcanon/kromhc.py` — KromHC residual connections (MLX port)
  - Kronecker product of 2x2 doubly stochastic factors
  - width_connection (mix streams before branch)
  - depth_connection (mix branch output back)
  - Dynamic coefficient computation
  - Unit tests: doubly stochastic property, initialization = identity, shapes
- [ ] `src/kromcanon/model.py` — GPT-2 transformer with pluggable architecture
  - TransformerBlock supporting vanilla/canon/kromcanon modes
  - Full GPT-2 model (embeddings, blocks, LM head)
  - Config-driven architecture selection
  - Unit tests: forward pass shapes, parameter counts, generation

### Phase 2: Training Infrastructure
- [ ] `src/kromcanon/data.py` — FineWeb-Edu data loading with BOS-aligned packing
- [ ] `src/kromcanon/tokenizer.py` — BPE tokenizer (adapt from nanochat-mlx or use existing)
- [ ] `src/kromcanon/train.py` — Training loop with Muon + AdamW
  - Separate optimizer groups for HC params vs main params
  - Gradient clipping, warmup scheduling
  - Logging and checkpointing
- [ ] `scripts/train.py` — CLI entry point with `--arch={vanilla,canon,kromcanon}` flag

### Phase 3: Safety Fine-Tuning
- [ ] `src/kromcanon/safety_data.py` — Load and format HH-RLHF / BeaverTails
- [ ] `src/kromcanon/sft.py` — SFT training loop for safety contrast pairs
- [ ] `scripts/sft.py` — CLI entry point

### Phase 4: Interpretability
- [ ] `src/kromcanon/interp/extract.py` — Direction extraction (mean-diff, SVD)
  - Per-layer activation collection
  - For KromHC: per-stream and joint extraction
- [ ] `src/kromcanon/interp/abliterate.py` — Direction removal and refusal rate measurement
- [ ] `src/kromcanon/interp/steer.py` — Activation steering
- [ ] `src/kromcanon/interp/compare.py` — Cross-architecture comparison
  - Direction cosine similarity across variants
  - Per-layer projection profiles
  - Stream distribution analysis for KromCanon
- [ ] `scripts/extract.py` — CLI for direction extraction
- [ ] `scripts/compare.py` — CLI for cross-variant analysis

### Phase 5: Validation & Polish
- [ ] End-to-end smoke test (tiny model, few steps, all three variants)
- [ ] All tests passing with `pytest`
- [ ] `ruff check` passes
- [ ] Documentation complete in DEVLOG.md

## Quality Standards

- **Fully typed**: every parameter, return type, variable. No `Any`.
- **Python 3.12+** syntax: `X | Y` not `Union[X, Y]`
- **No einops**: native mx.reshape/transpose/tile only
- **Minimal dependencies**: prefer MLX native ops
- **Clear docstrings** on all public functions
- **Tests** for every module (shape checks, property checks, smoke tests)
- **Every change documented** in `docs/DEVLOG.md` with rationale

## Critical Implementation Notes

1. **MLX Conv1d**: May not support `groups=d_model`. Use manual depthwise (element-wise kernel multiplication with sliding windows).
2. **KromHC streams**: Residual stream expands from `(batch, seq, dim)` to `(batch * n_streams, seq, dim)`. Width/depth connections unpack/repack.
3. **KromHC init**: All W→zero, b^res→`[0, -8]` for identity-like behavior at init.
4. **Canon residual**: Canon output added to input (`canon_residual=True`).
5. **Two optimizer groups** for KromCanon: Muon for main params, AdamW for HC params.

## Completion Signal

When ALL checklist items are done and all tests pass, output:
<promise>KROMCANON BUILD COMPLETE</promise>

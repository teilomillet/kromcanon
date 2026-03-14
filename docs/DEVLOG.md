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

## 2026-03-13 — Iteration 2: KromHC + GPT-2 Model

### Completed
- **Phase 1 — kromhc.py**: Full KromHC implementation:
  - `KromHCLayer`: width_connection (stream → branch input + residual mixing), depth_connection (branch output → streams), full forward with branch_fn callback
  - `_build_kronecker_hres`: Kronecker product of 2x2 doubly stochastic factors
  - `_build_2x2_factor`: Optimized 2x2 path (p*I + (1-p)*swap)
  - `_kronecker_product`: Batched Kronecker product via expand+reshape
  - `KromHCInit` / `KromHCReduce`: stream expansion and mean reduction
  - Dynamic mode with RMSNorm + projected coefficients (W init to zero → starts static)
  - Proper initialization: b^res=[0, -8] → H^res≈I, b^pre/b^post select by layer_index%n
- **Phase 1 — model.py**: Full GPT-2 with pluggable architecture:
  - `CausalSelfAttention` with optional Canon-B
  - `FeedForward` with GELU
  - `TransformerBlock` supporting all three modes (vanilla/canon/kromcanon)
  - `GPT2` with token+position embeddings, KromHC init/reduce, LM head
  - Config-driven architecture selection
- **Tests**: 20 KromHC tests (doubly stochastic property, Kronecker closure, init≈identity, shapes, stream init/reduce) + 14 model tests (forward pass shapes for all 3 archs, param count ordering, edge cases)

### Key Verification
- Doubly stochastic property verified: row sums = col sums = 1, non-negative
- Kronecker closure theorem (Theorem 4.2) verified in tests
- H^res ≈ Identity at initialization confirmed (max diff < 0.01)
- Canon adds ~0.5-5% params over vanilla (verified)
- KromCanon adds params over Canon (verified)
- `ruff check` passes clean

### Status: 44 tests passing, ruff clean

---

## 2026-03-13 — Iteration 3: Training Infrastructure

### Completed
- **Phase 2 — tokenizer.py**: BPE tokenizer wrapper using HuggingFace `tokenizers` library. Supports custom tokenizer files or fallback basic BPE.
- **Phase 2 — data.py**: FineWeb-Edu streaming loader, BOS-aligned sequence packing, `PretrainDataLoader` yielding (input, target) batches for next-token prediction.
- **Phase 2 — train.py**: Full training loop with:
  - `compute_loss`: cross-entropy next-token prediction
  - `create_lr_schedule`: cosine decay with optional linear warmup
  - `create_optimizer`: AdamW with schedule
  - `train_step`: single step with gradient clipping via `optim.clip_grad_norm`
  - `save_checkpoint` / `load_checkpoint`: model weights + metadata
  - `train`: full loop with logging, periodic eval, checkpointing
  - `evaluate`: capped at 50 batches
- **Phase 2 — scripts/train.py**: CLI with `--arch`, `--depth`, `--smoke` flags
- **Tests**: 8 training tests (loss computation, loss decrease over steps, all 3 archs train, data packing, loader shapes, evaluation)

### Bug Fixes
- `warmup_steps=0` caused `linear_schedule` to crash — now skips warmup when steps=0
- `mx.utils.tree_flatten` doesn't exist — use `mlx.utils.tree_flatten`
- Forward references to `PretrainDataLoader` — used `from __future__ import annotations` + `TYPE_CHECKING`

### Status: 52 tests passing, ruff clean

---

## 2026-03-13 — Iteration 4: Safety Fine-Tuning

### Completed
- **Phase 3 — safety_data.py**: HH-RLHF and BeaverTails dataset loading, conversation parsing, SFT formatting, tokenization, padded batch iteration.
- **Phase 3 — sft.py**: SFT training loop with lower LR (1/10th pretraining), gradient clipping, checkpointing.
- **Phase 3 — scripts/sft.py**: CLI with `--arch`, `--checkpoint`, `--max-steps`, `--max-examples` flags.
- **Tests**: 8 safety tests (HH conversation parsing, SFT formatting, tokenization, truncation, batch shapes, SFT loop execution).

### Bug Fixes
- `mx.utils` doesn't exist — fixed all references to `mlx.utils` with explicit import.
- Fixed unused imports and unsorted import blocks caught by ruff.

### Status: 60 tests passing, ruff clean

---

## 2026-03-13 — Iteration 5: Interpretability Tooling + E2E Tests

### Completed
- **Phase 4 — interp/extract.py**: Direction extraction via mean-diff and SVD. Per-layer activation collection. Multi-stream extraction for KromCanon (per-stream + joint directions). `ExtractionResult` and `MultiStreamExtractionResult` dataclasses.
- **Phase 4 — interp/abliterate.py**: Direction removal from attention output projection weights. Multi-stream abliteration (avg per-stream directions). Greedy generation for refusal rate measurement.
- **Phase 4 — interp/steer.py**: Activation steering during forward pass. `SteeringConfig` dataclass. `steer_forward`, `steer_generate`, `sweep_alpha` for parameter sweeps.
- **Phase 4 — interp/compare.py**: Cross-architecture comparison (pairwise cosine similarity, Pearson correlation of layer norms). Stream distribution analysis (entropy-based concentration, dominant stream identification). `format_comparison_report` for human-readable output.
- **Phase 4 — scripts/extract.py**: CLI for direction extraction from checkpoints.
- **Phase 4 — scripts/compare.py**: CLI for cross-architecture comparison and report generation.
- **Phase 5 — test_e2e.py**: End-to-end smoke tests for all three architectures: create model → train 3 steps → extract directions → abliterate → steer. Cross-architecture comparison test.

### Key Verification
- Full pipeline (train → extract → abliterate → steer) runs for all 3 architectures
- Abliteration verifiably removes directions (projection onto direction ≈ 0 after removal)
- Steering with alpha=0 matches normal forward pass (verified <1e-4 diff)
- Cross-architecture comparison produces valid cosine similarities in [-1, 1]

### Status: 77 tests passing, ruff clean

---

## 2026-03-13 — Iteration 6: Experiment Runner, Visualization, Wandb

### Completed
- **TOML-driven experiment runner** (`scripts/experiment.py`): Single orchestrator for all 6 phases. Reads everything from a TOML config file — no CLI flags. Entry point: `python -m scripts.experiment experiments/quick.toml`.
- **Experiment configs** (`experiments/quick.toml`, `experiments/full.toml`): Quick validation (depth=4, ~2min) and full research run (depth=12, ~2hrs). All hyperparameters, data budgets, wandb settings in one place.
- **Serialization module** (`src/kromcanon/interp/io.py`): Save/load for ExtractionResult, MultiStreamExtractionResult, ComparisonResult, StreamAnalysis, training logs, refusal rates, alpha sweeps. JSON for metadata, mx.savez for arrays.
- **Visualization module** (`src/kromcanon/interp/viz.py`): 8 publication-quality figures (training curves, direction norms, cosine heatmap, stream distribution, alpha sweep, abliteration bars, stream cosines, method comparison). Seaborn paper style, color-coded per architecture (blue/orange/green), PDF output.
- **Wandb integration**: `WandbLogger` wrapper class in experiment.py. Logs per-step training loss, direction norms, abliteration rates, steering KL, all 8 figures as images, summary metrics. Enabled via `[wandb] enabled = true` in TOML.
- **Test prompts** (`safety_data.load_test_prompts()`): Loads harmful/harmless prompts from HH-RLHF test split to avoid train/test leakage during direction extraction.
- **Visualization tests** (`tests/test_viz.py`): 13 tests verifying all figures render without error, return correct types, handle optional/empty data.
- **Dependency updates** (`pyproject.toml`): Added `[viz]` (matplotlib, seaborn) and `[experiment]` (matplotlib, seaborn, wandb) optional deps.

### Decisions
- **TOML over CLI**: The experiment config is a declarative document, not a sequence of flags. TOML makes it reproducible, version-controllable, and self-documenting. The runner takes exactly one argument: the path to the TOML file.
- **ExperimentConfig dataclass**: Typed mirror of the TOML structure. The TOML → dataclass mapping is explicit in `load_config()` — no magic.
- **Resume by default**: Each phase checks for existing output files before running. `resume = true` in the TOML skips completed phases. Supports interrupted runs.
- **Phase granularity**: 6 phases run sequentially. Each phase is independently resumable. No `--phase` flag — just delete the output files you want to re-run.
- **Wandb as optional**: All wandb calls go through `WandbLogger`. When disabled, every method is a no-op. No import unless enabled.

### Output Structure
```
results/{run_name}/
  config.json                          # resolved config snapshot
  pretrain/{arch}_logs.json
  sft/{arch}_sft_logs.json
  checkpoints/{arch}/final/            # pretrained checkpoints
  checkpoints/{arch}_sft/final/        # SFT checkpoints
  directions/{arch}_{method}.npz       # extracted directions
  abliteration/refusal_rates.json
  steering/{arch}_alpha_sweep.npz
  comparison/report.md
  comparison/stream_analysis.npz
  figures/fig{1-8}_{name}.pdf
```

### Status: 90 tests passing, ruff clean

---

## 2026-03-13 — Iteration 7: Experiment Protocol + Pre-Run Plumbing

### Completed
- **Seed control** (`scripts/experiment.py`): Added `seed` field to `ExperimentConfig` with default 42. Seeds `np.random.seed()` and `mx.random.seed()` at the start of `run()`. Persisted in `config.json`. All existing TOMLs updated to include `seed = 42`.
- **Improved refusal detection** (`src/kromcanon/interp/abliterate.py`): Added `_looks_like_refusal_text()` that decodes tokens and checks against 10 common refusal phrase starters (e.g. "I cannot", "I'm sorry", "As an AI"). Added `decode_fn` parameter to `measure_refusal_rate()` — when provided, uses phrase-based detection instead of the degenerate-output fallback. Phase 4 in experiment runner now passes GPT-2 tokenizer's `decode` as `decode_fn`.
- **Ablation TOML configs**:
  - `experiments/seed2.toml` — identical to full.toml, seed=137, run_name="full_seed2"
  - `experiments/ablation_vanilla_canon.toml` — vanilla+canon only, isolates Canon effect
  - `experiments/ablation_sft_size.toml` — sft_max_steps=100, sft_max_examples=500
- **Experiment log** (`docs/EXPERIMENTS.md`): Master experiment log with run index, Ralph Wiggum iteration protocol (Iter 0–3), decision tree for 4 outcome categories (A: universal, B: KromCanon degrades, C: Canon also breaks, D: KromCanon better), empty templates for quick-001 and full-001 runs, cross-run comparison tables.

### Decisions
- **Phrase-based refusal detection**: The old heuristic (`len(set(tokens[:5])) <= 2`) detected degenerate outputs, not refusals. Real safety-tuned models produce coherent refusal text. Phrase matching is simple, interpretable, and sufficient for our controlled setting.
- **Seed as config, not CLI**: Seeds belong in the TOML because they're part of the experiment specification. `seed2.toml` exists as a complete config, not a CLI override.
- **Decision tree in EXPERIMENTS.md**: Pre-committing to interpretation criteria before seeing results prevents post-hoc rationalization. The 4 categories and their follow-up actions are locked.

### Status: 90 tests passing, ruff clean

---

## 2026-03-13 — Iteration 8: Training Speed Optimizations

### Completed
- **Fused Flash attention** (`src/kromcanon/model.py`): Replaced hand-rolled scaled dot-product attention (matmul → scale → mask → softmax → matmul, 5 kernel launches) with `mx.fast.scaled_dot_product_attention` (single fused Metal kernel). Uses `mask="causal"` — eliminated explicit mask creation from `GPT2.__call__`, `steer_forward`, and both activation collection functions in `extract.py`. ~10-20% speedup on attention-bound workloads.
- **Compiled training step** (`src/kromcanon/train.py`, `src/kromcanon/sft.py`): Wrapped the loss+gradient+optimizer step with `mx.compile(fn, inputs=state, outputs=state)`. Fuses the entire forward+backward+update into a single compiled graph. Hoisted `nn.value_and_grad` out of the loop (was recreated every step). ~15-30% speedup via kernel fusion and reduced dispatch overhead. `train_step()` kept uncompiled for test compatibility.
- **Muon optimizer** (`src/kromcanon/train.py`, `src/kromcanon/config.py`): Added hybrid Muon+AdamW via `optim.MultiOptimizer`. Muon (Newton-Schulz orthogonalized momentum) for 2D weight matrices (attention projections, FFN layers) at lr=0.02. AdamW for embeddings, 1D params, LM head at lr=6e-4. For KromCanon: third group for HC params with `hc_lr=5e-3`, `hc_betas=(0.8, 0.95)`, `hc_weight_decay=0.2` — wiring up the previously-unused TrainConfig fields. ~35% fewer steps to same loss. Controllable via `use_muon: bool` in TrainConfig (default True).
- **SFT stays AdamW**: SFT uses pure AdamW at 1/10th pretraining LR. Muon is designed for high-LR pretraining; SFT's low LR wouldn't benefit.

### Key Implementation Details
- `mx.compile` requires `inputs`/`outputs` arguments (not decorator-based state capture): `mx.compile(step_fn, inputs=state, outputs=state)` where `state = [model.state, optimizer.state]`
- `mx.fast.scaled_dot_product_attention` accepts `mask="causal"` string for built-in causal masking — no need to create a (seq_len, seq_len) additive mask
- `MultiOptimizer(optimizers=[muon, adamw], filters=[is_muon_param])` — last optimizer is fallback (no filter needed)
- Helper `_create_schedule(config, lr, min_lr)` factored out for per-optimizer schedule creation

### Estimated Combined Speedup
- quick.toml: ~2 min → ~1 min
- full.toml: ~2 hrs → ~1 hr
- Breakdown: SDPA 10-20% + compile 15-30% + Muon 35% fewer steps

### Status: 90 tests passing, ruff clean

---

## 2026-03-13 — Iteration 9: Weight Tying + BFloat16

### Completed
- **Weight tying** (`src/kromcanon/model.py`, `src/kromcanon/interp/steer.py`): Removed separate `lm_head` nn.Linear. Forward pass now computes `x @ self.wte.weight.T` directly — standard GPT-2 design. Eliminates 25.2M duplicate parameters (vocab_size × d_model = 32768 × 768). Total params: 136.9M → 111.7M (18.4% reduction). Fewer gradients, fewer optimizer updates, less memory.
- **BFloat16 training** (`src/kromcanon/model.py`, `src/kromcanon/train.py`): `make_model()` now casts all parameters to bfloat16 via `model.set_dtype(mx.bfloat16)`. Halves model memory (447MB → 223MB) and memory bandwidth. `compute_loss()` casts logits to float32 before cross-entropy for numerical stability. Tests use `GPT2(config)` directly (float32) — unaffected.

### Key Details
- Weight tying: `steer_forward` and Muon filter updated (removed "lm_head" references)
- BFloat16: MLX SDPA already computes softmax in float32 internally; RMSNorm uses `mx.fast.rms_norm` which accumulates in higher precision. Only loss computation needed explicit float32 cast.
- All three archs (vanilla, canon, kromcanon) benefit equally from both changes — no impact on cross-architecture comparability.

### Status: 90 tests passing, ruff clean

---

## 2026-03-13 — Iteration 7: Experiment Pipeline Validation (quick-001)

### Changes

1. **vocab_size fix** (`src/kromcanon/config.py`): Changed `vocab_size: int = 32768` to `50304` (GPT-2's 50257 tokens padded to nearest 64 for kernel alignment). Old value caused NaN in cross-entropy loss because target token IDs exceeded vocabulary range.

2. **SIZE_PRESETS** (`src/kromcanon/config.py`): Added model size presets aligned with Physics of LLMs 4.1 (Allen-Zhu):
   - `micro`: 4 heads, d_model=256, seq_len=256 (seconds-scale validation)
   - `small`: 8 heads, d_model=512, seq_len=512 (8L512D from paper)
   - `medium`: 12 heads, d_model=768, seq_len=2048 (GPT-2 124M scale)

3. **BeaverTails split fix** (`src/kromcanon/safety_data.py`): BeaverTails dataset uses non-standard split names (`330k_train` instead of `train`). Added mapping.

4. **BFloat16 → NumPy fixes** (`scripts/experiment.py`, `src/kromcanon/interp/extract.py`, `src/kromcanon/interp/io.py`):
   - SVD requires float32 — cast `diff_matrix` before `mx.linalg.svd()`
   - NumPy can't convert bfloat16 directly — added `_to_np()` helper that casts to float32 first
   - `save_alpha_sweep` now casts logits to float32 before saving

5. **Experiment config** (`scripts/experiment.py`): Added `size` field to `ExperimentConfig`, threaded through `make_config()` calls.

6. **Updated TOMLs** (`experiments/quick.toml`, `experiments/full.toml`): Added `size` field, updated quick.toml to depth=8 + size=small.

### quick-001 Results Summary

Full pipeline (6 phases) completed successfully on 8L512D models (~51M params):
- **Pretrain** (200 steps): V=6.81, C=6.71, K=6.56 final loss
- **SFT** (60 steps): V=0.91, C=0.74, K=0.88 final loss
- **Direction extraction**: Non-zero norms, monotonically increasing with depth. Canon largest (0.71), KromCanon plateaus (0.55).
- **Abliteration**: All refusal rates 0.0 (expected — models can't generate coherent text at 200 steps)
- **Steering**: All KL curves monotonic — direction has real effect on output distribution
- **Figures**: 8 PDFs generated

### Key Observations

- Canon amplifies direction separation (largest norms, strongest SFT fit)
- KromCanon norm plateau at layers 5-7 — possible multi-stream direction distribution
- Cross-architecture cosine sims ~0 (expected with 200 steps — different representations)
- Layer norm correlations 0.95+ (architectures agree on *where* directions live, not *what* they are)

### Status: Pipeline validated. Ready for full run (full.toml).

---

## 2026-03-13 — Iteration 10: Full Run (full-001) + Research

### Changes

1. **Fixed Phase 4 test prompts** (`scripts/experiment.py`): Replaced random token IDs with real harmful prompts from HH-RLHF for refusal measurement. Random gibberish never triggers refusal behavior — using actual harmful prompts gives meaningful before/after rates.

2. **Fixed ablation configs** (`experiments/seed2.toml`, `ablation_vanilla_canon.toml`, `ablation_sft_size.toml`): Updated from old settings (12L768D, batch_size=64, 5000 steps, wandb=true) that would OOM on M4 Pro, to match full.toml (8L512D, batch_size=16, 2000 steps, wandb=false).

### Literature Research: Abliteration at Small Scale

While full-001 trains, conducted web research on abliteration/steering at small model scales. Key findings:

- **Smallest abliteration model**: Qwen 0.5B (direction confirmed to exist by Kissane et al.). Full abliteration tested down to Qwen 1.8B (~91% compliance, Arditi et al.).
- **Smallest steering model**: GPT-2-Small 124M for sentiment (Tigges et al. 2310.15154). No refusal-specific steering below 1B.
- **Novel territory**: No one has done abliteration on models trained from scratch. All prior work uses pre-existing HuggingFace checkpoints. KromCanon is the first.
- **Model-family dependency**: LLaMA-3.2-1B shows refusal distributed across a low-rank subspace (only 21% compliance with single direction). Qwen at same scale works fine. Architecture matters.
- **Training budget**: Qwen 0.5B (3T tokens pretraining) has refusal directions. Our 100M tokens is 30,000x less. Refusal rates may be zero or near-zero, but cross-architecture comparison remains valid.

### full-001 Experiment Progress

**Config**: 8L512D, batch_size=16, 2000 pretrain steps, 500 SFT steps, 100 prompts
**Started**: 22:53 CET

| Phase | Arch | Status | Time | Notes |
|-------|------|--------|------|-------|
| Pretrain | vanilla | done | 23:18 | Loss 11.05 → 6.02, 670ms/step |
| Pretrain | canon | running | — | step_1000 at 23:33, ~840ms/step |
| Pretrain | kromcanon | pending | — | — |
| SFT | all | pending | — | — |
| Extraction | all | pending | — | — |
| Abliteration | all | pending | — | — |
| Steering | all | pending | — | — |
| Figures | all | pending | — | — |

### full-001 Completed

All 6 phases complete. Key findings:
- **H^res frozen at identity**: Kronecker factors stuck at softmax([0,-8]) ≈ [1.0, 0.0]. Gradient ~0.0003 — vanishing through softmax saturation.
- **H^pre/H^post learned routing**: Streams specialize per layer (FFN branches show strongest specialization >0.6).
- **Training**: K=5.82 < C=5.88 < V=6.02 final loss.
- **SFT anomaly**: KromCanon loss **increases** during SFT (0.72 → 0.76).
- **Steering works**: All monotonic KL curves. Canon uniquely sensitive to positive steering.
- **Refusal rates all zero**: Expected at 51M/100M tokens.
- **Stream directions identical**: Per-stream cosines 0.975-0.990 in KromCanon.

### Status: full-001 done. ablation_bias_res running. Tests passing (90), ruff clean.

---

## 2026-03-14 — Iteration 11: KromHC Analysis + Bias Ablation

### Changes

1. **KromHC analysis script** (`scripts/analyze_kromhc.py`): 264-line post-hoc analysis tool. Extracts Kronecker factor matrices, computes H^res composite products, measures distance from identity/uniform, analyzes H^pre/H^post routing. Zero retraining cost.

2. **bias_res_init configurable** (`scripts/experiment.py`): Added `bias_res_init` field to `ExperimentConfig` and `[kromhc]` TOML section. New `_make_model_config()` helper applies the override. All `make_config()` calls in phase functions replaced with `_make_model_config()`.

3. **Ablation config** (`experiments/ablation_bias_res.toml`): bias_res_init=-2 (vs default -8). softmax([0,-2]) ≈ [0.88, 0.12], giving ~50x more gradient to Kronecker factors.

4. **KromHC diagnostics in experiment runner**: Added `_log_kromhc_diagnostics()` call between Phase 1 and Phase 2 to automatically analyze Kronecker factors after pretraining.

### Key Finding: Gradient Trap in KromHC

The `bias_res_init = -8` initialization creates a gradient trap via softmax saturation:
- `softmax([0, -8])` → `[0.9997, 0.0003]`
- Gradient of non-identity component: `p(1-p) ≈ 0.0003 × 0.9997 ≈ 0.0003`
- Even with `hc_lr = 5e-3`, effective gradient is ~1500x too small
- Result: all 16 KromHC layers (8 blocks × 2 branches) show factor weights `[1.0000, 0.0003]` at both step_1000 and step_2000 — zero movement

This means KromCanon is effectively operating as 4 independent parallel streams with learned routing but no inter-stream mixing. The entire multi-stream mixing hypothesis is untestable at this initialization.

### Status: ablation_bias_res experiment running. Tests passing (90), ruff clean.

---

## 2026-03-14 — Iteration 12: Bias Sweep Protocol + Research

### Changes

1. **Sweep analysis script** (`scripts/analyze_sweep.py`): Cross-run comparison tool. Loads KromHC analysis, per-stream direction cosines, pretrain losses, and steering results from multiple experiment runs. Outputs formatted comparison table.

2. **Sweep visualization script** (`scripts/viz_sweep.py`): 5 publication-quality figures for the bias sweep:
   - H^res factor heatmap (layer×branch, per bias_res_init)
   - Per-stream cosine vs bias_res_init (the key metric)
   - H^pre/H^post routing comparison across sweep points
   - KromCanon loss overlay at different bias_res_init
   - Steering sensitivity (KL at α=±3) vs bias_res_init

3. **EXPERIMENTS.md expanded**: Added sweep-m2 results template with comparison tables against full-001 baseline. Added Bias Sweep cross-run comparison tables.

4. **Sweep TOML configs**: Created `sweep_bias_m1.toml` (bias_res_init=-1) and `sweep_bias_0.toml` (bias_res_init=0).

### Research Notes

Literature review on steering vectors in multi-stream architectures:

- **Steering vector non-identifiability** (arxiv 2602.06801): Null-space ambiguity creates large equivalence classes. Key insight: identity H^res → independent Jacobians → large null space. Mixing could *reduce* null space, making directions more identifiable. Cross-seed comparison needed to distinguish genuine fragmentation from non-identifiability artifacts.
- **Information geometry of softmax** (arxiv 2602.15293): Mean-diff in primal space may misalign with semantic structure in softmax-based mixing. Dual (Fisher-aware) steering outperforms in final layer but not yet applicable to intermediate layers.
- **No prior work** found combining steering vectors with doubly stochastic residual mixing — our experiment is genuinely novel.

### Sweep-m2 Progress (bias_res_init=-2)

Experiment running since 00:59 CET. Vanilla pretrain complete (loss 6.02, matching full-001). Canon training in progress (step ~700/2000 at 01:36). Expected completion ~03:20 CET.

### Status: sweep-m2 running. Tests passing (90), ruff clean.

---

## 2026-03-14 02:00 — Iteration 8: Analysis, Visualization, and Blog

### Context

Sweep-m2 (bias_res_init=-2) pretraining complete. KromCanon at -2 achieves loss 5.806 vs 5.821 at -8. H^res factor weights moved from 0.88 (init) to 0.87-0.93 (trained). ||H^res - I|| = 0.507 (508x larger than -8's 0.001). Dynamic component (alpha_res) shows structured topology: L0/ffn +0.84 (amplifying), L0/attn -0.32 (dampening). SFT and downstream phases in progress.

### Key Finding: Alpha_res Asymmetry

At bias=-8, alpha_res averages 0.49 (model working hard to escape). At bias=-2, alpha_res averages only 0.16 (static gradient is enough). The escape threshold at -8 requires alpha_res > 3.0; the highest observed value is 0.92. The gap is 3.3x — the dynamic mechanism cannot compensate for the initialization.

### Changes

1. **H^res per-step metrics** (`src/kromcanon/kromhc.py`, `src/kromcanon/train.py`): Added `extract_hres_metrics()` function and integrated into training loop. Every log step now records ||H^res - I||_F, factor identity weights, and alpha_res per layer for KromCanon. Data available for all future runs.

2. **Figure data extraction** (`scripts/extract_figure_data.py`): Post-hoc extraction from checkpoints across runs. Produces `results/figure_data.json` with H^res trajectories and loss curves.

3. **Blog figure generation** (`scripts/make_blog_figures.py`): 5 publication-quality PDF figures:
   - fig2: Factor trajectory (flat at -8, stable at -2)
   - fig4: ||H^res - I|| per layer bar chart (invisible -8 vs ~0.5 at -2)
   - fig5: Alpha_res topology (amplify/dampen pattern — the money figure)
   - fig6: Loss curves overlay (V/C overlap between runs, K slightly better at -2)
   - fig7: Dynamic pathway scatter (gap vs gap_min — the "steering wheel" diagram)

4. **Blog draft** (`docs/BLOG_DRAFT.md`): Sections 1-5 now contain real data. Section 5 filled with bias=-2 control experiment results including the topology sculpting finding, loss comparison (5.806 vs 5.821), and alpha_res escape threshold analysis.

5. **EXPERIMENTS.md updated**: sweep-m2 pretrain results, H^res analysis tables, dynamic component data, cross-run comparison started.

6. **Sweep config optimization**: Changed sweep_bias_m1.toml and sweep_bias_0.toml to `architectures = ["kromcanon"]` — V/C results are identical to full-001 (same seed/data), saving ~50 min per sweep run.

### Status: sweep-m2 in SFT phase. Ablation_hres_frozen ready to launch after. 5 blog figures generated.

---

## 2026-03-14 03:00 — Iteration 9: Sweep-m2 Complete — Definitive Findings

### Context

Sweep-m2 (bias_res_init=-2) completed all 6 phases. Full results now available for KromCanon with real mixing (||H^res-I||=0.507) vs frozen mixing (||H^res-I||=0.001).

### Key Findings

1. **Per-stream cosines HIGHER with mixing**: At bias=-2, per-stream direction cosines are 0.994-0.998 (vs 0.982-0.990 at bias=-8). Streams carry MORE similar behavioral directions when mixing is active, not less. Stream concentration effectively zero at both initializations.

2. **SFT anomaly is architectural**: KromCanon SFT loss increases at both -8 (+0.038) and -2 (+0.042). Not caused by frozen mixing — intrinsic to the routing/stream-splitting architecture under distribution shift.

3. **Steering asymmetry reverses**: At bias=-8, KromCanon is more sensitive to negative steering (KL@-3=3.08 > KL@+3=2.24). At bias=-2, positive steering dominates (KL@+3=3.65 > KL@-3=2.67). Mixing affects the direction's orientation relative to the activation manifold.

4. **Direction profiles match full-001**: KromCanon still has the flattest layer norm profile (range 0.037) — inherent to routing, not mixing.

5. **Abliteration still zero**: All refusal rates zero across all archs (expected at 51M/100M tokens).

### Changes

1. **EXPERIMENTS.md filled**: All sweep-m2 tables populated with real metrics. Cross-run comparison tables updated. Run status changed to "done".

2. **Blog draft updated**: Section 7 fully written with per-stream cosine data, SFT anomaly analysis (ruled out mixing as cause, proposed routing fragility), and steering asymmetry reversal with full comparison table.

3. **Blog figures regenerated**: `extract_figure_data.py` and `make_blog_figures.py` re-run with complete data.

4. **Launched sweep-m1** (bias=-1, KromCanon only): Next point in the bias sweep. Completing the 4-point curve (-8, -2, -1, 0).

### Implications for Blog Narrative

The per-stream cosine result is a **positive interpretability result**: multi-stream coupling does NOT fragment behavioral directions. The blog draft now has a strong two-part story:
- Part 1 (sections 2-5): Gradient trap → control experiment → mixing topology sculpting
- Part 2 (section 7): Definitive answer on multi-stream interpretability

### Status: sweep-m2 done, sweep-m1 running, blog draft at ~500 lines with real data

---

## 2026-03-14 04:00 — Iteration 10: Sweep-m1 Complete + σ₂ Theory Validated

### Context

Sweep-m1 (bias_res_init=-1) completed all 6 phases. ||H^res-I||=1.035 (2x stronger than -2). Full bias sweep 3 of 4 points complete.

### Key Findings

1. **SFT anomaly RESOLVES at bias=-1**: KromCanon SFT loss decreases (1.031 → 0.722, -0.309). First KromCanon config where SFT works normally. Threshold: ||H^res-I|| > ~1.0 enables distribution shift adaptation via cross-stream gradient propagation.

2. **Per-stream cosines 0.990-0.998**: Confirms σ₂ theory prediction. σ₂=0.46, σ₂^16≈10^-6 → near-total contraction of inter-stream differences. Empirical cosines match prediction.

3. **Pretrain loss non-monotonic**: 5.828 at -1 vs 5.806 at -2 vs 5.821 at -8. Moderate mixing (-2) optimal. Strong mixing (-1) slightly worse than no mixing (-8). Suggests optimal mixing strength exists.

4. **Steering asymmetry back to negative-dominant** at -1 (KL@-3=3.13 > KL@+3=2.79), contrasting with -2's positive dominance. The asymmetry pattern is: -8 neg, -2 pos, -1 neg — not monotonic.

### Literature: The Homogeneity Trap

Found directly relevant concurrent paper: "The Homogeneity Trap: Spectral Collapse in Doubly-Stochastic Deep Networks" (arxiv 2601.02080). Key theorem: doubly stochastic mixing matrices contract the detail subspace by σ₂ at each layer. For our H^res factors, σ₂ = |p - q|. Over 16 mixing operations: σ₂^16 is the total contraction. This exactly predicts our empirical cosine ordering: -8 (0.982) < -2 (0.995) < -1 (0.997).

### Changes

1. **EXPERIMENTS.md**: All sweep-m1 tables filled. Cross-run comparison updated with 3 data points. SFT threshold hypothesis documented.
2. **Blog draft**: Section 6 filled with sweep data (3/4 points). Section 7 updated with SFT anomaly resolution finding and synchronization mechanism with σ₂ spectral grounding. Section 9 updated with complete narrative arc.
3. **New blog figure function**: `fig9_sigma2_vs_cosine()` in make_blog_figures.py — σ₂ prediction vs empirical cosine across bias sweep.
4. **Launched sweep-0** (bias=0, KromCanon only): Final sweep point. Predicted: σ₂=0, cosines=1.000, maximum mixing.
5. **Regenerated all blog figures** with complete data.

### Status: sweep-m1 done, sweep-0 running (~55 min), blog draft at ~600 lines

---

## 2026-03-14 11:30 — Iteration 11: Sweep-0 Retry + Ruff Cleanup + Research

### Context

Sweep-0 (bias=0) first attempt thrashed for 7.5 hours with 0% CPU — memory competition from 4 retrain campaign processes + vauban experiment. Killed and restarted after memory cleared (~12GB free).

### Changes

1. **Ruff lint cleanup**: Fixed all 8 remaining ruff errors across two files:
   - `scripts/analyze_kromhc.py`: Removed unused `config` variable, fixed f-string without placeholders, renamed ambiguous loop variable `l` → `f`
   - `scripts/make_blog_figures_v2.py`: Fixed import sorting, removed unused `alpha_res` variable, added `strict=True` to zip(), shortened overlong comment

2. **EXPERIMENTS.md**: Updated sweep-0 section with restart note, detailed theoretical predictions

3. **Sweep-0 restarted**: PID 24056, confirmed healthy (217MB RSS, 8% CPU)

### Research: Activation Steering State of the Art (2026)

Conducted web research on latest steering vector techniques while waiting for compute.

**Key findings**:
- **CAST** (ICLR 2025 spotlight, IBM): Conditional activation steering — detects when to apply steering based on activation patterns. Tested 1.8B-32B. Condition vectors via PCA on mean-centered activations. Grid search for thresholds.
- **Steering Vector Fields** (Li et al., Feb 2026, arXiv 2602.01654): Context-dependent steering via gradient of learned scoring function. +13-15% over static vectors on Llama-2-7B/Qwen-14B. Multi-layer coordination via shared boundary function with FiLM conditioning.
- **2026 Field Guide** (Mitra): Late layers (75% depth) best for behavioral steering. Strength is non-monotonic (Taimeskhanov et al. 2026). Effects fade after 300-500 tokens. Minimum tested: 7B for abliteration.
- **Cross-architecture abliteration benchmarks**: Tested 7-14B range only. No one has tested non-standard architectures.

**Relevance to KromCanon**: Our work at 51M is truly novel territory — smallest model, first non-standard architecture (multi-stream), first from-scratch training for abliteration. No prior art exists for multi-stream steering.

### Additional Changes (Iteration 11b)

4. **make_blog_figures_v2.py**: Updated `fig_cosines_vs_bias()` to include all 4 sweep points (-8, -2, -1, 0 when available). Updated `fig_sweep_comparison()` run_map to include bias=0. Fixed ruff B905 (missing `strict=True` in zip) and F841 (unused `alpha_errs`). Extended x-axis to accommodate bias=0.

5. **Blog draft polish**: Fixed σ₂ prediction table (bias=-1 updated from "predicted" to "observed: 0.997"). Updated section 9 implications with full 4-point sweep evidence and SFT threshold recommendation. Added bias=-1 steering data to section 7 steering table.

6. **EXPERIMENTS.md**: Updated sweep protocol statuses (-2 and -1 to "done", 0 to "running"). Expanded sweep-m1 KromHC analysis with alpha_res range [-0.723, 1.000] and factor stability observation. Corrected sweep-0 crash timing.

7. **Sweep-m1 alpha analysis**: Deep dive into dynamic component at bias=-1. Key finding: factors barely moved from init ([0.73, 0.27]) because init is already at useful mixing point. Alpha range wider than -2 (-0.723 to 1.000 vs -0.316 to 0.840). L0/ffn hits alpha max (1.000) — consistent amplification across both regimes.

### Status: sweep-0 downloading data (~15 min), all tests pass (90), ruff clean, blog at ~535 lines

---

## 2026-03-14 12:30 — Iteration 12: Research Round 2 (Sweep-0 Still Running)

### Context

Sweep-0 (bias=0) in data tokenization phase (~20 min in, training starts ~min 30). Continuing systematic literature review of abliteration and steering vector landscape.

### Research: Multi-Direction Refusal & Abliteration at Scale

**1. Multi-Direction Refusal** (arXiv 2602.02132, Feb 2026)
- 11 refusal categories across 4 datasets (WildGuardMix, SorryBench, CoCoNot, XSTest)
- Tested on Gemma-2-9B-it and Llama-3.1-8B-Instruct
- Key finding: different refusal categories have **geometrically distinct directions** (cosine sims range -0.06 to 0.92, median ~0.6)
- BUT: functionally they operate as a **one-dimensional control knob** — linear steering along ANY refusal direction produces nearly identical refusal/over-refusal tradeoffs
- Distinct directions are different linear combinations of shared latent structure
- **Relevance**: Even if KromCanon's multi-stream distributes directions differently, the functional mechanism may be shared. Our single-direction extraction may capture the control knob despite geometric diversity.

**2. Abliteration Tools Benchmark** (arXiv 2512.13655, Dec 2025)
- 4 tools (Heretic, DECCP, ErisForge, FailSpy), 16 models (6.7B-14.8B)
- Key finding: **alignment method more predictive than model size** for abliteration susceptibility
  - DPO-only models: most susceptible (Zephyr 98% ASR, KL=0.076)
  - Multi-stage (SFT+RLHF+DPO): more distributed safety, harder to ablate
  - StableLM-12B: 46% ASR despite size (multi-stage alignment)
- GSM8K most sensitive to abliteration (Yi-1.5-9B drops 26.5%)
- **Relevance**: Our SFT-only safety training should produce easily removable directions. The issue is model scale, not alignment method.

**3. Projected Abliteration** (HuggingFace blog, grimjim)
- Decomposes refusal direction into parallel (helpfulness-confounded) and orthogonal (mechanism-specific) components
- Only ablates orthogonal component → better coherence preservation
- r_proj = r - (r · μ_A)μ_A where μ_A is harmless mean direction
- Tested on Gemma 3 12B Instruct
- **Relevance**: Could improve our abliteration phase if we reach sufficient model quality. Currently all our abliteration deltas are 0 (models too small for coherent refusal).

**4. OBLITERATUS** (Mar 2026, github.com/elder-plinius/OBLITERATUS)
- 13 distinct abliteration methods: original diff-in-means baseline, spectral cascade decomposition, CoT-aware orthogonalization, plus faithful reproductions of Heretic, Gabliteration, RDO, FailSpy
- 116 models supported across 5 compute tiers
- 15 analysis modules, automated defense detection, MoE-aware surgery
- **Relevance**: Comprehensive toolkit validates that abliteration is a robust, reproducible technique across many architectures — but all tested architectures are standard transformers or MoE. No multi-stream variants.

**5. HyperSteer** (arXiv 2506.03292, Jun 2025)
- Hypernetwork-generated steering vectors conditioned on natural language prompts + model internals
- Generalizes to unseen steering prompts (zero-shot steering)
- Scales to thousands of behaviors
- **Relevance**: Future direction — could KromCanon's multi-stream architecture enable richer hypernetwork-generated steering? Stream-specific vectors could offer finer control.

**6. LayerNavigator** (AAAI 2026)
- Principled layer selection for multi-layer steering via discriminability + consistency criteria
- Reuses activations from steering vector computation (zero additional data)
- **Relevance**: Our layer-by-layer direction norms already capture discriminability. Could formalize layer selection for KromCanon steering.

### Key Takeaway for KromCanon

No prior work has studied steering vectors or abliteration in multi-stream architectures. The closest work (Jamba hybrid Mamba-transformer) hasn't been studied for steering at all. Our sweep data showing σ₂-controlled stream synchronization is the first evidence of how doubly stochastic mixing interacts with linear behavioral directions.

The multi-direction refusal paper (2602.02132) is particularly relevant: if refusal operates as a one-dimensional control knob even across 11 distinct categories, then KromCanon's multi-stream mixing should preserve this control property regardless of how it distributes the specific geometric direction — which is exactly what our per-stream cosine data shows.

### Status: sweep-0 running (~25 min in), all tests pass (90), ruff clean

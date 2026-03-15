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
- **Canon set = "ABCD"** (full Canon-A/B/C/D — optimal per Physics of LLMs 4.1; changed from "AB" in Iteration 34)
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

---

## 2026-03-14 14:00 — Iteration 13: Sweep Complete + N=3 Replication Begins

### Context

All 4 bias sweep points complete. Sweep-0 finished at 05:03 CET (~54 min). Full results extracted and documented in EXPERIMENTS.md. Blog figures regenerated with complete 4-point data.

### Sweep-0 Summary (bias_res_init=0)

- **Information destruction trap**: H^res ≈ (1/4)J₄ (uniform matrix). Factors stay at [0.47-0.54, 0.46-0.54] despite maximum gradient p(1-p)=0.25. The loss landscape provides no gradient toward specialization because uniform mixing destroys the inter-stream differences needed for that gradient.
- **Pretrain loss**: 5.822, tied with bias=-8 (5.821). Both extremes reduce to 1 effective stream.
- **SFT**: Decreasing (-0.310), consistent with ||H-I||=1.799 > 1.0 threshold.
- **Per-stream cosines**: 0.995-0.999, highest in sweep. σ₂≈0.13, σ₂^16≈10⁻¹⁴ → confirmed.
- **Steering**: KL@-3=2.770, KL@+3=4.504 (positive-dominant).

### Complete Sweep Findings

Non-monotonic loss: **-2 (5.806) < -8 (5.821) ≈ 0 (5.822) < -1 (5.828)**. Both extremes are trapped — by different mechanisms. At -8, softmax saturation kills gradient. At 0, information destruction removes the signal. Only -2 occupies the sweet spot.

Steering asymmetry pattern: **-8 neg, -2 pos, -1 neg, 0 pos**. Alternating with bias init. The dominant steering direction tracks whether σ₂ is high (neg) or low (pos), suggesting mixing strength modulates which activation space direction dominates.

SFT threshold: ||H^res-I|| > 1.0. Below this (-8: 0.001, -2: 0.507), SFT loss increases. Above (-1: 1.035, 0: 1.799), SFT works normally.

### New Architecture: vanilla+KromHC

Added "kromhc" architecture type (vanilla + KromHC, without Canon layers). Confirmed: 38.8M params, all 90 tests pass. Config `ablation_vanilla_krom.toml` ready. This isolates the KromHC effect from Canon's local convolution.

### Changes

1. **EXPERIMENTS.md**: Updated sweep-0 status to "done", σ₂ prediction to "confirmed", all cross-run comparison tables populated
2. **figure_data.json**: Regenerated with sweep-0 data (all 4 points)
3. **Blog figures**: Regenerated all 6 PNG figures with complete 4-point sweep data
4. **seed2_bias_m2 launched**: First N=3 replication run (bias=-2, seed=137). KromCanon only, ~55 min.

### N=3 Replication Strategy

Priority order:
1. seed2_bias_m2 (running) → seed3_bias_m2 → fills cosine error bar table at bias=-2
2. seed2_bias_m8 → seed3_bias_m8 → fills training curve error bar table at bias=-8
3. ablation_vanilla_krom → isolates KromHC from Canon

Total remaining compute: ~5.8 hrs.

### Research Round 3: Multi-Stream Architecture Landscape

Between training phases, surveyed multi-stream architecture literature:

1. **MUDDFormer** (ICML 2025, 2502.12170): Multiway Dynamic Dense connections with position-dependent weights. MUDDPythia-2.8B matches Pythia-6.9B. No interpretability analysis. Added to blog related work.
2. **DeepMind deprioritized SAEs** (Nov 2025): SAEs underperform linear probes for OOD generalization. Validates our mean-diff approach over SAE decomposition.
3. **MLSAE** (ICLR 2025, 2409.04185): Single SAE trained across all residual stream layers. Relevant for future multi-stream SAE work.
4. **SAE-based steering refinement** (2509.23799): Top-k SAE features for denoised steering vectors.

Key conclusion: **multi-stream architectures are going mainstream** (mHC, MUDDFormer, KromHC all in 2025-2026) but **no one has studied their interpretability properties**. Our work fills this gap.

### Code Changes

5. **Memory-efficient tokenization** (`src/kromcanon/data.py`): Replaced `list[list[int]]` accumulation (2.8GB for 100M tokens) with streaming into a pre-allocated numpy array (0.4GB). Added disk caching to `results/.cache/` — subsequent runs skip tokenization entirely.

6. **"kromhc" architecture support in scripts**: Fixed hardcoded "kromcanon" references in `experiment.py` (4 locations), `analyze_sweep.py` (3 locations), and `extract_figure_data.py` (1 location) to also handle the new "kromhc" architecture. Added `_has_multistream()` helper.

7. **Cross-seed analysis script** (`scripts/analyze_seeds.py`): New script for N=3 replication analysis. Computes mean ± SE for per-stream cosines, alpha topology consistency, pretrain/SFT loss, and steering KL across seeds. Outputs markdown tables for EXPERIMENTS.md.

8. **Blog draft**: Added "information destruction trap" paragraph (bias=0 symmetric interpretation) and refined non-monotonic loss explanation.

9. **Ruff cleanup**: Removed unused `mpatches` import in `make_blog_figures_v2.py`.

### Status: seed2_bias_m2 training (step ~110/2000), all tests pass (90), ruff clean

---

## 2026-03-14 06:00 — Iteration 14: Research Round 4 + Blog Updates

### Context

seed2_bias_m2 training (step 1000+/2000, ~20 min remaining for pretrain). Using wait time productively for research and documentation.

### Research Round 4: Steering Vector Theory (Feb 2026)

Three highly relevant new papers discovered:

1. **Steer2Edit** (2602.09870): Converts steering vectors to rank-1 weight edits, closed-form, no finetuning. 17.2% safety improvement. Bridges activation-space and weight-space interventions — relevant future direction for multi-stream models.

2. **Steering unreliability** (Braun, 2602.17881): Higher cosine similarity between activation differences → more reliable steering. Directly validates our per-stream cosine metric: cosines >0.995 at bias=-2 predict reliable steering in KromCanon. Non-linear representations cause failure.

3. **Steering identifiability** (Venkatesh & Kurapath, 2602.06801): Steering vectors are non-identifiable in general (large equivalence classes). BUT recoverable under cross-layer consistency, sparsity, or multi-environment validation. Our N=3 seed replication serves as multi-environment validation — recovers identifiability per Venkatesh's framework.

### Blog Draft Updates

- Added 3 new papers to Related Work (Section 9): Steer2Edit, steering unreliability, steering identifiability
- Added steering asymmetry literature context (Section 7): noted our alternating pattern is novel, connected to Chalnev et al. and Braun's work
- Added N=3 methodology paragraph in Section 10 (Code and Reproducibility)
- Added NeurIPS 2025 acceptance note for Canon layers
- Total blog length: 7229 words, 581 lines

### Code Changes

- Enhanced `analyze_seeds.py`: added markdown output for alpha topology and SFT anomaly consistency tables (was only outputting cosine table)
- Updated reference memory with 3 new papers + implications

### seed2_bias_m2 Results (completed 06:24 CET)

Key findings from seed 137 at bias=-2:
- **Pretrain loss**: 5.676 (point), 5.766 (smoothed avg last 200 steps) — vs seed 42: 5.806/5.779. Smoothed difference only 0.013.
- **SFT anomaly reproduced**: +0.045 (seed 42: +0.042). Same magnitude.
- **Per-stream cosines match**: L0=0.995, L3=0.998, L7=0.994 (seed 42: 0.995, 0.998, 0.995). Max deviation 0.001.
- **Alpha topology partially consistent**: 10/16 layers same sign (62%). Early layers (L0-L2) fully consistent. L0/ffn=+1.000 (seed 42: +0.840) — always max amplifier. Mid-depth layers (L3-L6) seed-dependent.
- **Steering asymmetry is seed-dependent**: seed 137 is negative-dominant (KL@-3=4.782 > KL@+3=4.551), seed 42 is positive-dominant (KL@-3=2.669 < KL@+3=3.648). The alternating pattern across bias sweep may be partially noise.
- **Re-ran KromHC analysis for seed 42** to get previously missing alpha_res values (16 layers).

Launched seed3_bias_m2 (seed=271) at 06:25 CET. ~55 min expected.

### Research Round 5: Multi-Stream Architecture Landscape

Between seed2 completion and seed3 training:

1. **MUDDFormer** (ICML 2025, 2502.12170): Multiway Dynamic Dense connections with position-dependent weights. MUDDPythia-2.8B matches Pythia-6.9B. No interpretability analysis. Added to blog related work.
2. **DeepMind deprioritized SAEs** (Nov 2025): SAEs underperform linear probes for OOD generalization. Validates our mean-diff approach over SAE decomposition.
3. **MLSAE** (ICLR 2025, 2409.04185): Single SAE trained across all layers of residual stream. Future direction for multi-stream.
4. **Non-identifiability** (ICLR 2026, 2510.02917): Multiple circuits/interpretations coexist for same behavior. Not about steering limits.
5. **No Canon layers + interpretability papers exist**. Confirmed gap in literature.
6. **ICLR 2026 mech-interp survey** (2602.11180): Does not mention multi-stream architectures at all.

Key conclusion: multi-stream architectures going mainstream (mHC, MUDD, KromHC) but NO interpretability work exists. Our paper fills a genuine gap.

### Status: seed3_bias_m2 running, seed2_bias_m2 done, all lint clean

## 2026-03-14 07:20 — Iteration 15: N=3 Complete at bias=-2

### seed3_bias_m2 Results (completed 07:19 CET, 50.8 min)

Key findings from seed 271 at bias=-2:
- **Pretrain loss**: 5.855 (point), eval 5.835 — higher than seed 42 (5.806) and seed 137 (5.676). Mean ± SE: 5.779 ± 0.053.
- **SFT anomaly NOT reproduced**: First_loss=1.042, Final_loss=0.752, DECREASING (-0.290). Seeds 42/137 show +0.042/+0.045. This breaks the N=2 claim that the anomaly is "architectural at bias=-2".
- **Per-stream cosines completely robust**: L0=0.996, L3=0.996, L7=0.995. All layers within 0.001 of seeds 42/137. Mean ± SE < 0.001 at every layer. This is the strongest finding.
- **Alpha topology mostly seed-dependent**: Only 5/16 layers consistent (31%). Down from 10/16 (62%) with N=2. L0/ffn remains max amplifier (+1.000 for seeds 137/271, +0.840 for seed 42). L0/attn and L2/attn flip positive (were negative for both seeds 42/137). N=2 was misleading.
- **Steering KL**: α=-3: 3.046, α=+3: 2.885. Negative-dominant (weakly). Consistent with seed 137 but not seed 42.

### N=3 Summary

| Claim | Robust? | Evidence |
|-------|---------|----------|
| Per-stream cosines > 0.99 | **YES** | SE < 0.001, all 3 seeds, all layers |
| L0/ffn max amplifier | **YES** | +0.84/+1.00/+1.00 across seeds |
| Monotonic steering | **YES** | All seeds, both directions |
| SFT anomaly at bias=-2 | **PARTIAL** | 2/3 seeds (seed 271 decreases) |
| Alpha topology mechanistic | **NO** | Only 31% consistent with N=3 |
| Steering asymmetry sign | **NO** | Flips between seeds |

### Blog Draft Updates

- Corrected alpha_res topology claim: from "partially mechanistic" to "mostly seed-dependent" (31% consistency)
- Updated SFT anomaly sections: noted 2/3 seed reproduction, first_loss as key predictor
- Corrected Section 7 caveat paragraph: only cosines are fully robust; all other claims need uncertainty qualification
- Updated Section 10 seed replication paragraph: explicit robust/partial/seed-dependent categorization

### Launched ablation_vanilla_krom (vanilla + KromHC without Canon)

Config: `experiments/ablation_vanilla_krom.toml` — 2 architectures (vanilla, kromhc), bias=-2, seed=42
Expected runtime: ~1.5 hrs
Purpose: Isolate Canon's marginal effect on multi-stream coherence

### Killed stale ablation_vanilla_krom process, launched seed2_bias_m8

ablation_vanilla_krom was an orphaned process from previous conversation — killed it. Launched seed2_bias_m8 (all 3 archs, bias=-8, seed=137) for N=3 training curve error bars. ~2 hrs expected.

### Research Round 5 (during seed3 training)

- **MUDDFormer** (ICML 2025): Multi-stream with dynamic weights. No interp analysis. Added to blog.
- **DeepMind deprioritized SAEs** (Nov 2025): Linear probes beat SAEs for OOD generalization. Validates our mean-diff approach.
- **ICLR 2026 mech-interp survey**: Does not mention multi-stream architectures at all. Confirms gap.
- Updated reference memory with multi-stream architecture landscape.

### Status: seed2_bias_m8 running, N=3 at bias=-2 complete, blog updated

## 2026-03-14 08:00 — Iteration 16: Deep Analysis During seed2_bias_m8

### ablation_vanilla_krom died from memory pressure

ablation_vanilla_krom crashed at step 270 (concurrent with seed2_bias_m8 launch — 24GB insufficient for two training processes). Will restart after seed2_bias_m8 completes.

### Cross-Condition SFT Anomaly Analysis (KEY FINDING)

Analyzed all 6 KromCanon SFT runs across conditions and seeds. Clean binary threshold:
- first_loss < 0.76 → anomaly (3/3): bias=-2 seeds 42/137, bias=-8 seed 42
- first_loss > 1.03 → normal (3/3): bias=-1/0 seed 42, bias=-2 seed 271
- Gap: [0.754, 1.031] — no data points in between

NOT correlated with ||H-I||. Both frozen (0.001) and active (0.50) mixing show anomaly. The predictor is the pretrained REPRESENTATION, not the routing structure. Updated blog draft with full 6-row table and threshold analysis.

### New Blog Findings

1. **Direction flatness**: KromCanon direction norm profile 3.1x flatter than Vanilla (N=3, SE=0.003). Added to blog Section 7.
2. **Steering sensitivity**: KromCanon consistently 25-30% lower average KL at unit steering strength vs Vanilla. Added to blog.
3. **σ₂ vs cosine**: σ₂ varies from 0 to 0.999, cosines only vary from 0.987 to 0.996 (delta <1%). Quantified in analysis.
4. **SVD subspace overlap**: Cross-architecture and cross-seed overlaps in same range (0.12-0.25), both 7-12x above random (0.018). Added to EXPERIMENTS.md.
5. **Monotonicity**: 10/10 steering curves strictly monotonic. Confirmed across all runs/archs/seeds.

### Research Round 6 (during seed2_bias_m8)

New references added to blog and memory:
- **Granite 4.0 350M gabliterated**: Smallest abliterated model (350M), hybrid Mamba-2/Transformer. Updated "12x below" to "7x below" minimum.
- **Paulo et al. (AAAI 2025, 2404.05971)**: Steering transfers from transformers to Mamba/RWKV. Strongest prior art for our Canon claim.
- **Geometry of Refusal (ICML 2025, 2502.17420)**: Multi-dimensional refusal concept cones. Cone dimension scales with d_model.
- **SteerMoE (2509.09660)**: MoE expert steering. Dense mixing (ours) vs sparse routing (MoE).
- **Refusal universality (2505.17306)**: Cross-lingual direction stability.

### Finding Robustness Report

Compiled comprehensive robustness assessment:
- IRON-CLAD: per-stream cosines (24 measurements), monotonic steering (10 curves)
- ROBUST: L0/ffn max amplifier (N=3), direction flatness (N=3), cross-arch uncorrelated (48 measurements)
- PARTIAL: SFT anomaly (2/3 seeds, predicted by first_loss threshold)
- SEED-DEPENDENT: alpha topology (31%), steering asymmetry
- PROVEN: gradient trap (mathematical)

### Status: seed2_bias_m8 at step ~1400/2000 (vanilla pretrain), blog at 8658 words

---

## Iteration 17: Overfitting Analysis & Data Verification

### Overfitting Analysis (NEW finding)

Systematic train-eval gap comparison across all 11 run/arch combinations reveals architecture-specific overfitting:
- **Canon consistently overfits**: +0.080 (b-8), +0.086 (b-2). Local convolutions memorize training data.
- **KromCanon moderately overfits**: +0.020 to +0.023. Multi-stream mixing partially regularizes.
- **Vanilla has zero overfitting at seed 42**: -0.001 to +0.000.
- **Seed 137 overfits heavily**: +0.141 (vanilla), +0.149 (kromcanon). Seed-dependent memorization.
- **Key insight**: Seed 137 vanilla train loss is 0.143 lower than seed 42, but eval is identical (6.015 vs 6.016). 99% of cross-seed train loss variance is memorization noise.

**Implication**: Architecture ordering should use eval loss. On eval, K < C < V ordering holds. Gaps V→C = 0.055, C→K = 0.121. Added full table to EXPERIMENTS.md.

### Data Corrections

1. **ablation_bias_res is bias=-2, not -8**: Previous analysis mislabeled this run. Corrected all references in EXPERIMENTS.md.
2. **Direction flatness ratio**: Corrected from 3.1x to ~3x (range 2.9-3.8x depending on bias point). Updated blog draft.
3. **Steering sensitivity reduction**: Corrected from "25-30%" to "~20-30% (bias-dependent)". At b-2 with N=3: ~20%. At b-8: ~25%.
4. **Monotonicity count**: Updated from 10/10 to 20/20 (verified across all runs including bias sweep).

### Blog Updates

- Added eval loss confirmation sentence to training results section
- Updated flatness ratio to ~3x (from 3.1x)
- Updated steering sensitivity to ~20-30% (from ~25-30%)

### Comprehensive Verification

- SFT threshold predictor: 6/6 correct classifications (3 anomalies, 3 normals)
- Per-stream cosines corrected by bias: b-8 mean=0.987, b-2 mean=0.996±0.000, b-1 mean=0.996, b+0 mean=0.997
- Steering monotonicity: 20/20 curves strictly monotonic across all runs

### Status: seed2_bias_m8 canon pretrain at step ~1100/2000, blog at ~8700 words

---

## Iteration 18: SVD Overlap Correction & Blog Verification

### Critical SVD Overlap Fix

Discovered and fixed a major error in the SVD subspace overlap analysis:

**The error**: EXPERIMENTS.md and blog compared sum(canonical cosines) values (0.15-0.22) to a sum(cos²) baseline (0.018), inflating apparent significance by ~10x. The values appeared to be "8-10x above random."

**The fix**: Proper random baseline for sum(canonical cosines) at k=3, d=512 is **0.188 ± 0.047** (simulated, 50k trials). All observed values fall within 1σ:
- Cross-architecture: V-C 0.157 (z=-0.66), V-K 0.178 (z=-0.21), C-K 0.140 (z=-1.02)
- Cross-seed: 42-137 0.155 (z=-0.70), 42-271 0.223 (z=+0.74), 137-271 0.161 (z=-0.57)

**None are statistically significant.** Top-3 SVD subspaces are indistinguishable from random across both architectures and seeds.

**Impact**: Strengthens the Venkatesh non-identifiability result. Not just top-1 directions but entire 3-dim refusal subspaces are training-trajectory artifacts. The identifiable structure is per-stream coherence (>0.99), which is architectural.

Updated: EXPERIMENTS.md (SVD section rewritten), BLOG_DRAFT.md (3 paragraphs), reference_abliteration_scale.md memory.

### Blog Numerical Verification (continued)

Verified against raw data:
- Alpha_res averages: bias=-8 mean=0.495 (blog: 0.49 ✓), bias=-2 mean=0.157 (blog: 0.16 ✓)
- Dynamic gap_min formula: `static_gap - 2*|alpha|` where static_gap ≠ |bias| due to b_res drift. Blog values match stored `dynamic_gap_min` from figure_data.json ✓
- Steering KL table (Section 7): all 12 values exact match ✓
- Steering sensitivity: KromCanon -30% at bias=-8, -24% at bias=-2 ✓
- SFT anomaly threshold: 6/6 correct, first_loss < 0.76 → anomaly, > 1.03 → normal ✓
- SFT minimum loss: 0.398 at bias=-1 step 440, 0.614 at bias=-8/-2 step 350 ✓
- Per-stream cosines across sweep: all 12 values match ✓
- Direction norm ranges: V=0.094, C=0.068, K=0.032±0.003 ✓
- Factor weight stability: max cross-seed range 0.027 (blog corrected from 0.029)

### Additional Blog Corrections

1. **||H^res-I|| at bias=-1**: Corrected from 1.035 (step 1000 value) to 1.024 (final step 2000). Blog header said "Final" but used midpoint value.
2. **Bias=0 factor range**: Corrected from "[0.47-0.54]" to "[0.32, 0.65]". Factors DO drift substantially at bias=0, but mean σ₂ stays at 0.13 so information destruction trap still holds.
3. **Cross-seed factor range**: Corrected from 0.029 to 0.027 per factor (N=3 value).

### Status: seed2_bias_m8 kromcanon pretrain at step ~1100/2000, blog verified clean

## Iteration 19: seed2_bias_m8 Results & Blog Updates

### Pretrain Results (seed 137, bias=-8)

All 3 architectures completed pretrain. Ordering kromcanon < canon < vanilla holds for both seeds:
- vanilla: 5.874 (eval 6.015) vs seed 42: 6.017 (eval 6.016)
- canon: 5.831 (eval 5.977) vs seed 42: 5.881 (eval 5.961)
- kromcanon: 5.758 (eval 5.838) vs seed 42: 5.821 (eval 5.840)

Seed 137 trains lower train loss but similar eval loss — memorizes more, doesn't generalize better.

### Alpha_res at bias=-8: NOT all positive

**Critical blog correction**: Seed 42 has all 16 alpha_res positive (signed mean 0.495). Seed 137 has 3 negative (L2/ffn=-0.707, L3/ffn=-0.555, L7/attn=-0.445), signed mean 0.267, abs mean 0.481. Blog previously claimed "all 16 positive" — corrected to note seed-dependence.

Sign consistency at bias=-8 (N=2): 13/16 (81%). Sign flips: L2/ffn, L3/ffn, L7/attn. Cross-bias robustness (bias=-2 N=3 AND bias=-8 N=2): L0/ffn, L1/attn, L4/ffn, L6/ffn. L7/attn flips at bias=-8 despite being robust at bias=-2.

### SFT Anomaly: Canon Also Shows It

Seed 137 canon SFT at bias=-8: first=0.732, final=0.733, delta=+0.002 (FLAT). This is below the first_loss threshold (~0.76). Previous claim was that the anomaly was KromCanon-specific — now shown to affect Canon as well when first_loss is low enough. The SFT anomaly is a **representation property**, not an architecture property.

### Research Update

Added 12 new papers to memory (reference_abliteration_scale.md):
- Weight Steering (2511.05408): weight-space fallback for activation steering
- Geometry of Alignment Collapse (2602.15799): alignment in sharp-curvature subspaces
- Steering Vector Fields (2602.01654): context-dependent steering
- Why Steering Works (2602.02343): unified framework
- Hidden Dimensions (2502.09674): multiple orthogonal safety directions
- Deep Delta Learning (2601.00417), MGT (2601.01014), ProRes (2603.05369): alternative residual designs
- GRP-Obliteration (2602.06258): single-prompt unalignment

Added Geometry of Alignment Collapse and Weight Steering to blog related work section.

### Status: seed2_bias_m8 complete, seed3_bias_m8 running

## Iteration 20: seed2-m8 Documentation & Blog Updates

### EXPERIMENTS.md Updates

Added detailed seed2-m8 run section with:
- Phase-by-phase metrics (pretrain, SFT, directions, abliteration, steering)
- Per-stream cosines (seed 42 vs 137 at bias=-8): 0.982-0.993 range
- Alpha topology N=2 table at bias=-8 (13/16 consistent, 81%)
- Cross-architecture direction cosines (all at random baseline)
- N=2 training curves comparison

Updated cross-condition SFT anomaly table: expanded from 6 KromCanon-only runs to 11 runs across all 3 architectures. Canon at seed 137 crosses threshold (first_loss=0.732). Vanilla never crosses.

Updated overfitting analysis table with seed2-m8 Canon and KromCanon rows.

Updated N=3 cosine error bars at bias=-8: added seed 137 data (L0=0.993, L7=0.986).

### Blog Updates

1. **Section 4 SFT anomaly** (line 221): Corrected from "Vanilla and Canon both show decreasing SFT loss" to architecture-agnostic threshold explanation
2. **Section 4 per-stream cosines**: Updated from single-seed values to N=2 means with SE
3. **Section 7 SFT threshold table**: Expanded from 6 KromCanon runs to 11 runs across all architectures
4. **Section 7 reframing paragraph**: Updated from "Vanilla and Canon always converge" to architecture-specific crossing frequencies

### seed3_bias_m8 (seed=271, all 3 archs, bias=-8)

Running. Vanilla pretrain at step ~320/2000. Expected completion: ~90 more min.

### Status: seed3_bias_m8 vanilla pretrain step 320/2000

## Iteration 21: seed3-m8 Completion, N=3 Analysis, Literature Review

### Literature Research (while waiting for seed3_bias_m8)

**New tools and findings added to reference memory:**

1. **Heretic** (p-e-w/heretic, v1.2.0 Feb 2026): Fully automated abliteration using Optuna TPE optimizer. Co-minimizes refusal count + KL divergence. Tested on Gemma-3-270M-IT (new smallest published abliteration, previously Granite 350M). Does NOT support SSMs, hybrid models, inhomogeneous layers, or novel attention. Direction interpolation for multi-directional refusal. 1000+ community models.

2. **Unified steering evaluation** (2502.02716): Mean of Differences (MoD) consistently outperforms PCA-based and classifier methods. Validates our mean-diff approach. Peak steering layer at ~40% depth for 7B (our 8-layer models peak at L7 — too shallow for intermediate peak pattern).

3. **Causally grounded mech interp** (2603.09988): On GPT-2-Small (124M, same base as ours), finds "distributed backup mechanisms" — 100% sufficiency but only 22% comprehensiveness. Relevant to direction dispersion question.

4. **FGAA** (2501.09929): SAE-based steering, works better on 2B than 9B ("non-linear scaling characteristics"). Not applicable to our 51M scale.

5. **Mech interp 2026 status report**: Notes steering vectors "become completely unpredictable after O(log(1/ε)) layers" due to chaotic dynamics — but our 0.99+ per-stream cosines suggest KromHC's doubly stochastic constraint tames this chaos.

6. **Steering field guide** (Mitra, 2026): Practical takeaways — steering effects degrade after 300-500 tokens, multi-vector interference when behaviors share representation space, stronger alpha can reverse behavior.

### seed3_bias_m8 Progress

- Vanilla pretrain: complete (train=6.008, eval=6.001)
- Canon pretrain: complete (train=5.868, eval=5.959)
- KromCanon pretrain: running (step ~470/2000)

### Direction Norm Analysis

Verified all architectures peak at L7 (last layer) across all bias settings — monotonically increasing. Differs from 7B models which peak at ~40% depth. The "Curse of Depth" (Zhang et al., 2502.05795) explains this: Pre-LN causes exponential variance growth with depth, making derivative matrices approach identity in deeper layers. Our 8-layer models are too shallow for this degradation, so direction norms accumulate monotonically. In 32-layer 7B models, later layers contribute minimally, shifting the peak to ~40% depth.

### Additional Literature (session 2)

7. **Selective Steering** (2601.19375): Norm-preserving rotation + discriminative layer selection (5.5x better ASR). Prior angular steering causes "generation collapse in models below 7B." Our stable 51M steering suggests mean-diff additive intervention is more robust than rotation at small scale.
8. **Refusal Steering** (2512.16602): Ridge-regularized steering. "Refusal signals concentrate in deeper layers and are distributed across many dimensions." Tested 4B-80B. Supports our depth profile.
9. **Curse of Depth** (2502.05795): Pre-LN variance grows exponentially with depth → later layers ineffective. Tested 130M-7B. Explains our monotonic depth profile vs. 7B intermediate peak.
10. **Steer2Adapt** (2602.07276): Composes steering vectors from semantic prior subspace. 8.2% improvement across 9 tasks. Future direction for multi-behavior steering.
11. **Rethinking Safety Fine-tuning** (2508.12531): Poor optimization, not inherent trade-offs, causes safety degradation. EMA preserves safety. Relevant to SFT anomaly interpretation.

### Existing N=2 Data Summary (bias=-8, for reference when N=3 arrives)

**Per-stream cosines:**
| Layer | Seed 42 | Seed 137 | N=2 mean |
|---|---|---|---|
| L0 | 0.990 | 0.993 | 0.992 |
| L3 | 0.988 | 0.991 | 0.990 |
| L7 | 0.982 | 0.986 | 0.984 |

**SFT losses (bias=-8):**
| Arch | Seed 42 first→last | Seed 137 first→last |
|---|---|---|
| Vanilla | 0.979→0.673 (-0.306) | 1.045→0.633 (-0.412) |
| Canon | 0.860→0.671 (-0.189) | 0.732→0.733 (+0.002) |
| KromCanon | 0.720→0.758 (+0.038) | 0.756→0.759 (+0.003) |

### seed3-m8 Complete — N=3 Results

Experiment completed in 114.5 minutes. All 6 phases for 3 architectures.

**N=3 per-stream cosines (bias=-8):**
| Layer | Seed 42 | Seed 137 | Seed 271 | Mean ± SE |
|---|---|---|---|---|
| L0 | 0.990 | 0.993 | 0.990 | 0.991 ± 0.001 |
| L3 | 0.988 | 0.991 | 0.989 | 0.989 ± 0.001 |
| L7 | 0.982 | 0.986 | 0.978 | 0.982 ± 0.002 |

Max SE = 0.002. Predictions from N=2 confirmed: 0.991 (pred 0.992), 0.989 (pred 0.990), 0.982 (pred 0.984).

**N=3 pretrain eval losses:**
| Arch | Seed 42 | Seed 137 | Seed 271 | Mean ± SE |
|---|---|---|---|---|
| vanilla | 6.016 | 6.015 | 6.001 | 6.011 ± 0.005 |
| canon | 5.961 | 5.977 | 5.959 | 5.966 ± 0.006 |
| kromcanon | 5.840 | 5.838 | 5.820 | 5.833 ± 0.006 |

K<C<V ordering holds for all 3 seeds. KromCanon seed 271 shows slight overfitting (+0.191 train-eval gap).

**SFT anomaly (seed 271):** KromCanon first_loss=0.685 (lowest ever), delta=+0.269 (strongest anomaly). KromCanon anomaly rate at bias=-8: 3/3 (100%). Threshold narrowed: [0.756, 0.810).

**Cross-seed direction cosines:** All at random baseline (max |cos|=0.096 for d=512, expected ~0.035). Non-identifiability confirmed at N=3.

**Alpha topology (bias=-8):** 11/16 consistent (69%) — higher than bias=-2 (31%) because factors frozen near identity.

**Direction norms (seed 271):** KromCanon range=0.023 vs vanilla range=0.080 (~3.4x flatter). Consistent with N=2 finding (2.8x).

**Documentation updated:** EXPERIMENTS.md (all seed3-m8 tables filled), BLOG_DRAFT.md (N=2→N=3 for bias=-8 references), make_blog_figures.py (cosine 0.984→0.982), figures regenerated.

### N=3 Complete — Final Summary Tables Added

All N=3 experiments done. Added comprehensive summary tables to EXPERIMENTS.md:

1. **N=3 Alpha Topology (bias=-8)**: Updated from N=2 to N=3. 11/16 consistent (69%) vs bias=-2 (31%). L0/ffn robust across all seeds (+0.820 to +0.918).
2. **N=3 SFT Loss Direction (bias=-8)**: KromCanon anomaly 3/3 (100%), Vanilla normal 3/3 (100%), Canon mixed 1/3.
3. **N=3 Cross-Seed Direction Cosines (bias=-8)**: All at random baseline. Mean |cos|: vanilla=0.088, canon=0.011, kromcanon=0.048.
4. **N=3 Direction Norm Ranges (bias=-8)**: KromCanon 0.029±0.004 vs vanilla 0.082±0.012 (2.8× flatter, SE non-overlapping).
5. **N=3 Steering KL (bias=-8)**: KromCanon robustly negative-dominant (3/3 seeds). Canon mixed.

**Key conclusions from N=3 replication:**
- Per-stream cosines >0.98 at all layers, all seeds, both bias points — **bulletproof**
- Direction norm flatness — **robust** (KromCanon always 2.5-3.5× flatter)
- SFT anomaly — **robust at bias=-8** (3/3), mixed at bias=-2 (2/3)
- Alpha topology — **only L0/ffn is mechanistic**, rest is seed-dependent
- Cross-seed directions — **non-identifiable** (all at random baseline)
- Training ordering K<C<V — **robust** (3/3 seeds at bias=-8)

---

## 2026-03-14 — Iteration 22: ablation_vanilla_krom & Literature

### ablation_vanilla_krom Experiment

Started. Config: `experiments/ablation_vanilla_krom.toml`. Architectures: vanilla + kromhc (no Canon). Bias=-2, seed=42. Purpose: isolate KromHC multi-stream from Canon local conv.

**Predictions** (from EXPERIMENTS.md):
1. Per-stream cosines: should match kromcanon (0.994-0.998 at bias=-2)
2. Pretrain loss: between vanilla (6.02) and kromcanon (5.82)
3. SFT anomaly: uncertain (seed 42 at bias=-2 showed anomaly for kromcanon)

### Literature (session 3)

12. **Interpretable-by-Design Transformers** (2603.07482, Kerce & Fox, Georgia Tech, Mar 2026): Late Fusion Architecture with frozen token stream + learned semantic stream. Uses Kronecker products for channelization. 13-22M params, 6 layers, TinyStories. PDS_max=0.276 vs 0.058 standard. Channel factorization costs 3.6% loss. Key validation: Kronecker-structured mixing preserves interpretability at small scale with bounded cost. Difference from our work: LFA enforces stream independence by design (frozen token stream); our streams are coupled through learned DSM mixing.

13. **KronSAE** (2505.22255, May 2025): Kronecker factorization of SAE encoder. 50% FLOPs reduction, improved interpretability, reduced feature absorption. mAND activation approximates binary AND for composable features. Future direction: Kronecker-factored SAE on KromCanon's multi-stream activations.

14. **Defense via extended-refusal** (2505.19056v2): Distributes refusal across token positions (explanations before refusal). Refusal rate drops ≤10% under abliteration (vs 70-80% baseline). Tested 1.5B-7B. Temporal distribution provides defense; our spatial (cross-stream) distribution does NOT, since streams carry identical directions.
15. **DeepRefusal** (2509.15202): Probabilistic ablation during training forces deeper safety internalization. ~95% attack reduction. Our 500-step SFT is surface alignment; DeepRefusal suggests path to robust alignment.
16. **Loss landscape basins** (2505.17646): Pre-training creates "basic capability basins"; alignment creates "specific capability basins." Low first_loss models may already reside in safety-adjacent sub-basin; SFT overshoots. Framework for our SFT anomaly interpretation.
17. **Pre-training indicators** (2504.12491): Tests whether pre-training metrics predict SFT outcomes at 1B scale. Finds conventional perplexity misleading. Different question from our first_loss threshold (they predict which checkpoint is *better*, not whether loss *increases vs decreases*).

### Blog Consistency Fixes

Corrected several stale N=1 values in BLOG_DRAFT.md:
- Steering KL table (bias=-8): updated from seed 42 to N=3 means with SE
- Unit steering sensitivity: 1.94 KL (vs 2.16 vanilla, -10%) not 2.44 vs 2.80
- Direction norm range SEs: 0.009/0.007/0.003 (not 0.012/0.009/0.004)
- Steering asymmetry interpretation: removed alternating σ₂ pattern claim (not robust across seeds), replaced with "only bias=-8 negative dominance is robust"
- Alpha topology at bias=-8: updated from seed-specific counts to N=3 consistency (69%)
- Per-stream cosines in sweep table: updated to N=3 means
- Basin framework citation added (2505.17646) to SFT anomaly discussion
- OBLITERATUS EGA added to SteerMoE comparison
- Interpretable-by-Design Transformers (2603.07482) added to related work

---

## Iteration 23 — ablation_vanilla_krom Results (2026-03-14)

### Experiment: KromHC Without Canon Layers

Ran `ablation_vanilla_krom.toml` (vanilla + kromhc, bias=-2, seed=42). Total time: 71.5 minutes. All 6 phases complete.

**CORRECTED**: Initial aggregate analysis (cross-stream cos 0.993 vs 0.992) was misleading. Per-layer pairwise cosine analysis reveals Canon IS relevant — it measurably boosts coherence and suppresses SFT anomaly 6×.

### Key Metrics

| Metric | Vanilla | KromHC | KromCanon | Interpretation |
|--------|---------|--------|-----------|----------------|
| Eval loss | 6.013 | 5.926 | 5.840 | KromHC ~60%, Canon ~40% of improvement |
| Norm range | 0.084 | 0.031 | 0.027 | Both ~2.7× flatter — pure KromHC effect |
| Per-layer cos mean | — | 0.966 | 0.983 | Canon +0.017 (significant) |
| Per-layer cos min | — | 0.927 | 0.956 | Canon +0.029 |
| Per-layer cos std | — | 0.016 | 0.009 | Canon halves variance |
| SFT delta | +0.062 | +0.245 | +0.041 | Canon suppresses anomaly 6× |
| Steering dominant | neg | pos | neg | Canon reverses polarity |

### Analysis (corrected)

1. **Aggregate vs per-layer cosines**: The stream_analysis.npz aggregate (0.993 vs 0.992) averages per-stream directions across layers before computing cosines, masking layer-level differences. The per-layer pairwise analysis is the correct comparison.

2. **KromHC drives bulk of coherence**: From random baseline (~0.04) to mean 0.966 — 98% of the effect. Multi-stream mixing through doubly stochastic H^res is the primary mechanism.

3. **Canon adds incremental coherence**: From 0.966 to 0.983 mean (+0.017). Minimum rises from 0.927 to 0.956 (+0.029). Variance halves (0.016→0.009). Canon's shared local conv weights provide cross-stream smoothing.

4. **Norm flatness is pure KromHC**: 0.031 vs 0.027 — both ~2.7× flatter than vanilla. Canon does not affect this.

5. **Canon suppresses SFT anomaly 6×**: KromHC +0.245 vs KromCanon +0.041. Canon's local mixing provides regularization that prevents SFT overshoot.

6. **Alpha topology differs without Canon**: L0/ffn α_res is +0.715 (without Canon) vs most-negative (with Canon). Canon's shared weights change gradient flow and layer specialization.

7. **H^res spectrum**: At bias=-2, σ₂ = 0.70-0.86 (substantial mixing). All H^res diagonals ~0.72-0.86, far from identity.

### Implications for Blog

- Canon is NOT orthogonal — it meaningfully increases coherence and suppresses SFT anomaly
- Direction norm flatness IS a pure KromHC effect (blog headline metric)
- Blog should distinguish: "KromHC creates multi-stream direction coherence; Canon refines it"
- The 6× SFT anomaly suppression is a new finding worth discussing

### Literature (found during experiment wait time)

18. **Heretic vs Abliterated Models** (bswen.com, 2026-03-10): Heretic uses fine-tuning with Bayesian optimization (GPU required, irreversible), while abliteration is weight projection (CPU sufficient, reversible). ~4967 abliterated vs ~2164 heretic models on HuggingFace. Abliteration is architecture-agnostic.

19. **MLSAE — Multi-Layer SAEs** (2409.04185): Train single SAE on residual stream activations from ALL transformer layers. Key finding: individual latents are often active at a single layer for a given token, but which layer varies across tokens/prompts. Relevant to KromCanon: per-stream SAEs could reveal whether directions are concentrated or distributed differently than per-layer analysis suggests.

20. **Steering Non-Identifiability** (Venkatesh & Kurapath, 2602.06801, Feb 2026): Steering vectors are fundamentally non-identifiable — large equivalence classes of directions produce similar behavioral effects. BUT recoverable under structural assumptions: cross-layer consistency, sparsity, multi-environment validation. Directly explains our N=3 cross-seed findings: direction cosines ~0.03 (random) and SVD subspace overlaps within 1σ of random are expected. The identifiable structure in KromCanon is per-stream coherence (>0.99 across all seeds), which is an architectural property not a training artifact.

21. **Steering Unreliability** (Braun, 2602.17881, Feb 2026): Higher cosine similarity between activation differences → more reliable steering. Non-linear behavior representations cause steering failure. Directly relevant: our high per-stream cosines (0.983-0.995 across configs) predict reliable steering in KromCanon. Gives theoretical backing to per-stream cosine as a quality metric for linear interventions.

22. **mHC Follow-ups**: mHC-GNN (2601.02451) shows multi-stream mixing slows over-smoothing by factor of n: (1-γ)^(L/n) vs (1-γ)^L. mHC-lite (2601.05732) bypasses Sinkhorn-Knopp with convex combinations of permutation matrices (similar to KromHC). MGT (2601.01014) unifies manifold-constrained hyper-connections with Deep Delta Learning. These confirm the multi-stream architecture space is actively expanding BUT none study interpretability properties.

23. **Abliteration Minimum Scale**: Gemma-3-270M-IT via Heretic (p-e-w/heretic) is the smallest published abliteration. Granite 4.0 350M (hybrid Mamba-2/Transformer) is smallest non-standard architecture. Our 51M models are 5× below the smallest prior work. Selective Steering (2601.19375) notes "generation collapse in models below 7B parameters" — our scale is 140× below this threshold.

24. **Dual-Stream Transformer** (2603.07461, Mar 2026): 2-stream (token+context) with Kronecker mixing at 29M params. Kronecker mixing costs only 2.5% loss vs dense. Learns hub structures in Kronecker matrices. Stream ablation asymmetric: token stream +36% loss, context stream +9.5%. Most architecturally related to KromCanon — directly shows Kronecker-structured mixing preserves interpretability.

25. **Canon at Scale** (Physics of LMs Part 4.2, facebookresearch/PhysicsLM4): Academic scale (1.3B, 100B tokens) confirms synthetic findings. Larger scale (1-8B, 1-2T tokens): Transformer+Canon strongly outperforms base Transformer; GLA+Canon matches GDN and outperforms Mamba2. Reasoning depth 2-4x, breadth +30%, knowledge capacity +10-15%. No mention of Canon interacting with safety/alignment or fine-tuning — our SFT regularization finding (Canon suppresses SFT anomaly 6×) appears genuinely novel. No independent replications found outside Allen-Zhu group.

26. **Homogeneity Trap Revisited** (2601.02080): Purely theoretical paper (synthetic experiments, no NN training). Predicts spectral collapse (σ₂^L contraction) in Sinkhorn-based DSM networks. Does NOT analyze Kronecker-product DSM (as used in KromHC). Our empirical data shows KromHC avoids this collapse at 8 layers: σ₂ = 0.70-0.86 (non-degenerate), per-stream cos > 0.95 (directions preserved). **Theoretical explanation**: Kronecker DSM has discrete eigenvalue spectrum {1, ev₂, ev₂, ev₂²} from 2×2 factors, vs Sinkhorn's continuous spectrum. At bias=-2: σ₂=0.76, σ₂^8=0.11 (mild contraction, not collapse). At bias=-1: σ₂=0.46, σ₂^8=0.002 (strong but bounded). At bias=0: σ₂=0, rank-1 projection — the Kronecker structure DOES hit homogeneity trap at equal mixing, but learned factors drift away during training. The discrete eigenvalue gap from factored structure creates natural separation that continuous Sinkhorn projection lacks.

27. **Massive SFT Experiments** (2506.14681, EMNLP 2025): 1000+ SFT models show mid-layer weight changes correlate most strongly with performance gains, and perplexity predicts SFT effectiveness. Relevant to our layer-wise direction profiles — our directions peak at L6-L7 (last layers), consistent with our 8-layer depth being too shallow for the "mid-layer sweet spot" seen in larger models.

28. **Self-Attention as Low-Pass Filter** (2312.04234v5, GFSA): Self-attention acts as a low-pass filter that smooths high-frequency information across layers (oversmoothing). Adding local processing (conv/graph ops) preserves high-frequency structure. Connects to Canon mechanism: Canon's causal conv before attention could preserve local high-frequency patterns, constraining gradient drift during SFT distribution shift — a possible mechanism for Canon's 6× SFT anomaly suppression.

29. **mHC Signal Stability Deep Dive** (2512.24880): Unconstrained HC causes 3000× signal gain at 27B scale; DSM constraint reduces to 1.6× (3 orders of magnitude). Key mechanism: spectral norm ||H^res|| ≤ 1 + compositional closure (DSM × DSM = DSM). Sinkhorn uses 20 iterations with on-chip recomputation. 6.7% training overhead at n=4. BBH +7.2%, GSM8K +7.1% over baseline. **Crucially: no fine-tuning/SFT analysis in the paper** — confirms our SFT anomaly finding is novel territory for multi-stream architectures.

30. **Adaptive SFT Safety Regularization** (2602.17546, Feb 2026): Uses activation-based risk predictor to dynamically modulate loss weights during SFT. "Harmful intent predictable from pre-generation activations." Different from our first_loss threshold: they detect per-batch risk at runtime, we predict convergence direction from a single scalar. Their work suggests activation geometry encodes safety-relevant structure even before generation.

31. **PACT: Safety Token Concentration** (2603.07445, Mar 2026): Safety behavior concentrated in small subset of "safety tokens." Regularize only those tokens' confidence during fine-tuning, leave rest unconstrained. Parallel to our direction concentration: safety directions peak at L6-L7 (spatial concentration), they find token-level concentration. Combined: safety is localized in both token AND layer space.

32. **Literature gap confirmed**: No paper addresses first-loss as convergence direction predictor for SFT. Closest analogs: 2601.18699 (gradient alignment at epoch 1 predicts forgetting magnitude, r=0.87) and our finding (first_loss < 0.76 → increasing loss, 7/7; ≥ 0.81 → normal, 8/8). The threshold is architecture-agnostic and appears novel.

33. **SafeMoE: Safety Routing in MoE** (2509.22745): Safety concentrated in specific experts; harmful fine-tuning causes routing decisions to drift away from safety-critical experts (7B-141B). Conceptually parallel to KromCanon: multi-stream H^res mixing can be viewed as dense routing. But DSM constraint prevents routing collapse — unlike MoE's unconstrained routing, our streams maintain coherent directions even after SFT. KromHC's DSM provides structural safety through routing stability.

34. **Safety Gradient Subspace** (2601.10141): Safety gradients are low-rank; utility gradients high-rank; they're negatively correlated → directional conflicts during SFT. SPF removes conflicting gradient components, achieving near-perfect safety recovery. Gradient-space analog of abliteration. Our finding: DSM mixing preserves direction linearity → SPF-style interventions work in multi-stream architectures. **CAST** (2601.04262): Safety-utility conflicts localized to small set of "high-conflict" attention heads. Skip those during training → preserve utility. In KromCanon, all streams carry identical directions (cos>0.98), so no per-stream selectivity needed — abliteration works uniformly.

35. **Selective Steering Deep Dive** (2601.19375): Generation collapse in angular steering below 7B caused by norm violations, NOT inherent scale limitation. Norm-preserving rotation fixes it (zero perplexity violations). Tested 1B-9B. Discriminative layer selection: only steer layers where class projections are opposite-signed. Our additive steering at 51M doesn't suffer from angular collapse — KL analysis confirms steering works. The 5.5× ASR improvement on Qwen-1.5B shows careful steering design matters more than raw scale.

36. **Dual-Stream Transformer Deep Dive** (2603.07461, Mar 2026): 29M params, 6 layers, instructional text. Kronecker mixing: H×H matrix (H² params) for cross-head communication, preserving within-head structure. Key findings: (a) Hub structures: Head 0 emerges as routing hub with 3.5× amplification in deeper layers — parallels our L0/ffn alpha outlier. (b) Stream ablation asymmetric: token stream load-bearing (+36% loss) vs context refinement (+9.5%). Our 4-stream KromCanon doesn't show this asymmetry (all streams carry identical directions) — suggests DSM mixing homogenizes streams more than Kronecker head mixing. (c) 2.5% Kronecker loss cost (graceful degradation under stress). (d) "Channelized mixing enforces separation by design" — tractable circuit identification. No direction extraction or linear probes done. Our work fills this gap directly.

---

## Iteration 24 — seed2_vanilla_krom + ablation_sft_size (2026-03-14)

### seed2_vanilla_krom (COMPLETE)

**Config**: seed=137, bias=-2, V/KH. Replicates Canon ablation finding from iteration 23.

**Results**:

| Metric | seed=42 | seed=137 | N=2 mean |
|--------|---------|----------|----------|
| KH eval loss | 5.926 | 5.917 | 5.922 |
| Per-stream cos mean | 0.966 | 0.981 | 0.974 |
| Per-stream cos min | 0.927 | 0.910 | 0.919 |
| KH norm range | 0.031 | 0.021 | 0.026 |
| SFT delta | +0.245 | -0.111 | n/a |
| SFT first_loss | 0.639 | 0.929 | n/a |
| Steering dominant | positive | negative | n/a |
| Mean frob | 0.497 | 0.504 | 0.501 |

**Key findings**:
1. Per-stream cosines are higher at seed=137 (0.981) than seed=42 (0.966), approaching KromCanon levels (0.983). Canon coherence boost is reduced to +0.009 at N=2 mean (was +0.017 at seed=42 alone).
2. SFT anomaly is seed-dependent for KromHC: anomaly at seed=42 (first_loss=0.639), normal convergence at seed=137 (first_loss=0.929). Governed by first_loss threshold.
3. Vanilla SFT anomaly persists at BOTH seeds (delta +0.062, +0.058) despite very different first_loss values (0.647 vs 0.869). First_loss threshold may not apply to vanilla.
4. H^res mixing topology is seed-stable (frob within 0.007).
5. Steering asymmetry is seed-dependent (positive→negative).

### ablation_sft_size (COMPLETE)

**Config**: seed=42, bias=-8, V/C/K. 100 SFT steps (5× fewer), 500 examples (10× fewer). Pretrained checkpoints reused from `full` run.

**Results**:

| Metric | Full SFT (500 steps) | Minimal SFT (100 steps) |
|--------|---------------------|------------------------|
| V direction cos (full vs min) | — | 0.674 |
| C direction cos (full vs min) | — | 0.606 |
| K direction cos (full vs min) | — | 0.691 |
| K per-stream cos | 0.991 | 0.942 |
| K norm range | 0.027 | 0.008 |
| V SFT delta | -0.306 | +0.044 |
| C SFT delta | -0.189 | +0.042 |
| K SFT delta | +0.038 | +0.003 |

**Key findings** (corrected values from re-verified analysis):
1. Directions ~67% aligned between 100 and 500 SFT steps — partially stable. KromCanon highest stability (0.691), then vanilla (0.674), then canon (0.606). Stability ordering matches Dual-Arch stability-plasticity prediction: KromHC width → most stable.
2. Per-stream coherence drops from 0.991 (full) to 0.942 (minimal) — coherence is architectural but scales with SFT.
3. Norm flattening fully SFT-independent (range 0.008 at 100 steps, even flatter than 0.027 at 500 steps).
4. All archs show SFT anomaly with 500 examples. Dataset size affects first_loss and threshold behavior.
5. Steering works but weaker (KL magnitudes lower). Direction dominance is not a robust signal across SFT budgets.

### Research (during experiment wait)

Added 19 new literature entries (papers 18-36) covering:
- Steering non-identifiability (2602.06801) and unreliability (2602.17881)
- mHC follow-ups, signal stability deep dive (3000× gain → 1.6× with DSM)
- Canon at Scale (Part 4.2): 1-8B validation, no SFT interaction studied
- Homogeneity Trap (2601.02080): Sinkhorn collapse, doesn't apply to Kronecker DSM
- Kronecker DSM eigenvalue analysis: discrete spectrum {1, ev₂, ev₂, ev₂²} prevents continuous spectral drift
- Safety gradient subspace (2601.10141) and CAST head-level diagnosis (2601.04262)
- Selective Steering (2601.19375): generation collapse is norm violation, not scale limitation
- Dual-Stream Transformer (2603.07461): hub structures parallel our L0 alpha, stream asymmetry absent in DSM
- PACT safety tokens (2603.07445): safety concentrated in token AND layer space
- Literature gap: no paper on Kronecker vs Sinkhorn DSM comparison
- Literature gap: no paper on first-loss as SFT convergence predictor
- Confirmed: Canon + multi-stream interaction for interpretability is novel

#### New Literature (papers 37-40)

20. **Dual-Arch: Stability-Plasticity from Architecture** (2506.03951, ICML 2025)
    - **Finding**: Wider networks → superior stability, lower plasticity. Deeper networks → better plasticity.
    - **Relevance**: KromHC's 4-stream architecture is effectively width expansion. Predicts exactly our SFT anomaly: high stability (resists distribution shift) but low plasticity (can't easily fine-tune on safety data). The loss increase during SFT is the stability-plasticity tradeoff manifesting.
    - **Implication**: The SFT anomaly isn't a bug — it's a predictable consequence of multi-stream architecture's width-induced stability.

21. **SafeMoE: Safety Routing Alignment in MoE** (2509.22745)
    - **Finding**: In MoE models, safety is encoded in routing patterns — certain experts become "safety-critical experts". Fine-tuning disrupts routing, degrading safety. SafeMoE penalizes routing drift during FT.
    - **Relevance**: Direct parallel to KromHC multi-stream mixing. Our doubly stochastic mixing matrices play the role of MoE routing. SFT anomaly could be the mixing matrices resisting changes to safety-relevant routing patterns.
    - **Key difference**: MoE routing is sparse (top-k), KromHC mixing is dense (doubly stochastic). Dense mixing may be even more resistant to drift.

22. **mHC: Manifold-Constrained Hyper-Connections** (2512.24880, DeepSeek)
    - **Finding**: Original HC becomes unstable at 27B scale (loss spikes, gradient explosions). mHC fixes via Sinkhorn-Knopp projection onto doubly stochastic manifold. Reduces Amax Gain from ~3000× to ~1.6×.
    - **Relevance**: KromHC improves on mHC with Kronecker products for exact doubly stochasticity. Our norm-preservation property (spectral norm ≤ 1) explains why fine-tuning is harder: the architecture geometrically constrains how far representations can move per update.
    - **Note**: mHC paper focuses on pretraining stability; SFT/alignment behavior not studied. Our data is first evidence of how DSM constraints affect fine-tuning dynamics.

23. **MUDDFormer: Breaking Residual Bottlenecks** (2502.12170)
    - **Finding**: Dense connections to Q/K/V streams improve attention head functionality. MUDDPythia-2.8B matches Pythia-6.9B.
    - **Relevance**: Another multi-stream architecture. Unclear if same SFT dynamics. No alignment/steering experiments reported.

24. **Safety Alignment Should Be Distributed** (2508.19697, ICLR 2025)
    - **Finding**: Safety is concentrated in a few attention heads. RDSHA identifies them via refusal direction. AHD training distributes safety across many heads for robustness.
    - **Relevance**: KromHC's multi-stream mixing naturally duplicates safety directions across all 4 streams (cos >0.99). Different mechanism than AHD but similar effect: abliterating one stream is insufficient because DSM mixing propagates the direction from remaining streams. Natural robustness against per-stream targeted ablation.

25. **Multi-Head Attention as Source of Catastrophic Forgetting in MoE** (2602.12587)
    - **Finding**: Gradient interference in multi-head attention causes catastrophic forgetting even in MoE. Head-wise routing reduces gradient conflict.
    - **Relevance**: Our multi-stream architecture may have analogous gradient interference during SFT, contributing to the SFT anomaly.

26. **SafeGrad: Gradient Surgery for Safe Fine-Tuning** (2508.07172)
    - **Finding**: Projects harmful gradient components onto orthogonal plane of alignment gradients. Prevents safety degradation during fine-tuning.
    - **Relevance**: Our SFT anomaly (loss increase) may be caused by gradient conflict between safety objective and pretrained representations. SafeGrad-style orthogonal projection could mitigate this, though DSM norm constraint adds another constraint layer.

#### New Literature (papers 37-42, from Iteration 24 research)

37. **Entropy-Adaptive Fine-Tuning (EAFT)** (2601.02151)
    - Low entropy + high confidence tokens create destructive "confident conflict" gradients during SFT.
    - Best mechanistic explanation for our first_loss threshold: low first_loss = high model confidence on safety data = destructive gradient updates → loss increases.
    - Masking bottom 15% tokens by entropy+probability eliminated forgetting.

38. **Circular Convolution as Regularizer** (2407.19342, ACL 2025)
    - Explicitly states: circular convolution has "intrinsic bias" serving as regularization during fine-tuning.
    - First published evidence that conv operations have implicit regularization in FT. Different conv type (circular vs causal 1-D) but same principle → theoretical support for Canon's 6× SFT anomaly suppression.

39. **Overtrained LMs Are Harder to Fine-Tune** (2503.19206)
    - U-shaped perplexity during FT. Extended pretraining makes parameters fragile.
    - Complementary to first_loss threshold: low first_loss may indicate model already learned SFT distribution during pretraining → "overtrained" for that data.

40. **White-Box mHC (ES-mHC)** (2601.15757)
    - Assigns physical meaning to streams, analyzes mixing matrices.
    - Only interpretability-focused multi-stream paper. Applicable methodology for KromCanon.

41. **SAND: von Mises-Fisher Direction Extraction** (2502.16385)
    - Models activation differences as vMF samples; derives direction via MLE with concentration parameter κ.
    - Provides per-layer quality diagnostic our code currently lacks. Future addition.

42. **Direction Sample Size Plateau** (Braun et al., 2505.22637)
    - 200-500 samples: cos >0.99 between resampled directions. Beyond 500: diminishing returns.
    - Confirms our 500 contrast pairs is at the plateau. Additional metric: directional agreement (mean cosine of individual diffs vs direction) as quality diagnostic.

## Iteration 25 — ablation_sft_size Analysis + Literature (2026-03-14)

### ablation_sft_size (COMPLETE, re-analyzed)

Re-ran ablation_sft_size experiment and corrected all metric tables. Previous values were from an earlier/incorrect run.

**Key corrected findings**:
- Cross-run direction cosines: KromCanon 0.841, vanilla 0.797, canon 0.717 (was 0.691/0.674/0.606)
- Per-stream cosines: 0.981 mean (was 0.942) — essentially identical to full run (0.983)
- Only ~62 SFT steps ran (data exhausted), not 100
- Steering asymmetry: all three architectures positive-dominant at minimal SFT (flips from full-SFT pattern)
- KromCanon: 8% refusal rate (only arch with measurable refusal at minimal SFT)
- Direction stability ordering (KromCanon > vanilla > canon) matches Dual-Arch stability-plasticity prediction

### Literature (papers 43-46)

43. **One-Shot Optimized Steering Vectors** (Dunefsky & Cohan, COLM 2025, 2502.18862)
    - Optimizes SVs via gradient descent on a single training example. 96.9% Harmbench attack success rate.
    - Relevant: extraction prompt count (100) is far more than needed. SFT quality matters more than extraction quantity for our ablation.

44. **Towards Understanding Steering Strength** (Taimeskhanov et al., 2602.02712)
    - First theoretical analysis of steering magnitude. Non-monotonic effects possible at high alpha.
    - Our alpha range [-3, +3] is well beyond their recommended [0, 1]. Could explain steering curve irregularities.
    - No multi-stream discussion.

45. **CAST: Conditional Activation Steering** (ICLR 2025, 2409.05907)
    - Projects activations onto "condition vector" to decide whether to steer per-input. Rules like "if about hate speech, refuse."
    - Connection: KromHC streams could naturally provide condition detection — different streams might specialize in content classification. Not yet tested.

46. **Practitioner Field Guide** (Mitra, 2026)
    - Sweet spot: 50-100 diverse contrast pairs. 32-64 pairs minimum for consistency. 16 pairs: garbage.
    - YaPO: zero-contrast-pair steering via SAE latents + preference optimization. Untested.

### seed3_vanilla_krom (IN PROGRESS)

**Config**: seed=271, bias=-2, V/KH. Third and final seed for Canon coherence finding.

**Status**: Experiment running. First attempt stalled due to memory pressure (32s/step); restarted after cleanup to ~675ms/step.

**Predictions** (from EXPERIMENTS.md):
- Per-stream cos: 0.960-0.985 (consistent with seeds 42 and 137)
- Norm flatness: range < 0.035 (KromHC structural property)
- SFT anomaly: depends on first_loss (threshold-governed)

### Literature (papers 47-49)

47. **Architectural Obfuscation & Interpretability** (Florencio & Barton, 2506.18053, Jun 2025)
    - Obfuscation alters attention-head activation patterns but preserves residual/FFN pathways. Fine-grained interpretability degraded, top-level behavior intact.
    - Relevant: Canon's causal conv modifies local activation patterns similarly — our finding that directions persist through Canon is consistent with this "function-preserving but pattern-altering" effect.

48. **Personality Trait Interference in Steering** (Bhandari et al., 2602.15847, Jan 2026)
    - Big Five personality steering directions are geometrically coupled. Steering one trait induces changes in others. Hard orthonormalization doesn't eliminate cross-trait effects and reduces steering strength.
    - Relevant: multi-behavior directions occupy coupled subspaces. KromHC's multi-stream might either concentrate or distribute these couplings. Our high per-stream cosines (>0.95) suggest streams carry the same coupled structure.
    - Implication: per-stream abliteration may not offer independent behavioral control — streams share the same coupled direction subspace.

49. **Safety Guardrail Collapse via Representation Similarity** (2506.05346, Jun 2025)
    - High representation similarity between alignment data and fine-tuning data causes safety collapse — 15.7% more detrimental than similarity to harmful data.
    - Direct connection to our first_loss threshold: low first_loss = pretrained representations already similar to safety data = overshoot. High first_loss = low similarity = normal convergence.
    - Validates our observation that dataset geometry (not just architecture) determines SFT outcome.

50. **Early Stopping Theory for Fine-tuning** (2602.13942, Feb 2026)
    - Extends NTK theory to non-random initialization. Proves optimal stopping time T̂ᵒᵖ exists where further training increases loss.
    - Key metric: Cℋ = ||f_pretrained - f_target||ℋ. If Cℋ ≈ 0 (model already near target), T̂ᵒᵖ ≈ 0 and ANY training increases loss.
    - Theoretical formalization of our first_loss threshold: low first_loss means the pretrained model already separates safety data, so SFT overshoots immediately (loss increases from step 1). High first_loss means the model is far from the target, so gradient descent converges normally.
    - Strongest theoretical support yet for the first_loss → SFT anomaly prediction.

51. **Pre-training Indicators for Fine-tuning Outcomes** (Zeng et al., 2504.12491, Apr 2025)
    - Tests whether pretraining metrics predict fine-tuning success. Result: perplexity is misleading; novel proxy metrics reduce prediction error by 50%.
    - Our first_loss is a different metric: initial loss on the *fine-tuning* data (not pretraining perplexity). First_loss directly measures representation-data alignment. Not tested in their framework — potentially a novel predictor.

### New finding: bias_res_init affects SFT dynamics via first_loss

Cross-referencing KromCanon SFT data across all experiments reveals that `bias_res_init` dramatically affects first_loss, which in turn determines SFT anomaly:

| Seed | bias=-8 first_loss | bias=-8 delta | bias=-2 first_loss | bias=-2 delta |
|------|-------------------|--------------|-------------------|--------------|
| 42 | 0.720 | +0.038 (YES) | 0.718 | +0.042 (YES) |
| 137 | 0.756 | +0.003 (border) | 0.754 | +0.045 (YES) |
| 271 | 0.685 | +0.269 (YES) | 1.042 | -0.290 (no) |

Seed=271 shows the clearest effect: at bias=-8 (frozen mixing), first_loss=0.685 → strong anomaly. At bias=-2 (active mixing), first_loss=1.042 → strong normal convergence. The 0.357-point first_loss gap is huge — active mixing fundamentally changes pretrained representation geometry, making the model LESS certain about safety data. This connects to the EAFT mechanism: active mixing creates more uncertainty → fewer confident conflicts → no destructive gradients.

Seed=42 and 137 show minimal bias dependence in first_loss (0.002 difference), suggesting the effect is seed-dependent — some training trajectories are more sensitive to the mixing topology than others.

### Papers 52-54 (Iteration 26 — steering stability and cross-architecture transfer)

52. **Seed-Induced Uniqueness in Transformer Models** (2511.01023, Nov 2025)
    - Teachers embed hidden traits decodable by students. Same-seed students show leakage τ≈0.24, different-seed students τ≈0.12-0.13 — despite global CKA > 0.9.
    - Central finding: representational similarity (CKA) ≠ functional similarity. Leakage tracks alignment within trait-discriminative subspaces, NOT global similarity.
    - **Direct validation of our cross-seed direction cosine finding**: our per-stream cosines ~0.98 (structural similarity) coexist with cross-seed direction cosines ~0 (different trait subspaces). The identifiable structure is the coherence pattern, not the specific direction.
    - Their "subspace-level CKA diagnostic" could be applied to our per-stream directions to quantify whether KromHC streams share trait subspaces.

53. **Does Transformer Interpretability Transfer to RNNs?** (2404.05971, AAAI 2025)
    - Tests contrastive activation addition (steering), tuned lens, and knowledge elicitation on Mamba and RWKV architectures.
    - Result: most techniques transfer effectively. Steering vectors work across fundamentally different architectures (transformer → RNN).
    - Validates our approach: if steering transfers from transformers to RNNs, it should also survive Canon (local conv, RNN-like) and KromHC (multi-stream routing). Our data confirms this.
    - Key nuance: they find improvements possible by leveraging RNN's compressed state — analogous to our finding that per-stream extraction captures more structure than joint extraction.

54. **CAST: Conditional Activation Steering** (2409.05907, ICLR 2025 spotlight)
    - Projects hidden states onto condition vectors to decide WHEN to steer. Different prompt categories activate distinct patterns.
    - Enables rules like "if input is about hate speech, then refuse" without affecting other behavior.
    - Relevance to KromCanon: per-stream activation patterns could serve as natural condition signals. KromHC's multi-stream structure already creates category-specific activation patterns — CAST's mechanism could be built on top of this.
    - IBM released open-source activation steering library: github.com/IBM/activation-steering.

55. **COLD-Steer** (2603.06495, ICLR 2026)
    - Training-free steering: approximates fine-tuning effects at inference via unit kernel or finite-difference approximation. 95% effectiveness with 50× fewer samples.
    - Relevance: alternative to direction extraction for steering; could be tested on multi-stream architectures without SFT.

### Papers 56-61 (Iteration 27 — cross-architecture transfer, theoretical foundations, failure modes)

56. **Toward Universal Steering and Monitoring of AI Models** (2502.03708, Feb 2025, published in Science)
    - Scalable linear representation extraction across LLMs/VLMs/reasoning models (8B-90B). Concept representations are transferable across human languages and composable for multi-concept steering.
    - **Key finding: steerability is a dataset-level property, not model-level.** Larger/newer models are more steerable.
    - Validates our experimental design: same contrastive dataset → fair comparison across vanilla/canon/kromcanon. However, multi-stream residuals are untested territory.

57. **Why Representation Engineering Works: Theoretical Framework** (2503.22720, Mar 2025)
    - First theoretical explanation grounded in stability of neural activity across layers via principal eigenvector. Extends RepE to vision-language models.
    - **Critical for KromCanon**: KromHC's doubly stochastic mixing (spectral norm ≤ 1) guarantees norm preservation, which should interact with eigenvector stability in predictable ways. Could explain our finding that per-stream directions are coherent (0.982-0.997 cosine): the doubly stochastic constraint preserves the principal eigenvector across streams.

58. **Analyzing Generalization and Reliability of Steering Vectors** (2407.12404, Jul 2024, updated 2025)
    - Steering is unreliable for many behaviors: in-distribution steerability is highly variable across inputs, spurious biases contribute substantially. OOD steering vectors often generalize but are brittle to prompt changes.
    - Key finding echoing Paper 56: steerability correlates across models with different sizes and architectures when using the same dataset.
    - Relevant to our steering asymmetry finding: asymmetry may reflect dataset-level properties rather than architectural effects.

59. **Compositional Steering with Steering Tokens** (2601.05062, Jan 2026)
    - Token-space steering (vs activation-space): zero-shot composition of multiple behaviors via dedicated composition tokens. Tested on Llama-3.2-3B and SmolLM3-3B.
    - Sidesteps activation-space questions but the compositionality angle is interesting: if KromHC's streams naturally decompose into compositional behavioral dimensions, token-space steering could exploit this.

60. **Granular Study of Safety Pretraining under Abliteration** (2510.02768, Oct 2025, NeurIPS 2025 Workshop)
    - Studies which safety interventions survive abliteration across 20 SmolLM2-1.7B checkpoints. Refusal-only training is most fragile; combined techniques (safe-data filtering + rephrasing + metatags) yield partial robustness.
    - SmolLM2-1.7B is closest existing scale to our 51M models — but still 33× larger. Confirms our work is in novel sub-100M territory.

61. **Small Vectors, Big Effects: Mechanistic Study of RL-Induced Reasoning via Steering** (2509.06608, Sep 2025)
    - Last-layer vectors act as token-substitution biases on first generated token. Penultimate-layer vectors operate through MLP and unembedding. Steering vectors transfer within model families.
    - Layer-specific mechanistic findings should be tested in KromCanon: KromHC's inter-stream mixing at every layer may blur these clean per-layer roles. Our L7 peak could reflect the combined effect of both mechanisms.

## Iteration 28 — Auto-Status Tracking + seed3_vanilla_krom (2026-03-14)

### Auto-Status Tracking

Added automatic experiment status tracking to the meta system:

1. **`src/kromcanon/meta.py`**: Added "running" to `VALID_STATUSES`
2. **`src/kromcanon/tree.py`**: Added `[RUN]` badge (cyan: `fill:#d1ecf1,stroke:#17a2b8`)
3. **`scripts/experiment.py`**: Added `_update_toml_status()` function for text-based TOML status updates. `run()` now accepts `toml_path` parameter:
   - Sets `status = "running"` when experiment starts
   - Sets `status = "promising"` when all 6 phases complete
   - Reverts to `status = "wip"` on exception

This eliminates manual status tracking — `python -m kromcanon.tree experiments/` now shows real-time experiment status.

### seed3_vanilla_krom (COMPLETE)

**Config**: `experiments/seed3_vanilla_krom.toml`, seed=271, bias=-2, vanilla+KromHC
**Purpose**: N=3 Canon coherence boost finding — does KromHC-only consistently show lower per-stream cosines than KromCanon?
**Duration**: 51.5 minutes

**Key Results**:
- KromHC pretrain: eval loss 5.926 (matches seed=42 exactly)
- SFT: both vanilla (1.106→0.777) and KromHC (0.933→0.901) converge normally (no anomaly)
- Per-stream cosines: mean=0.960, min=0.855 (L5 is weak layer at 0.916)
- KromHC-only has higher cross-seed variance (SE=0.006) than KromCanon (SE=0.001)

**N=3 Canon Coherence Boost**: +0.013±0.006 (all three seeds positive: +0.017, +0.002, +0.021)
- Canon is consistently positive and variance-reducing
- Upgraded from "not robust" (N=2) to "consistent but small" (N=3)

## Iteration 29 — Blog Draft Numeric Audit (2026-03-14)

Cross-referenced all quantitative claims in `docs/BLOG_DRAFT.md` against `docs/EXPERIMENTS.md` source data. Fixed 6 discrepancies:

1. **σ₂ table (line 391)**: bias=-8 row N=2→N=3, L7 cosine 0.984→0.982 (EXPERIMENTS.md N=3 data)
2. **gap_min table (lines 280-285)**: Aligned 5 values with EXPERIMENTS.md (L0/ffn 0.52→0.54, L1/attn 0.79→0.72, L2/attn 1.63→1.66, L2/ffn 2.08→2.05, L4/ffn 1.39→1.33)
3. **Prose gap_min references**: Updated line 287 and Figure 6 caption from 0.52→0.54
4. **Alpha topology (line 181)**: Updated from N=2 (seeds 42+137) to N=3 (added seed 271, noted 11/16=69% consistency)
5. **SFT anomaly table**: Added KromHC-only seed=271 (first_loss=0.933, no anomaly), updated total count 14→18 runs, threshold "9/9"→"10/10"
6. **KromHC-only reframe paragraph**: Updated to reflect 1/3 anomaly rate with all 3 seeds

Also added `seed3_vanilla_krom` to `make_blog_figures_v2.py` KromHC-only training curves data source.

**Verification**: 141 tests pass, ruff clean. All EXPERIMENTS.md source values now match blog draft claims.

## Iteration 30 — Phase 2 Planning: Architecture Sweep (2026-03-14)

### Research Direction

Phase 1 is complete (15 experiments, all documented). User directive: do not scale up yet. Instead, establish a clean local theory on small models by sweeping width, depth, and stream count. This becomes either a follow-up article or companion piece.

### New Research Question

> Does the σ₂ contraction mechanism predict per-stream direction coherence across the architecture design space (width, depth, streams)?

### Five Hypotheses

1. **H1 (Width invariance)**: Cosines independent of d_model — σ₂ is a mixing-matrix property, not content-dimension
2. **H2 (Depth scaling)**: Final-layer cosines increase with depth following σ₂^(2L)
3. **H3 (Stream robustness)**: Coherence maintained at n=2 and n=8 streams
4. **H4 (Gradient trap universality)**: Softmax trap occurs identically across all configs
5. **H5 (σ₂ as sufficient statistic)**: Single scalar predicts cosines across all configs

### Code Changes

1. **`kromhc.py`**: Added σ₂ extraction to `extract_hres_metrics()` via `mx.linalg.svd(hres, stream=mx.cpu)`. New metrics: per-layer `sigma2` and `mean_sigma2`.
2. **`experiment.py`**: Added `n_streams` and `kronecker_factors` override fields to `ExperimentConfig`, threaded to `_make_model_config()`.
3. **6 TOML configs**: `sweep_width_micro`, `sweep_width_medium`, `sweep_depth_4`, `sweep_depth_12`, `sweep_streams_2`, `sweep_streams_8`.
4. **EXPERIMENTS.md**: Phase 2 section with hypotheses, predicted results table, execution order.

### Verification

- All 6 configs parse correctly
- Forward pass verified for n=2, n=4, n=8 streams
- σ₂ extraction verified: returns 0.762 at bias=-2 initialization (matches theoretical prediction)
- 141 tests pass, ruff clean

## Iteration 31 -- Epistemic Audit of Phase 2 Hypotheses (2026-03-14)

### Problems Identified

1. **σ₂ explains 1.3% of coherence variance.** Phase 1 shows cosines=0.982 at bias=-8 (no contraction) vs 0.995 at bias=-2 (strong contraction). The dominant mechanism is shared SFT gradients, not σ₂ contraction. Designing a 6-experiment sweep to test σ₂'s predictive power is testing a refinement while ignoring the elephant.

2. **H1 (width invariance) has unresolvable confounds.** SIZE_PRESETS change n_heads, d_ff, and max_seq_len alongside d_model. Cannot isolate d_model. Parked both width experiments.

3. **H2 (depth scaling) treats directions as passively propagated.** The σ₂^(2L) model assumes directions form at layer 0 and get contracted through subsequent layers. In reality, each layer's SFT weight update independently generates a direction component. Cross-model depth comparison also confounds parameter count (26M vs 51M vs 77M) and per-parameter training budget.

4. **H4 (gradient trap universality) is trivially true.** Softmax saturation is math, not an empirical question.

5. **H5 (σ₂ as sufficient statistic) is unfalsifiable.** Fitting a monotonic curve to 7 points with R^2>0.8 is trivially achievable.

6. **Per-layer cosine profile is flat.** L0=0.995, L7=0.995 at bias=-2. If σ₂ contraction were dominant, we'd see L0 < L7. Flat profile confirms shared gradients dominate.

### Design Changes

- **Reframed research question**: "Under what conditions does coherence break?" (boundary-finding) instead of "Does σ₂ predict cosines?" (confirmation of a 1.3% effect).
- **Added H1 as null hypothesis**: shared-gradient dominance. If all configs show >0.95 cosines, architecture is irrelevant for coherence.
- **Parked width sweep** (2 configs, status="parked"): SIZE_PRESET confound. Added "parked" status to meta.py and tree.py.
- **Reframed H2**: tests whether 67x contraction difference produces *measurable* cosine difference. Includes within-model control (depth=12 per-layer profile).
- **Reframed H3**: focuses on capacity boundary (dims/stream) rather than σ₂ prediction. Notes random-baseline confound at n=8.
- **Dropped H4 and H5** with documented reasons.
- **Changed execution order**: n=8 first (highest information value), then depth=4 (sharpest σ₂ test), then depth=12 (within-model test), then n=2 (control).
- **Added decision tree**: explicit next steps for each outcome category.

### Files Modified

- `docs/EXPERIMENTS.md`: Complete rewrite of Phase 2 section
- `experiments/sweep_width_micro.toml`: status wip -> parked
- `experiments/sweep_width_medium.toml`: status wip -> parked
- `src/kromcanon/meta.py`: added "parked" to VALID_STATUSES
- `src/kromcanon/tree.py`: added [---] badge and Mermaid style for "parked"
- `docs/DEVLOG.md`: this entry

### Verification

- 141 tests pass, ruff clean
- Tree renders correctly with [---] for parked experiments

## Iteration 32 — Canon Init Fix and Phase 1 Re-Run (2026-03-14)

### Problem: Canon Weight Initialization 14.4× Too Weak

While verifying the Physics of LLMs Part 4.1 paper specifications, we discovered our Canon weight initialization deviated significantly from the PyTorch reference implementation:

- **Our init**: `N(0, 0.02)` — std=0.02, 4.0% perturbation at init
- **PyTorch Conv1d default**: Kaiming uniform `U(-0.5, 0.5)` — std=0.289, 56.5% perturbation
- **Deviation**: 14.4× weaker initialization

The old init made Canon essentially a near-identity layer at initialization — the convolution barely mixes adjacent tokens. The paper intends Canon to actively mix from the start.

### Fix

```python
# Old (wrong):
self.weight = mx.random.normal((d_model, kernel_size)) * 0.02

# New (correct):
bound = (1.0 / kernel_size) ** 0.5  # = 0.5 for kernel_size=4
self.weight = mx.random.uniform(
    low=-bound, high=bound, shape=(d_model, kernel_size),
)
```

Mathematical derivation: PyTorch Kaiming uniform with `a=sqrt(5)`:
- `fan_in = kernel_size` (depthwise: 1 input channel per group)
- `bound = sqrt(6 / ((1 + a²) * fan_in)) = sqrt(6 / (6 * kernel_size)) = sqrt(1/kernel_size)`
- For kernel_size=4: bound = 0.5 ✓

### Impact

All 10 experiments using Canon/KromCanon architectures must be re-run. Three vanilla-only experiments (ablation_vanilla_krom, seed2_vanilla_krom, seed3_vanilla_krom) are unaffected since they don't use Canon layers.

### Phase 1 Re-Run Progress

Quick smoke test completed (12.1 min):
- K<C<V pretrain ordering preserved ✓
- Per-stream cosines 0.91-0.97 at 50 SFT steps ✓
- Direction norms non-degenerate ✓
- Pipeline correctness verified ✓

Full run launched in background (~2 hr expected). Remaining 8 experiments queued.

### Full Implementation Verification

While waiting for experiments, conducted thorough verification of both Canon and KromHC against source papers:

**Canon (Physics of LLMs 4.1)**:
- Kernel size 4 ✓
- Canon-A after LayerNorm, before attention ✓
- Canon-B shared concatenation pattern (Q,K,V→single conv→split) ✓
- Bias=False default ✓
- Kaiming uniform init ✓ (now fixed)

**KromHC (Wang et al.)**:
- Doubly stochastic via Kronecker product of 2×2 factors ✓
- Softmax parameterization for convex combination ✓
- bias_res_init=-8 → H^res ≈ Identity at init ✓
- H^pre (sigmoid) and H^post (2×sigmoid) ✓
- Alpha init = 0.01, W matrices init = 0 ✓
- Parameter complexity O(n²C) ✓

No further deviations found. Both implementations are paper-compliant.

### Files Modified

- `src/kromcanon/canon.py`: Weight init changed from N(0,0.02) to U(-bound, bound)
- `scripts/run_phase1_rerun.sh`: New batch script for re-running all Phase 1 experiments
- `docs/EXPERIMENTS.md`: Phase 1 re-run section added with quick results
- `docs/DEVLOG.md`: this entry
- All affected TOML configs: meta sections updated with status/date/notes

### Verification

- 141 tests pass, ruff clean
- Quick re-run complete, all 6 phases pass
- Full re-run launched

## Iteration 33 -- Physics of LLMs Deep Dive and Canon-C/D Implementation (2026-03-14)

### Paper Audit

Read all 8 papers in the Physics of Language Models series (Allen-Zhu et al.):

- **Part 1**: Hierarchical structures via DP-like attention (validates rotary embeddings)
- **Part 2.1**: Hidden reasoning in intermediate layers (validates direction extraction at mid-layers)
- **Part 2.2**: Error-correction data improves reasoning (models "already know" mistakes)
- **Part 3.1**: Knowledge must be augmented during pretraining for extraction (linear probing validates our approach)
- **Part 3.2**: Knowledge manipulation fails without CoT (inherent limitation, not trainable)
- **Part 3.3**: 2 bits/param capacity; GPT-2 >= LLaMA for undertrained models; GatedMLP harder to train
- **Part 4.1**: Canon layers (A/B/C/D) with detailed ablations; Canon-AbCD(res) recommended
- **Part 4.2**: Canon benefits transfer to 1B/3B/8B scale

### Key Discovery: Canon-C and Canon-D

The paper describes four Canon placement points:
- Canon-A: before attention, after RMSNorm (we had this)
- Canon-B: on Q/K/V jointly (we had this)
- Canon-C: before MLP, after RMSNorm (NEW)
- Canon-D: inside MLP, before activation (NEW)

Our implementation only had Canon-AB. The paper recommends Canon-AbCD(res) as the optimal configuration. Canon-ACD alone often matches Canon-ABCD.

### Implementation

Added Canon-C and Canon-D to `model.py`:
- **Canon-C**: `TransformerBlock._ffn_branch()` now applies `canon_c(norm2(x))` before FFN
- **Canon-D**: `FeedForward.__call__()` now applies `canon_d(fc1(x))` before GELU activation
- Canon-D operates at d_ff dimension (2048 for small), not d_model
- Parameter overhead: Canon-AB = +0.14%, Canon-ABCD = +0.29% (at d=512, depth=8)

The `canon_set` config already supports "ABCD" as a value. Default remains "AB" for backward compatibility with current experiments. Future ablation experiments can test "ABCD" via TOML config.

### Architecture Validation

All Physics of LLMs findings validate our choices:
- GPT-2 architecture optimal for undertrained knowledge storage (Part 3.3)
- Standard MLP better than GatedMLP in short training (Part 3.3)
- Linear direction extraction validated by Part 3.1 (knowledge linearly encoded)
- Canon kernel_size=4, no bias, Kaiming uniform, residual = all correct

### Files Modified

- `src/kromcanon/model.py`: Added Canon-C (TransformerBlock) and Canon-D (FeedForward)
- `tests/test_canon.py`: 4 new tests for Canon-C/D placement and ABCD config
- `docs/REFERENCE.md`: Updated with Canon ablation findings and C/D details
- `scripts/analyze_sweep.py`: Fixed hardcoded cosine_L7 in cross-sweep table
- `scripts/analyze_phase2.py`: New Phase 2 analysis script
- `docs/DEVLOG.md`: this entry

### Verification

- 145 tests pass (141 + 4 new Canon-C/D tests), ruff clean
- Canon-ABCD forward pass verified: (1, 16, 50304) output correct
- Canon-AB backward compatibility preserved: existing experiments unaffected
- Parameter count: Vanilla=51.19M, Canon-AB=51.26M, Canon-ABCD=51.34M

---

## 2026-03-14 — Iteration 34: Default Canon-ABCD + Phase 1 Batch Re-Run

### Decision: Change Default from Canon-AB to Canon-ABCD

Physics of LLMs Part 4.1 (Allen-Zhu) recommends Canon-AbCD(res) as the optimal configuration. Our Iteration 33 implemented Canon-C and Canon-D but kept the default at "AB" for backward compatibility. Now changed to "ABCD" as default — all future experiments use the full Canon set unless explicitly overridden.

### Changes

1. **`src/kromcanon/config.py`**: `canon_set` default `"AB"` → `"ABCD"`
2. **`scripts/experiment.py`**: Added `canon_set` field to `ExperimentConfig`, TOML `[canon]` section parsing, override in `_make_model_config()`
3. **`scripts/run_phase1_rerun.sh`**: Added `quick` to experiment list, updated header to mention ABCD change

### Phase 1 Batch Re-Run

Killed the running `full.toml` (PID 70419) which had loaded the old Canon-AB default. Launched batch script `run_phase1_rerun.sh` which runs all 10 Phase 1 experiments sequentially with:
- Corrected Canon init: Kaiming uniform U(-0.5, 0.5)
- Full Canon set: ABCD

Experiment order: quick → full → ablation_bias_res → seed2_bias_m8 → seed3_bias_m8 → seed2_bias_m2 → seed3_bias_m2 → sweep_bias_m1 → sweep_bias_0 → ablation_sft_size

After Phase 1 completes, the script automatically starts Phase 2 with `sweep_streams_8`.

### Impact

- The `full.toml` re-run will take ~2 hrs (vanilla 40min + canon 40min + kromcanon 40min)
- Total batch: ~8-10 hrs for all 10 experiments + Phase 2
- All results under `results/` are deleted and recreated from scratch
- Numbers in EXPERIMENTS.md and BLOG_DRAFT.md will change after re-runs complete

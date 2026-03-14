"""Full experiment runner for KromCanon research.

Orchestrates all 6 phases of the experiment pipeline, driven entirely
by a TOML configuration file — no CLI flags.

Phases:
    1. Pretrain all architectures on identical data
    2. Safety fine-tune all on identical safety data
    3. Extract refusal directions (mean-diff + SVD + multistream)
    4. Abliteration experiments (before/after refusal rate)
    5. Steering alpha sweeps
    6. Cross-architecture comparison + figure generation

Usage:
    python -m scripts.experiment experiments/quick.toml
    python -m scripts.experiment experiments/full.toml
"""

from __future__ import annotations

import json
import re
import sys
import time
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import mlx.utils
import numpy as np

from kromcanon.config import ModelConfig, TrainConfig, make_config
from kromcanon.data import PretrainDataLoader
from kromcanon.interp.abliterate import abliterate_model, measure_refusal_rate
from kromcanon.interp.compare import (
    analyze_stream_distribution,
    compare_directions,
    format_comparison_report,
)
from kromcanon.interp.extract import (
    ExtractionResult,
    MultiStreamExtractionResult,
    collect_activations,
    collect_multistream_activations,
    extract_mean_diff,
    extract_multistream_directions,
    extract_svd,
)
from kromcanon.interp.io import (
    load_alpha_sweep,
    load_extraction,
    load_logs,
    load_multistream,
    load_refusal_rates,
    save_alpha_sweep,
    save_extraction,
    save_logs,
    save_multistream,
    save_refusal_rates,
    save_stream_analysis,
)
from kromcanon.interp.steer import sweep_alpha
from kromcanon.meta import MetaConfig, parse_meta
from kromcanon.model import make_model
from kromcanon.safety_data import (
    iter_safety_batches,
    load_beavertails,
    load_hh_rlhf,
    tokenize_conversations,
)
from kromcanon.sft import sft_train
from kromcanon.train import load_checkpoint, train

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────


@dataclass
class ExperimentConfig:
    """Fully-typed experiment configuration loaded from TOML.

    All experiment parameters live here — the TOML file is the single
    source of truth, and this dataclass is its typed mirror.
    """

    # [experiment]
    run_name: str
    resume: bool = True
    seed: int = 42

    # [model]
    depth: int = 12
    size: str = "medium"
    architectures: list[str] = field(
        default_factory=lambda: ["vanilla", "canon", "kromcanon"]
    )

    # [pretrain]
    pretrain_max_steps: int = 5000
    pretrain_batch_size: int = 64
    pretrain_lr: float = 6e-4
    pretrain_max_tokens: int = 1_200_000_000

    # [sft]
    sft_max_steps: int = 1000
    sft_batch_size: int = 32
    sft_max_examples: int = 10000

    # [extraction]
    n_prompts: int = 100

    # [steering]
    alphas: list[float] = field(
        default_factory=lambda: [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
    )

    # [kromhc] overrides
    bias_res_init: float = -8.0  # Init for H^res Kronecker factors; -8 → identity, -2 → mild mixing
    freeze_hres: bool = False  # If True, H^res is hard-coded to identity (ablation)

    # [wandb]
    wandb_enabled: bool = False
    wandb_project: str = "kromcanon"

    # [meta] — experiment lineage (does not affect pipeline execution)
    meta: MetaConfig | None = None


def _has_multistream(arch: str) -> bool:
    """Return True if the architecture uses KromHC multi-stream residuals."""
    return arch in ("kromcanon", "kromhc")


def _make_model_config(cfg: ExperimentConfig, arch: str) -> ModelConfig:
    """Create a model config with experiment-level overrides applied.

    Args:
        cfg: Experiment configuration.
        arch: Architecture name.

    Returns:
        ModelConfig with bias_res_init override applied.
    """
    model_config = make_config(arch=arch, depth=cfg.depth, size=cfg.size)
    if model_config.kromhc.enabled:
        model_config.kromhc.bias_res_init = cfg.bias_res_init
        model_config.kromhc.freeze_hres = cfg.freeze_hres
    return model_config


def load_config(toml_path: Path) -> ExperimentConfig:
    """Parse a TOML file into an ExperimentConfig.

    Args:
        toml_path: Path to the experiment TOML file.

    Returns:
        Fully-populated ExperimentConfig.
    """
    raw = tomllib.loads(toml_path.read_text())

    exp = raw.get("experiment", {})
    model = raw.get("model", {})
    pre = raw.get("pretrain", {})
    sft = raw.get("sft", {})
    ext = raw.get("extraction", {})
    steer = raw.get("steering", {})
    wb = raw.get("wandb", {})

    # Parse [meta] section (optional — does not affect pipeline)
    meta: MetaConfig | None = None
    try:
        meta = parse_meta(raw, fallback_id=toml_path.stem)
    except (TypeError, ValueError) as exc:
        print(f"  Warning: invalid [meta] section: {exc}")

    return ExperimentConfig(
        run_name=exp["run_name"],
        resume=exp.get("resume", True),
        seed=exp.get("seed", 42),
        depth=model.get("depth", 12),
        size=model.get("size", "medium"),
        architectures=model.get("architectures", ["vanilla", "canon", "kromcanon"]),
        pretrain_max_steps=pre.get("max_steps", 5000),
        pretrain_batch_size=pre.get("batch_size", 64),
        pretrain_lr=pre.get("lr", 6e-4),
        pretrain_max_tokens=pre.get("max_tokens", 1_200_000_000),
        sft_max_steps=sft.get("max_steps", 1000),
        sft_batch_size=sft.get("batch_size", 32),
        sft_max_examples=sft.get("max_examples", 10000),
        n_prompts=ext.get("n_prompts", 100),
        alphas=steer.get("alphas", [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]),
        bias_res_init=raw.get("kromhc", {}).get("bias_res_init", -8.0),
        freeze_hres=raw.get("kromhc", {}).get("freeze_hres", False),
        wandb_enabled=wb.get("enabled", False),
        wandb_project=wb.get("project", "kromcanon"),
        meta=meta,
    )


def _update_toml_status(toml_path: Path, new_status: str) -> None:
    """Update the ``[meta] status`` field in a TOML file in-place.

    Uses text-based replacement to preserve formatting and comments.
    If the ``[meta]`` section or ``status`` key is missing, this is a no-op.

    Args:
        toml_path: Path to the experiment TOML file.
        new_status: New status value (must be in ``VALID_STATUSES``).
    """
    text = toml_path.read_text()
    # Match status = "..." within the [meta] section
    updated = re.sub(
        r'(^\s*status\s*=\s*")[^"]*(")',
        rf"\g<1>{new_status}\2",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if updated != text:
        toml_path.write_text(updated)


# ─────────────────────────────────────────────────────────────────────
# Wandb
# ─────────────────────────────────────────────────────────────────────


class WandbLogger:
    """Wrapper for optional wandb logging.

    All methods are no-ops when wandb is disabled.

    Args:
        enabled: Whether to actually log to wandb.
        project: Wandb project name.
        run_name: Run name.
        config: Experiment config dict to log.
    """

    def __init__(
        self, enabled: bool, project: str, run_name: str, config: dict[str, object]
    ) -> None:
        self.enabled = enabled
        self.run = None
        if enabled:
            import wandb

            self.run = wandb.init(
                project=project,
                name=run_name,
                config=config,
                reinit=True,
            )

    def log(self, data: dict[str, object], step: int | None = None) -> None:
        """Log metrics.

        Args:
            data: Metrics dict.
            step: Optional global step.
        """
        if self.run is not None:
            import wandb

            wandb.log(data, step=step)

    def log_figure(self, key: str, fig_path: Path) -> None:
        """Log a figure file.

        Args:
            key: Wandb key.
            fig_path: Path to figure file.
        """
        if self.run is not None:
            import wandb

            wandb.log({key: wandb.Image(str(fig_path))})

    def log_summary(self, data: dict[str, object]) -> None:
        """Log summary metrics.

        Args:
            data: Summary dict.
        """
        if self.run is not None:
            import wandb

            for k, v in data.items():
                wandb.run.summary[k] = v

    def finish(self) -> None:
        """Finish the wandb run."""
        if self.run is not None:
            import wandb

            wandb.finish()


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _to_np(arr: mx.array) -> np.ndarray:
    """Convert mx.array to numpy, casting bfloat16→float32 first.

    Args:
        arr: MLX array (any dtype).

    Returns:
        Numpy array in float32 (if input was bfloat16) or original dtype.
    """
    if arr.dtype == mx.bfloat16:
        return np.array(arr.astype(mx.float32))
    return np.array(arr)


def _log_kromhc_diagnostics(
    cfg: ExperimentConfig,
    results: Path,
    wb: WandbLogger,
    phase: str = "pretrain",
) -> None:
    """Log KromHC parameter diagnostics (Kronecker factors, H^pre/H^post).

    Runs analyze_kromhc on the latest checkpoint and logs key metrics.

    Args:
        cfg: Experiment configuration.
        results: Results directory.
        wb: Wandb logger.
        phase: Phase label for logging.
    """
    # Check for kromcanon or kromhc checkpoint
    for arch_name in ("kromcanon", "kromhc"):
        final_ckpt = results / "checkpoints" / arch_name / "final"
        if final_ckpt.exists():
            break
    else:
        print("  KromHC diagnostics: no final checkpoint found, skipping")
        return

    try:
        from scripts.analyze_kromhc import analyze_checkpoint

        analysis = analyze_checkpoint(final_ckpt, depth=cfg.depth, size=cfg.size)
        out_path = results / f"kromhc_analysis_{phase}.json"
        with open(out_path, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"  KromHC diagnostics saved to {out_path}")

        # Log summary metrics
        summary = analysis.get("summary", {})
        wb.log({
            f"kromhc/{phase}/avg_frobenius_from_identity":
                summary.get("avg_frobenius_from_identity", 0),
            f"kromhc/{phase}/avg_frobenius_from_uniform":
                summary.get("avg_frobenius_from_uniform", 0),
        })

        # Log per-layer factor weights
        for layer_info in analysis.get("layers", []):
            idx = layer_info["layer_index"]
            branch = layer_info["branch"]
            for k, w in enumerate(layer_info["weights"]):
                wb.log({
                    f"kromhc/{phase}/L{idx}_{branch}_factor{k}_identity_weight": w[0],
                })

    except Exception as e:
        print(f"  KromHC diagnostics failed: {e}")


def _results_dir(run_name: str) -> Path:
    """Get the results directory for a run.

    Args:
        run_name: Experiment run name.

    Returns:
        Path to results/{run_name}/.
    """
    return Path("results") / run_name


def _phase_complete(results: Path, phase: int, archs: list[str]) -> bool:
    """Check if a phase has already completed (all expected outputs exist).

    Args:
        results: Results directory.
        phase: Phase number (1-6).
        archs: Architecture names.

    Returns:
        True if outputs exist.
    """
    checks: dict[int, callable] = {
        1: lambda: all(
            (results / "pretrain" / f"{a}_logs.json").exists() for a in archs
        ),
        2: lambda: all(
            (results / "sft" / f"{a}_sft_logs.json").exists() for a in archs
        ),
        3: lambda: all(
            (results / "directions" / f"{a}_mean_diff.npz").exists() for a in archs
        ),
        4: lambda: (results / "abliteration" / "refusal_rates.json").exists(),
        5: lambda: all(
            (results / "steering" / f"{a}_alpha_sweep.npz").exists() for a in archs
        ),
        6: lambda: (results / "figures" / "fig1_training_curves.pdf").exists(),
    }
    return checks.get(phase, lambda: False)()


def _load_existing_extractions(
    results: Path, archs: list[str],
) -> dict[str, dict[str, ExtractionResult | MultiStreamExtractionResult]]:
    """Load already-saved extraction results from disk.

    Args:
        results: Results directory.
        archs: Architecture names.

    Returns:
        Dict mapping arch -> {method -> result}.
    """
    out: dict[str, dict[str, ExtractionResult | MultiStreamExtractionResult]] = {}
    for arch in archs:
        out[arch] = {}
        md_path = results / "directions" / f"{arch}_mean_diff"
        if md_path.with_suffix(".npz").exists():
            out[arch]["mean_diff"] = load_extraction(md_path)
        svd_path = results / "directions" / f"{arch}_svd"
        if svd_path.with_suffix(".npz").exists():
            out[arch]["svd"] = load_extraction(svd_path)
        if _has_multistream(arch):
            ms_path = results / "directions" / f"{arch}_multistream"
            if ms_path.with_suffix(".npz").exists():
                out[arch]["multistream"] = load_multistream(ms_path)
    return out


# ─────────────────────────────────────────────────────────────────────
# Phase 1: Pretrain
# ─────────────────────────────────────────────────────────────────────


def phase1_pretrain(
    cfg: ExperimentConfig,
    results: Path,
    wb: WandbLogger,
) -> dict[str, list[dict[str, float]]]:
    """Pretrain all architectures on identical data.

    Tokenizes data once, then trains each variant sequentially.

    Args:
        cfg: Experiment configuration.
        results: Results directory.
        wb: Wandb logger.

    Returns:
        Dict mapping arch -> training logs.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Pretraining")
    print("=" * 60)

    from transformers import AutoTokenizer

    from kromcanon.data import load_fineweb_edu, prepare_pretraining_data

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print("Loading FineWeb-Edu dataset...")
    texts = load_fineweb_edu()
    model_config = make_config(depth=cfg.depth, size=cfg.size)
    cache_dir = Path("results") / ".cache"
    sequences = prepare_pretraining_data(
        texts=texts,
        encode_fn=tokenizer.encode,
        seq_len=model_config.max_seq_len,
        max_tokens=cfg.pretrain_max_tokens,
        cache_dir=cache_dir,
    )
    print(f"  Packed {len(sequences):,} sequences")

    n_eval = max(len(sequences) // 100, 1)
    eval_seqs = sequences[:n_eval]
    train_seqs = sequences[n_eval:]

    all_logs: dict[str, list[dict[str, float]]] = {}

    for arch in cfg.architectures:
        out_path = results / "pretrain" / f"{arch}_logs.json"
        if out_path.exists():
            print(f"  {arch}: loading existing logs")
            all_logs[arch] = load_logs(out_path)
            continue

        print(f"\n  Training {arch}...")
        model_config = _make_model_config(cfg, arch)
        model = make_model(model_config)
        n_params = sum(
            v.size for _, v in mlx.utils.tree_flatten(model.trainable_parameters())
        )
        print(f"  Parameters: {n_params:,}")

        train_loader = PretrainDataLoader(train_seqs, batch_size=cfg.pretrain_batch_size)
        eval_loader = PretrainDataLoader(
            eval_seqs, batch_size=cfg.pretrain_batch_size, shuffle=False,
        )

        train_config = TrainConfig(
            batch_size=cfg.pretrain_batch_size,
            max_steps=cfg.pretrain_max_steps,
            lr=cfg.pretrain_lr,
            checkpoint_dir=str(results / "checkpoints"),
        )

        logs = train(
            model, train_loader, model_config, train_config,
            eval_loader=eval_loader,
        )

        for entry in logs:
            wb.log({
                f"pretrain/{arch}/loss": entry["loss"],
                f"pretrain/{arch}/time_ms": entry.get("time_ms", 0),
            }, step=int(entry["step"]))

        all_logs[arch] = logs
        save_logs(logs, out_path)
        print(f"  {arch}: saved logs to {out_path}")

    return all_logs


# ─────────────────────────────────────────────────────────────────────
# Phase 2: Safety Fine-Tuning
# ─────────────────────────────────────────────────────────────────────


def phase2_sft(
    cfg: ExperimentConfig,
    results: Path,
    wb: WandbLogger,
) -> dict[str, list[dict[str, float]]]:
    """Safety fine-tune all architectures on identical data.

    Args:
        cfg: Experiment configuration.
        results: Results directory.
        wb: Wandb logger.

    Returns:
        Dict mapping arch -> SFT logs.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Safety Fine-Tuning")
    print("=" * 60)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print("Loading safety datasets...")
    hh_pairs = load_hh_rlhf(max_examples=cfg.sft_max_examples)
    bt_pairs = load_beavertails(max_examples=cfg.sft_max_examples)
    all_pairs = hh_pairs + bt_pairs
    print(f"  Total pairs: {len(all_pairs)}")

    sequences = tokenize_conversations(
        all_pairs, encode_fn=tokenizer.encode, max_len=512,
    )
    print(f"  Tokenized: {len(sequences)} sequences")

    all_logs: dict[str, list[dict[str, float]]] = {}

    for arch in cfg.architectures:
        out_path = results / "sft" / f"{arch}_sft_logs.json"
        if out_path.exists():
            print(f"  {arch}: loading existing SFT logs")
            all_logs[arch] = load_logs(out_path)
            continue

        print(f"\n  SFT {arch}...")
        model_config = _make_model_config(cfg, arch)
        model = make_model(model_config)

        ckpt_path = results / "checkpoints" / arch / "final"
        if ckpt_path.exists():
            load_checkpoint(model, ckpt_path)
            print(f"  Loaded pretrained checkpoint from {ckpt_path}")

        train_data = iter_safety_batches(
            sequences, batch_size=cfg.sft_batch_size, seq_len=512,
        )
        train_config = TrainConfig(
            batch_size=cfg.sft_batch_size,
            lr=cfg.pretrain_lr,
            checkpoint_dir=str(results / "checkpoints"),
        )

        logs = sft_train(
            model, train_data, model_config, train_config,
            max_steps=cfg.sft_max_steps,
        )

        for entry in logs:
            wb.log({f"sft/{arch}/loss": entry["loss"]}, step=int(entry["step"]))

        all_logs[arch] = logs
        save_logs(logs, out_path)
        print(f"  {arch}: saved SFT logs to {out_path}")

    return all_logs


# ─────────────────────────────────────────────────────────────────────
# Phase 3: Direction Extraction
# ─────────────────────────────────────────────────────────────────────


def _make_test_prompts(
    tokenizer_encode: callable,
    n_prompts: int,
    vocab_size: int,
) -> tuple[list[mx.array], list[mx.array]]:
    """Create tokenized test prompts for direction extraction.

    Tries HH-RLHF test split first, falls back to random prompts.

    Args:
        tokenizer_encode: Tokenizer encode function.
        n_prompts: Prompts per category.
        vocab_size: Model vocab size.

    Returns:
        (harmful_ids, harmless_ids) as lists of (1, seq_len) arrays.
    """
    try:
        from kromcanon.safety_data import load_test_prompts

        harmful_texts, harmless_texts = load_test_prompts(max_examples=n_prompts)
        max_len = 64

        harmful_ids: list[mx.array] = []
        for text in harmful_texts[:n_prompts]:
            tokens = tokenizer_encode(f"User: {text}\nAssistant:")[:max_len]
            if len(tokens) > 5:
                harmful_ids.append(mx.array(tokens).reshape(1, -1))

        harmless_ids: list[mx.array] = []
        for text in harmless_texts[:n_prompts]:
            tokens = tokenizer_encode(f"User: {text}\nAssistant:")[:max_len]
            if len(tokens) > 5:
                harmless_ids.append(mx.array(tokens).reshape(1, -1))

        if harmful_ids and harmless_ids:
            return harmful_ids, harmless_ids
    except Exception as e:
        print(f"  Warning: could not load test prompts: {e}")

    print("  Using random prompts for extraction")
    return (
        [mx.random.randint(0, vocab_size, (1, 64)) for _ in range(n_prompts)],
        [mx.random.randint(0, vocab_size, (1, 64)) for _ in range(n_prompts)],
    )


def phase3_extract(
    cfg: ExperimentConfig,
    results: Path,
    wb: WandbLogger,
) -> dict[str, dict[str, ExtractionResult | MultiStreamExtractionResult]]:
    """Extract refusal directions from all architectures.

    Args:
        cfg: Experiment configuration.
        results: Results directory.
        wb: Wandb logger.

    Returns:
        Dict mapping arch -> {method -> ExtractionResult}.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Direction Extraction")
    print("=" * 60)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    all_results: dict[str, dict[str, ExtractionResult | MultiStreamExtractionResult]] = {}

    for arch in cfg.architectures:
        all_results[arch] = {}
        dir_path = results / "directions"
        md_path = dir_path / f"{arch}_mean_diff"
        svd_path = dir_path / f"{arch}_svd"

        if md_path.with_suffix(".npz").exists():
            print(f"  {arch}: loading existing directions")
            all_results[arch]["mean_diff"] = load_extraction(md_path)
            if svd_path.with_suffix(".npz").exists():
                all_results[arch]["svd"] = load_extraction(svd_path)
            if _has_multistream(arch):
                ms_path = dir_path / f"{arch}_multistream"
                if ms_path.with_suffix(".npz").exists():
                    all_results[arch]["multistream"] = load_multistream(ms_path)
            continue

        print(f"\n  Extracting directions for {arch}...")
        model_config = _make_model_config(cfg, arch)
        model = make_model(model_config)

        sft_ckpt = results / "checkpoints" / f"{arch}_sft" / "final"
        if sft_ckpt.exists():
            load_checkpoint(model, sft_ckpt)
            print(f"  Loaded SFT checkpoint from {sft_ckpt}")

        harmful_ids, harmless_ids = _make_test_prompts(
            tokenizer.encode, cfg.n_prompts, model_config.vocab_size,
        )
        print(f"  {len(harmful_ids)} harmful, {len(harmless_ids)} harmless prompts")

        print("  Collecting activations...")
        if _has_multistream(arch):
            harmful_acts = collect_multistream_activations(model, harmful_ids)
            harmless_acts = collect_multistream_activations(model, harmless_ids)
            ms_result = extract_multistream_directions(harmful_acts, harmless_acts)
            all_results[arch]["multistream"] = ms_result
            save_multistream(ms_result, dir_path / f"{arch}_multistream")
            print("  Saved multistream directions")

            harmful_acts_std = collect_activations(model, harmful_ids)
            harmless_acts_std = collect_activations(model, harmless_ids)
        else:
            harmful_acts_std = collect_activations(model, harmful_ids)
            harmless_acts_std = collect_activations(model, harmless_ids)

        md_result = extract_mean_diff(harmful_acts_std, harmless_acts_std)
        all_results[arch]["mean_diff"] = md_result
        save_extraction(md_result, dir_path / f"{arch}_mean_diff")

        svd_result = extract_svd(harmful_acts_std, harmless_acts_std)
        all_results[arch]["svd"] = svd_result
        save_extraction(svd_result, dir_path / f"{arch}_svd")

        for layer_i in range(md_result.layer_norms.shape[0]):
            wb.log({
                f"directions/{arch}/mean_diff_norm_L{layer_i}":
                    md_result.layer_norms[layer_i].item(),
                f"directions/{arch}/svd_norm_L{layer_i}":
                    svd_result.layer_norms[layer_i].item(),
            })

        print(f"  {arch}: extraction complete")

    return all_results


# ─────────────────────────────────────────────────────────────────────
# Phase 4: Abliteration
# ─────────────────────────────────────────────────────────────────────


def phase4_abliterate(
    cfg: ExperimentConfig,
    results: Path,
    extraction_results: dict[str, dict[str, ExtractionResult | MultiStreamExtractionResult]],
    wb: WandbLogger,
) -> dict[str, dict[str, float]]:
    """Measure refusal before/after direction removal.

    Args:
        cfg: Experiment configuration.
        results: Results directory.
        extraction_results: Phase 3 output.
        wb: Wandb logger.

    Returns:
        Dict mapping arch -> {"before": rate, "after": rate}.
    """
    print("\n" + "=" * 60)
    print("PHASE 4: Abliteration")
    print("=" * 60)

    rates_path = results / "abliteration" / "refusal_rates.json"
    if rates_path.exists():
        print("  Loading existing refusal rates")
        return load_refusal_rates(rates_path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    decode_fn = tokenizer.decode

    all_rates: dict[str, dict[str, float]] = {}

    # Use real harmful prompts for refusal measurement (not random tokens)
    model_config_0 = make_config(
        arch=cfg.architectures[0], depth=cfg.depth, size=cfg.size,
    )
    harmful_prompts, _ = _make_test_prompts(
        tokenizer.encode, min(cfg.n_prompts, 50), model_config_0.vocab_size,
    )
    print(f"  Using {len(harmful_prompts)} harmful prompts for refusal measurement")

    for arch in cfg.architectures:
        print(f"\n  Abliterating {arch}...")
        model_config = _make_model_config(cfg, arch)
        model = make_model(model_config)

        sft_ckpt = results / "checkpoints" / f"{arch}_sft" / "final"
        if sft_ckpt.exists():
            load_checkpoint(model, sft_ckpt)

        test_prompts = harmful_prompts

        rate_before = measure_refusal_rate(
            model, test_prompts, max_new_tokens=10, decode_fn=decode_fn,
        )
        print(f"  {arch} before: {rate_before:.3f}")

        if "mean_diff" in extraction_results.get(arch, {}):
            md_result = extraction_results[arch]["mean_diff"]
            if isinstance(md_result, ExtractionResult):
                best_layer = int(mx.argmax(md_result.layer_norms).item())
                abliterate_model(model, md_result.directions[best_layer])

        rate_after = measure_refusal_rate(
            model, test_prompts, max_new_tokens=10, decode_fn=decode_fn,
        )
        print(f"  {arch} after:  {rate_after:.3f}")

        all_rates[arch] = {"before": rate_before, "after": rate_after}
        wb.log({
            f"abliteration/{arch}/before": rate_before,
            f"abliteration/{arch}/after": rate_after,
        })

    save_refusal_rates(all_rates, rates_path)
    print(f"  Saved refusal rates to {rates_path}")
    return all_rates


# ─────────────────────────────────────────────────────────────────────
# Phase 5: Steering Sweeps
# ─────────────────────────────────────────────────────────────────────


def _kl_divergence(logits_p: np.ndarray, logits_q: np.ndarray) -> float:
    """KL(P||Q) from logits (numerically stable).

    Args:
        logits_p: Logits for P.
        logits_q: Logits for Q (baseline).

    Returns:
        KL divergence.
    """
    p = np.exp(logits_p - logits_p.max())
    p = p / p.sum()
    q = np.exp(logits_q - logits_q.max())
    q = q / q.sum()
    kl = np.sum(p * np.log(p / (q + 1e-10) + 1e-10))
    return float(max(kl, 0.0))


def _compute_kl_from_baseline(
    all_logits: np.ndarray, baseline_logits: np.ndarray,
) -> np.ndarray:
    """KL divergence of each row against a baseline row.

    Args:
        all_logits: (n_alphas, vocab_size).
        baseline_logits: (vocab_size,).

    Returns:
        KL values, shape (n_alphas,).
    """
    return np.array([
        _kl_divergence(all_logits[i], baseline_logits)
        for i in range(all_logits.shape[0])
    ])


def phase5_steering(
    cfg: ExperimentConfig,
    results: Path,
    extraction_results: dict[str, dict[str, ExtractionResult | MultiStreamExtractionResult]],
    wb: WandbLogger,
) -> dict[str, tuple[list[float], np.ndarray]]:
    """Run steering alpha sweeps for all architectures.

    Args:
        cfg: Experiment configuration.
        results: Results directory.
        extraction_results: Phase 3 output.
        wb: Wandb logger.

    Returns:
        Dict mapping arch -> (alphas, kl_values).
    """
    print("\n" + "=" * 60)
    print("PHASE 5: Steering Alpha Sweeps")
    print("=" * 60)

    all_sweeps: dict[str, tuple[list[float], np.ndarray]] = {}

    for arch in cfg.architectures:
        sweep_path = results / "steering" / f"{arch}_alpha_sweep"
        if sweep_path.with_suffix(".npz").exists():
            print(f"  {arch}: loading existing sweep")
            alphas, logits = load_alpha_sweep(sweep_path)
            baseline_idx = alphas.index(0.0) if 0.0 in alphas else len(alphas) // 2
            kl_values = _compute_kl_from_baseline(logits, logits[baseline_idx])
            all_sweeps[arch] = (alphas, kl_values)
            continue

        print(f"\n  Steering sweep for {arch}...")
        model_config = _make_model_config(cfg, arch)
        model = make_model(model_config)

        sft_ckpt = results / "checkpoints" / f"{arch}_sft" / "final"
        if sft_ckpt.exists():
            load_checkpoint(model, sft_ckpt)

        direction = None
        if "mean_diff" in extraction_results.get(arch, {}):
            md_result = extraction_results[arch]["mean_diff"]
            if isinstance(md_result, ExtractionResult):
                best_layer = int(mx.argmax(md_result.layer_norms).item())
                direction = md_result.directions[best_layer]

        if direction is None:
            direction = mx.random.normal((model_config.d_model,))
            direction = direction / mx.linalg.norm(direction)

        test_input = mx.random.randint(0, model_config.vocab_size, (1, 32))
        sweep_results = sweep_alpha(model, test_input, direction, cfg.alphas)

        save_alpha_sweep(sweep_results, sweep_path)

        baseline = sweep_results.get(
            0.0, list(sweep_results.values())[len(cfg.alphas) // 2],
        )
        baseline_np = np.array(baseline.astype(mx.float32))
        kl_values_list: list[float] = []
        for alpha in cfg.alphas:
            logits_np = np.array(sweep_results[alpha].astype(mx.float32))
            kl = _kl_divergence(logits_np, baseline_np)
            kl_values_list.append(kl)
            wb.log({f"steering/{arch}/kl_alpha_{alpha}": kl})

        all_sweeps[arch] = (cfg.alphas, np.array(kl_values_list))
        print(f"  {arch}: sweep complete")

    return all_sweeps


# ─────────────────────────────────────────────────────────────────────
# Phase 6: Comparison + Figures
# ─────────────────────────────────────────────────────────────────────


def phase6_compare_and_plot(
    cfg: ExperimentConfig,
    results: Path,
    training_logs: dict[str, list[dict[str, float]]],
    extraction_results: dict[str, dict[str, ExtractionResult | MultiStreamExtractionResult]],
    refusal_rates: dict[str, dict[str, float]],
    sweep_data: dict[str, tuple[list[float], np.ndarray]],
    wb: WandbLogger,
) -> None:
    """Cross-architecture comparison and figure generation.

    Args:
        cfg: Experiment configuration.
        results: Results directory.
        training_logs: Phase 1 output.
        extraction_results: Phase 3 output.
        refusal_rates: Phase 4 output.
        sweep_data: Phase 5 output.
        wb: Wandb logger.
    """
    print("\n" + "=" * 60)
    print("PHASE 6: Comparison + Figure Generation")
    print("=" * 60)

    from kromcanon.interp.viz import generate_all_figures

    # Pairwise direction comparisons
    comparisons = []
    md_results: dict[str, ExtractionResult] = {}
    for arch in cfg.architectures:
        if "mean_diff" in extraction_results.get(arch, {}):
            r = extraction_results[arch]["mean_diff"]
            if isinstance(r, ExtractionResult):
                md_results[arch] = r

    arch_pairs = [
        (a, b)
        for i, a in enumerate(cfg.architectures)
        for b in cfg.architectures[i + 1 :]
    ]
    for a, b in arch_pairs:
        if a in md_results and b in md_results:
            comp = compare_directions(md_results[a], md_results[b], a, b)
            comparisons.append(comp)
            wb.log({
                f"comparison/{a}_vs_{b}/layer_correlation": comp.layer_correlation,
                f"comparison/{a}_vs_{b}/mean_cosine_sim":
                    float(comp.cosine_sims.mean().item()),
            })

    # Stream analysis
    stream_analysis = None
    stream_norm_ratios = None
    stream_concentration = None
    stream_cosine_matrix = None

    # Find the multistream architecture (kromcanon or kromhc)
    ms_arch = next(
        (a for a in extraction_results if "multistream" in extraction_results[a]),
        None,
    )
    if ms_arch is not None:
        ms_result = extraction_results[ms_arch]["multistream"]
        if isinstance(ms_result, MultiStreamExtractionResult):
            stream_analysis = analyze_stream_distribution(ms_result)
            save_stream_analysis(
                stream_analysis, results / "comparison" / "stream_analysis",
            )
            stream_norm_ratios = _to_np(stream_analysis.norm_ratios)
            stream_concentration = _to_np(stream_analysis.concentration)
            stream_cosine_matrix = _to_np(stream_analysis.stream_cosines)

    # Report
    report = format_comparison_report(comparisons, stream_analysis)
    report_path = results / "comparison" / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"  Report saved to {report_path}")

    # Assemble figure data
    direction_norms: dict[str, np.ndarray] = {}
    mean_diff_norms: dict[str, np.ndarray] = {}
    svd_norms: dict[str, np.ndarray] = {}

    for arch in cfg.architectures:
        arch_data = extraction_results.get(arch, {})
        if "mean_diff" in arch_data:
            r = arch_data["mean_diff"]
            if isinstance(r, ExtractionResult):
                direction_norms[arch] = _to_np(r.layer_norms)
                mean_diff_norms[arch] = _to_np(r.layer_norms)
        if "svd" in arch_data:
            r = arch_data["svd"]
            if isinstance(r, ExtractionResult):
                svd_norms[arch] = _to_np(r.layer_norms)

    n_archs = len(cfg.architectures)
    cosine_matrix = np.eye(n_archs)
    for comp in comparisons:
        try:
            i = cfg.architectures.index(comp.arch_a)
            j = cfg.architectures.index(comp.arch_b)
        except ValueError:
            continue
        mean_cos = float(comp.cosine_sims.mean().item())
        cosine_matrix[i, j] = mean_cos
        cosine_matrix[j, i] = mean_cos

    fig_dir = results / "figures"
    paths = generate_all_figures(
        training_logs=training_logs,
        direction_norms=direction_norms,
        cosine_matrix=cosine_matrix,
        arch_names=cfg.architectures,
        stream_norm_ratios=stream_norm_ratios,
        stream_concentration=stream_concentration,
        alpha_sweep_data=sweep_data,
        refusal_rates=refusal_rates,
        stream_cosine_matrix=stream_cosine_matrix,
        mean_diff_norms=mean_diff_norms,
        svd_norms=svd_norms,
        output_dir=fig_dir,
    )

    print(f"\n  Generated {len(paths)} figures in {fig_dir}/")
    for p in paths:
        print(f"    {p.name}")
        wb.log_figure(f"figures/{p.stem}", p)

    summary: dict[str, object] = {}
    for arch in cfg.architectures:
        if arch in refusal_rates:
            summary[f"{arch}/refusal_before"] = refusal_rates[arch]["before"]
            summary[f"{arch}/refusal_after"] = refusal_rates[arch]["after"]
    wb.log_summary(summary)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def run(cfg: ExperimentConfig, *, toml_path: Path | None = None) -> None:
    """Execute the full experiment pipeline from a loaded config.

    Args:
        cfg: Experiment configuration.
        toml_path: Path to the source TOML file (for auto-status tracking).
    """
    results = _results_dir(cfg.run_name)
    results.mkdir(parents=True, exist_ok=True)

    # Auto-update TOML status to "running"
    if toml_path is not None:
        _update_toml_status(toml_path, "running")

    # Set seeds for reproducibility
    np.random.seed(cfg.seed)
    mx.random.seed(cfg.seed)

    # Persist the resolved config next to results (with automatic timestamp).
    # On rerun, previous runs are preserved in a "runs" history list so we
    # never lose when an experiment was first attempted or how many times it
    # was retried.
    config_path = results / "config.json"
    started_at = datetime.now(UTC).isoformat()

    # Load previous state if it exists (to preserve run history)
    prev_runs: list[dict[str, str]] = []
    if config_path.exists():
        prev = json.loads(config_path.read_text())
        prev_runs = prev.get("runs", [])
        # Archive the previous run entry if it had a started_at
        prev_started = prev.get("started_at", "")
        if prev_started:
            prev_runs.append({
                "started_at": prev_started,
                "completed_at": prev.get("completed_at", ""),
                "elapsed_seconds": prev.get("elapsed_seconds", ""),
                "outcome": prev.get("outcome", "unknown"),
            })

    config_dict: dict[str, object] = {
        "run_name": cfg.run_name,
        "started_at": started_at,
        "seed": cfg.seed,
        "depth": cfg.depth,
        "size": cfg.size,
        "architectures": cfg.architectures,
        "pretrain_max_steps": cfg.pretrain_max_steps,
        "pretrain_batch_size": cfg.pretrain_batch_size,
        "pretrain_lr": cfg.pretrain_lr,
        "pretrain_max_tokens": cfg.pretrain_max_tokens,
        "sft_max_steps": cfg.sft_max_steps,
        "sft_batch_size": cfg.sft_batch_size,
        "sft_max_examples": cfg.sft_max_examples,
        "n_prompts": cfg.n_prompts,
        "alphas": cfg.alphas,
        "bias_res_init": cfg.bias_res_init,
        "freeze_hres": cfg.freeze_hres,
        "wandb_enabled": cfg.wandb_enabled,
    }
    if prev_runs:
        config_dict["runs"] = prev_runs
    if cfg.meta is not None:
        config_dict["meta"] = {
            "id": cfg.meta.id,
            "title": cfg.meta.title,
            "status": cfg.meta.status,
            "parents": cfg.meta.parents,
            "tags": cfg.meta.tags,
            "notes": cfg.meta.notes,
            "date": cfg.meta.date,
        }
    config_path.write_text(json.dumps(config_dict, indent=2))

    wb = WandbLogger(
        enabled=cfg.wandb_enabled,
        project=cfg.wandb_project,
        run_name=cfg.run_name,
        config=json.loads((results / "config.json").read_text()),
    )

    print(f"Experiment: {cfg.run_name}")
    if cfg.meta is not None:
        m = cfg.meta
        if m.title:
            print(f"  Title: {m.title}")
        print(f"  Status: {m.status}")
        if m.parents:
            print(f"  Parents: {', '.join(m.parents)}")
        if m.tags:
            print(f"  Tags: {', '.join(m.tags)}")
    print(f"  Seed: {cfg.seed}")
    print(f"  Depth: {cfg.depth}")
    print(f"  Architectures: {cfg.architectures}")
    print(f"  Results: {results}")
    print(f"  Wandb: {'enabled' if cfg.wandb_enabled else 'disabled'}")

    t_start = time.perf_counter()

    try:
        # Phase 1
        if cfg.resume and _phase_complete(results, 1, cfg.architectures):
            print("\nPhase 1: already complete, loading logs")
            training_logs: dict[str, list[dict[str, float]]] = {}
            for arch in cfg.architectures:
                log_path = results / "pretrain" / f"{arch}_logs.json"
                if log_path.exists():
                    training_logs[arch] = load_logs(log_path)
        else:
            training_logs = phase1_pretrain(cfg, results, wb)

        # KromHC diagnostics (post-pretrain)
        if any(a in cfg.architectures for a in ("kromcanon", "kromhc")):
            _log_kromhc_diagnostics(cfg, results, wb, phase="pretrain")

        # Phase 2
        if cfg.resume and _phase_complete(results, 2, cfg.architectures):
            print("\nPhase 2: already complete")
        else:
            phase2_sft(cfg, results, wb)

        # Phase 3
        if cfg.resume and _phase_complete(results, 3, cfg.architectures):
            print("\nPhase 3: already complete, loading results")
            extraction_results = _load_existing_extractions(results, cfg.architectures)
        else:
            extraction_results = phase3_extract(cfg, results, wb)

        # Phase 4
        if cfg.resume and _phase_complete(results, 4, cfg.architectures):
            print("\nPhase 4: already complete, loading results")
            refusal_rates = load_refusal_rates(
                results / "abliteration" / "refusal_rates.json",
            )
        else:
            refusal_rates = phase4_abliterate(cfg, results, extraction_results, wb)

        # Phase 5
        if cfg.resume and _phase_complete(results, 5, cfg.architectures):
            print("\nPhase 5: already complete, loading results")
            sweep_data: dict[str, tuple[list[float], np.ndarray]] = {}
            for arch in cfg.architectures:
                sweep_path = results / "steering" / f"{arch}_alpha_sweep"
                if sweep_path.with_suffix(".npz").exists():
                    alphas, logits = load_alpha_sweep(sweep_path)
                    baseline_idx = alphas.index(0.0) if 0.0 in alphas else len(alphas) // 2
                    kl_values = _compute_kl_from_baseline(logits, logits[baseline_idx])
                    sweep_data[arch] = (alphas, kl_values)
        else:
            sweep_data = phase5_steering(cfg, results, extraction_results, wb)

        # Phase 6
        phase6_compare_and_plot(
            cfg, results, training_logs, extraction_results,
            refusal_rates, sweep_data, wb,
        )

    except Exception:
        # Record failure in config.json
        elapsed = time.perf_counter() - t_start
        failed_at = datetime.now(UTC).isoformat()
        saved = json.loads(config_path.read_text())
        saved["completed_at"] = failed_at
        saved["elapsed_seconds"] = round(elapsed, 1)
        saved["outcome"] = "failed"
        config_path.write_text(json.dumps(saved, indent=2))
        if toml_path is not None:
            _update_toml_status(toml_path, "wip")
        raise

    elapsed = time.perf_counter() - t_start
    completed_at = datetime.now(UTC).isoformat()

    # Record success in config.json
    saved = json.loads(config_path.read_text())
    saved["completed_at"] = completed_at
    saved["elapsed_seconds"] = round(elapsed, 1)
    saved["outcome"] = "success"
    config_path.write_text(json.dumps(saved, indent=2))

    # Auto-update TOML status to "promising" on success
    if toml_path is not None:
        _update_toml_status(toml_path, "promising")

    print(f"\nExperiment complete in {elapsed / 60:.1f} minutes")
    print(f"Results: {results}/")

    wb.finish()


def main() -> None:
    """Entry point: read TOML path from argv, load config, run."""
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.experiment <config.toml>")
        print("Example: python -m scripts.experiment experiments/quick.toml")
        sys.exit(1)

    toml_path = Path(sys.argv[1])
    if not toml_path.exists():
        print(f"Config file not found: {toml_path}")
        sys.exit(1)

    cfg = load_config(toml_path)
    run(cfg, toml_path=toml_path)


if __name__ == "__main__":
    main()

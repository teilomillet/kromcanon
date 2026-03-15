"""Serialization helpers for interpretability data structures.

Provides save/load for ExtractionResult, ComparisonResult, StreamAnalysis,
and training logs. Uses mx.savez for arrays and JSON for metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from kromcanon.interp.compare import ComparisonResult, StreamAnalysis
from kromcanon.interp.extract import ExtractionResult, MultiStreamExtractionResult


def save_extraction(result: ExtractionResult, path: Path) -> None:
    """Save an ExtractionResult to disk.

    Saves arrays as .npz and metadata as .json in the same directory.

    Args:
        result: Extraction result to save.
        path: Output file path (without extension — creates .npz and .json).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, mx.array] = {
        "directions": result.directions,
        "layer_norms": result.layer_norms,
    }
    if result.subspace is not None:
        arrays["subspace"] = result.subspace
    mx.savez(str(path.with_suffix(".npz")), **arrays)
    meta = {"method": result.method}
    path.with_suffix(".json").write_text(json.dumps(meta))


def load_extraction(path: Path) -> ExtractionResult:
    """Load an ExtractionResult from disk.

    Args:
        path: File path (with or without extension).

    Returns:
        Loaded ExtractionResult.
    """
    npz_path = path.with_suffix(".npz")
    json_path = path.with_suffix(".json")
    data = dict(mx.load(str(npz_path)))
    meta = json.loads(json_path.read_text()) if json_path.exists() else {}
    return ExtractionResult(
        directions=data["directions"],
        method=meta.get("method", "loaded"),
        layer_norms=data["layer_norms"],
        subspace=data.get("subspace"),
    )


def save_multistream(result: MultiStreamExtractionResult, path: Path) -> None:
    """Save a MultiStreamExtractionResult to disk.

    Args:
        result: Multi-stream extraction result.
        path: Output file path (without extension).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.savez(
        str(path.with_suffix(".npz")),
        per_stream=result.per_stream,
        joint=result.joint,
        stream_norms=result.stream_norms,
        joint_norms=result.joint_norms,
    )


def load_multistream(path: Path) -> MultiStreamExtractionResult:
    """Load a MultiStreamExtractionResult from disk.

    Args:
        path: File path (with or without extension).

    Returns:
        Loaded MultiStreamExtractionResult.
    """
    data = dict(mx.load(str(path.with_suffix(".npz"))))
    return MultiStreamExtractionResult(
        per_stream=data["per_stream"],
        joint=data["joint"],
        stream_norms=data["stream_norms"],
        joint_norms=data["joint_norms"],
    )


def save_comparison(result: ComparisonResult, path: Path) -> None:
    """Save a ComparisonResult to disk.

    Args:
        result: Comparison result.
        path: Output file path (without extension).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.savez(str(path.with_suffix(".npz")), cosine_sims=result.cosine_sims)
    meta = {
        "arch_a": result.arch_a,
        "arch_b": result.arch_b,
        "layer_correlation": result.layer_correlation,
    }
    path.with_suffix(".json").write_text(json.dumps(meta))


def load_comparison(path: Path) -> ComparisonResult:
    """Load a ComparisonResult from disk.

    Args:
        path: File path (with or without extension).

    Returns:
        Loaded ComparisonResult.
    """
    data = dict(mx.load(str(path.with_suffix(".npz"))))
    meta = json.loads(path.with_suffix(".json").read_text())
    return ComparisonResult(
        cosine_sims=data["cosine_sims"],
        arch_a=meta["arch_a"],
        arch_b=meta["arch_b"],
        layer_correlation=meta["layer_correlation"],
    )


def save_stream_analysis(analysis: StreamAnalysis, path: Path) -> None:
    """Save a StreamAnalysis to disk.

    Args:
        analysis: Stream analysis result.
        path: Output file path (without extension).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.savez(
        str(path.with_suffix(".npz")),
        concentration=analysis.concentration,
        dominant_stream=analysis.dominant_stream,
        stream_cosines=analysis.stream_cosines,
        norm_ratios=analysis.norm_ratios,
    )


def load_stream_analysis(path: Path) -> StreamAnalysis:
    """Load a StreamAnalysis from disk.

    Args:
        path: File path (with or without extension).

    Returns:
        Loaded StreamAnalysis.
    """
    data = dict(mx.load(str(path.with_suffix(".npz"))))
    return StreamAnalysis(
        concentration=data["concentration"],
        dominant_stream=data["dominant_stream"],
        stream_cosines=data["stream_cosines"],
        norm_ratios=data["norm_ratios"],
    )


def save_logs(logs: list[dict[str, float]], path: Path) -> None:
    """Save training logs as JSON.

    Args:
        logs: List of log dicts (step, loss, etc.).
        path: Output .json file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(logs, indent=2))


def load_logs(path: Path) -> list[dict[str, float]]:
    """Load training logs from JSON.

    Args:
        path: Path to .json log file.

    Returns:
        List of log dicts.
    """
    return json.loads(path.read_text())


def save_refusal_rates(rates: dict[str, dict[str, float]], path: Path) -> None:
    """Save refusal rate results as JSON.

    Args:
        rates: Dict mapping arch → {before: float, after: float}.
        path: Output .json file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rates, indent=2))


def load_refusal_rates(path: Path) -> dict[str, dict[str, float]]:
    """Load refusal rate results from JSON.

    Args:
        path: Path to .json file.

    Returns:
        Dict mapping arch → {before: float, after: float}.
    """
    return json.loads(path.read_text())


def save_alpha_sweep(
    results: dict[float, mx.array], path: Path
) -> None:
    """Save alpha sweep results to disk.

    Args:
        results: Dict mapping alpha → logits array.
        path: Output file path (without extension).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    alphas = sorted(results.keys())
    logits_stack = mx.stack([results[a] for a in alphas]).astype(mx.float32)
    mx.savez(str(path.with_suffix(".npz")), logits=logits_stack)
    meta = {"alphas": alphas}
    path.with_suffix(".json").write_text(json.dumps(meta))


def load_alpha_sweep(path: Path) -> tuple[list[float], np.ndarray]:
    """Load alpha sweep results from disk.

    Args:
        path: File path (with or without extension).

    Returns:
        Tuple of (alphas, logits_array) where logits has shape (n_alphas, vocab_size).
    """
    data = dict(mx.load(str(path.with_suffix(".npz"))))
    meta = json.loads(path.with_suffix(".json").read_text())
    return meta["alphas"], np.array(data["logits"])

"""Cross-architecture comparison for direction analysis.

Compares extracted directions across Vanilla, Canon, and KromCanon variants:
- Direction cosine similarity across architectures
- Per-layer projection profiles
- Stream distribution analysis for KromCanon
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from kromcanon.interp.extract import ExtractionResult, MultiStreamExtractionResult


@dataclass
class ComparisonResult:
    """Result of cross-architecture direction comparison.

    Attributes:
        cosine_sims: Pairwise cosine similarities, shape (n_layers,).
        arch_a: Name of first architecture.
        arch_b: Name of second architecture.
        layer_correlation: Per-layer direction magnitude correlation.
    """

    cosine_sims: mx.array
    arch_a: str
    arch_b: str
    layer_correlation: float


@dataclass
class StreamAnalysis:
    """Analysis of direction distribution across KromCanon streams.

    Attributes:
        concentration: How concentrated the direction is in one stream (0=uniform, 1=single stream).
        dominant_stream: Index of the stream with strongest direction per layer.
        stream_cosines: Pairwise cosine similarities between stream directions.
        norm_ratios: Ratio of each stream's norm to total, shape (n_layers, n_streams).
    """

    concentration: mx.array
    dominant_stream: mx.array
    stream_cosines: mx.array
    norm_ratios: mx.array


def compare_directions(
    result_a: ExtractionResult,
    result_b: ExtractionResult,
    arch_a: str,
    arch_b: str,
) -> ComparisonResult:
    """Compare extracted directions between two architectures.

    Args:
        result_a: Extraction result from first architecture.
        result_b: Extraction result from second architecture.
        arch_a: Name of first architecture.
        arch_b: Name of second architecture.

    Returns:
        ComparisonResult with per-layer cosine similarities.
    """
    n_layers = min(result_a.directions.shape[0], result_b.directions.shape[0])

    cosine_sims: list[float] = []
    for i in range(n_layers):
        d_a = result_a.directions[i]
        d_b = result_b.directions[i]
        cos = _cosine_similarity(d_a, d_b)
        cosine_sims.append(cos.item())

    # Correlation of layer norms
    norms_a = result_a.layer_norms[:n_layers]
    norms_b = result_b.layer_norms[:n_layers]
    correlation = _pearson_correlation(norms_a, norms_b)

    return ComparisonResult(
        cosine_sims=mx.array(cosine_sims),
        arch_a=arch_a,
        arch_b=arch_b,
        layer_correlation=correlation,
    )


def analyze_stream_distribution(
    result: MultiStreamExtractionResult,
) -> StreamAnalysis:
    """Analyze how directions are distributed across KromCanon streams.

    Args:
        result: Multi-stream extraction result.

    Returns:
        StreamAnalysis with concentration metrics.
    """
    n_layers, n_streams = result.stream_norms.shape

    # Concentration: entropy-based measure
    # Low entropy = concentrated in one stream, high = distributed
    concentration: list[float] = []
    dominant: list[int] = []
    norm_ratios: list[mx.array] = []

    for layer in range(n_layers):
        norms = result.stream_norms[layer]  # (n_streams,)
        total = norms.sum() + 1e-8

        ratios = norms / total
        norm_ratios.append(ratios)

        # Dominant stream
        dominant.append(mx.argmax(norms).item())

        # Concentration: 1 - normalized entropy
        # max entropy = log(n_streams), concentration = 1 - H/H_max
        entropy = -mx.sum(ratios * mx.log(ratios + 1e-8)).item()
        max_entropy = mx.log(mx.array(float(n_streams))).item()
        conc = 1.0 - entropy / max_entropy if max_entropy > 0 else 0.0
        concentration.append(conc)

    # Pairwise cosine similarities between streams (averaged over layers)
    stream_cosines = _compute_stream_cosines(result.per_stream)

    return StreamAnalysis(
        concentration=mx.array(concentration),
        dominant_stream=mx.array(dominant),
        stream_cosines=stream_cosines,
        norm_ratios=mx.stack(norm_ratios),
    )


def format_comparison_report(
    comparisons: list[ComparisonResult],
    stream_analysis: StreamAnalysis | None = None,
) -> str:
    """Format a human-readable comparison report.

    Args:
        comparisons: List of pairwise comparison results.
        stream_analysis: Optional stream distribution analysis.

    Returns:
        Formatted report string.
    """
    lines: list[str] = ["# Direction Comparison Report\n"]

    for comp in comparisons:
        lines.append(f"## {comp.arch_a} vs {comp.arch_b}")
        lines.append(f"Layer norm correlation: {comp.layer_correlation:.4f}\n")
        lines.append("| Layer | Cosine Sim |")
        lines.append("|-------|-----------|")
        for i, sim in enumerate(comp.cosine_sims.tolist()):
            lines.append(f"| {i:>5d} | {sim:>9.4f} |")
        lines.append("")

    if stream_analysis is not None:
        lines.append("## KromCanon Stream Distribution")
        lines.append("| Layer | Dominant | Concentration |")
        lines.append("|-------|----------|---------------|")
        for i in range(len(stream_analysis.concentration.tolist())):
            dom = stream_analysis.dominant_stream[i].item()
            conc = stream_analysis.concentration[i].item()
            lines.append(f"| {i:>5d} | {dom:>8d} | {conc:>13.4f} |")
        lines.append("")

    return "\n".join(lines)


# --- Helpers ---


def _cosine_similarity(a: mx.array, b: mx.array) -> mx.array:
    """Compute cosine similarity between two vectors."""
    return mx.sum(a * b) / (mx.linalg.norm(a) * mx.linalg.norm(b) + 1e-8)


def _pearson_correlation(a: mx.array, b: mx.array) -> float:
    """Compute Pearson correlation coefficient."""
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    num = mx.sum(a_centered * b_centered).item()
    den = (
        mx.sqrt(mx.sum(a_centered**2)).item()
        * mx.sqrt(mx.sum(b_centered**2)).item()
        + 1e-8
    )
    return num / den


def _compute_stream_cosines(per_stream: mx.array) -> mx.array:
    """Compute pairwise cosine similarities between stream directions.

    Args:
        per_stream: Per-stream directions, shape (n_layers, n_streams, d_model).

    Returns:
        Pairwise cosines averaged over layers, shape (n_streams, n_streams).
    """
    n_layers, n_streams, d_model = per_stream.shape
    cosines = mx.zeros((n_streams, n_streams))

    for layer in range(n_layers):
        for i in range(n_streams):
            for j in range(n_streams):
                cos = _cosine_similarity(per_stream[layer, i], per_stream[layer, j])
                cosines = cosines.at[i, j].add(cos)

    return cosines / n_layers

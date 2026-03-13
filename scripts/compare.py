"""CLI for cross-architecture direction comparison.

Usage:
    python -m scripts.compare \
        --vanilla=directions/vanilla_mean_diff.npz \
        --canon=directions/canon_mean_diff.npz \
        --kromcanon=directions/kromcanon_multistream.npz
"""

import argparse
from pathlib import Path

import mlx.core as mx

from kromcanon.interp.compare import (
    ComparisonResult,
    StreamAnalysis,
    analyze_stream_distribution,
    compare_directions,
    format_comparison_report,
)
from kromcanon.interp.extract import ExtractionResult, MultiStreamExtractionResult


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare directions across architectures")
    parser.add_argument("--vanilla", type=str, help="Path to vanilla directions .npz")
    parser.add_argument("--canon", type=str, help="Path to canon directions .npz")
    parser.add_argument("--kromcanon", type=str, help="Path to kromcanon directions .npz")
    parser.add_argument("--output", type=str, default="comparison_report.md")
    return parser.parse_args()


def load_extraction(path: str) -> ExtractionResult:
    """Load extraction result from .npz file."""
    data = dict(mx.load(path))
    return ExtractionResult(
        directions=data["directions"],
        method="loaded",
        layer_norms=data["layer_norms"],
    )


def load_multistream_extraction(path: str) -> MultiStreamExtractionResult:
    """Load multi-stream extraction result from .npz file."""
    data = dict(mx.load(path))
    return MultiStreamExtractionResult(
        per_stream=data["per_stream"],
        joint=data["joint"],
        stream_norms=data["stream_norms"],
        joint_norms=data["joint_norms"],
    )


def main() -> None:
    """Main comparison entry point."""
    args = parse_args()

    comparisons: list[ComparisonResult] = []
    stream_analysis: StreamAnalysis | None = None

    results: dict[str, ExtractionResult] = {}
    if args.vanilla:
        results["vanilla"] = load_extraction(args.vanilla)
    if args.canon:
        results["canon"] = load_extraction(args.canon)

    # Pairwise comparisons
    pairs = [("vanilla", "canon")]
    for arch_a, arch_b in pairs:
        if arch_a in results and arch_b in results:
            comp = compare_directions(
                results[arch_a], results[arch_b], arch_a, arch_b
            )
            comparisons.append(comp)

    # KromCanon vs others (using joint direction)
    if args.kromcanon:
        ms_result = load_multistream_extraction(args.kromcanon)
        # Create ExtractionResult from joint directions for comparison
        krom_result = ExtractionResult(
            directions=ms_result.joint,
            method="joint",
            layer_norms=ms_result.joint_norms,
        )
        for arch_name, result in results.items():
            comp = compare_directions(result, krom_result, arch_name, "kromcanon")
            comparisons.append(comp)

        # Stream distribution analysis
        stream_analysis = analyze_stream_distribution(ms_result)

    # Generate report
    report = format_comparison_report(comparisons, stream_analysis)
    print(report)

    # Save report
    Path(args.output).write_text(report)
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()

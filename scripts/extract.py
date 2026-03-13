"""CLI for direction extraction from trained models.

Usage:
    python -m scripts.extract --arch=vanilla --checkpoint=checkpoints/vanilla_sft/final
    python -m scripts.extract --arch=kromcanon --checkpoint=checkpoints/kromcanon_sft/final
"""

import argparse
from pathlib import Path

import mlx.core as mx

from kromcanon.config import make_config
from kromcanon.interp.extract import (
    collect_activations,
    collect_multistream_activations,
    extract_mean_diff,
    extract_multistream_directions,
    extract_svd,
)
from kromcanon.model import make_model
from kromcanon.train import load_checkpoint


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract refusal directions")
    parser.add_argument(
        "--arch", type=str, required=True,
        choices=["vanilla", "canon", "kromcanon"],
    )
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--method", type=str, default="mean_diff",
        choices=["mean_diff", "svd"],
    )
    parser.add_argument("--output", type=str, default="directions")
    parser.add_argument("--n-prompts", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    """Main extraction entry point."""
    args = parse_args()

    config = make_config(arch=args.arch, depth=args.depth)
    model = make_model(config)
    load_checkpoint(model, Path(args.checkpoint))
    print(f"Loaded {args.arch} model from {args.checkpoint}")

    # Generate synthetic test prompts (placeholder — real usage loads actual prompts)
    harmful_ids = [mx.random.randint(0, config.vocab_size, (1, 64)) for _ in range(args.n_prompts)]
    harmless_ids = [mx.random.randint(0, config.vocab_size, (1, 64)) for _ in range(args.n_prompts)]

    if args.arch == "kromcanon":
        print("Collecting multi-stream activations...")
        harmful_acts = collect_multistream_activations(model, harmful_ids)
        harmless_acts = collect_multistream_activations(model, harmless_ids)
        result = extract_multistream_directions(harmful_acts, harmless_acts)
        out_path = Path(args.output) / f"{args.arch}_multistream.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mx.savez(
            str(out_path),
            per_stream=result.per_stream,
            joint=result.joint,
            stream_norms=result.stream_norms,
            joint_norms=result.joint_norms,
        )
        print(f"Saved multi-stream directions to {out_path}")
    else:
        print(f"Collecting activations ({args.method})...")
        harmful_acts = collect_activations(model, harmful_ids)
        harmless_acts = collect_activations(model, harmless_ids)

        if args.method == "mean_diff":
            result = extract_mean_diff(harmful_acts, harmless_acts)
        else:
            result = extract_svd(harmful_acts, harmless_acts)

        out_path = Path(args.output) / f"{args.arch}_{args.method}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mx.savez(
            str(out_path),
            directions=result.directions,
            layer_norms=result.layer_norms,
        )
        print(f"Saved directions to {out_path}")


if __name__ == "__main__":
    main()

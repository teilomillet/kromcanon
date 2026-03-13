"""Training CLI for KromCanon models.

Usage:
    python -m scripts.train --arch=vanilla --depth=12
    python -m scripts.train --arch=canon --depth=12
    python -m scripts.train --arch=kromcanon --depth=12
    python -m scripts.train --arch=vanilla --depth=4 --smoke  # quick smoke test
"""

import argparse

import mlx.core as mx

from kromcanon.config import ModelConfig, TrainConfig, make_config
from kromcanon.data import PretrainDataLoader, load_fineweb_edu, prepare_pretraining_data
from kromcanon.model import make_model
from kromcanon.train import train


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train KromCanon GPT-2 variants")
    parser.add_argument(
        "--arch",
        type=str,
        default="vanilla",
        choices=["vanilla", "canon", "kromcanon"],
        help="Architecture variant",
    )
    parser.add_argument("--depth", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=5000, help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1_200_000_000,
        help="Max tokens to load from dataset",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: tiny model, few steps, random data",
    )
    return parser.parse_args()


def run_smoke_test(arch: str) -> None:
    """Run a quick smoke test with random data.

    Args:
        arch: Architecture variant to test.
    """
    print(f"=== Smoke test: {arch} ===")
    config = ModelConfig(
        arch=arch,
        vocab_size=256,
        n_layers=2,
        n_heads=4,
        d_model=64,
        d_ff=256,
        max_seq_len=64,
    )
    model = make_model(config)

    # Count parameters
    import mlx.utils
    n_params = sum(v.size for _, v in mlx.utils.tree_flatten(model.trainable_parameters()))
    print(f"  Parameters: {n_params:,}")

    # Generate random data
    sequences = [
        mx.random.randint(0, 256, (64,)).tolist()
        for _ in range(128)
    ]
    loader = PretrainDataLoader(sequences, batch_size=8, shuffle=True)

    train_config = TrainConfig(
        batch_size=8,
        max_steps=20,
        lr=1e-3,
        warmup_steps=5,
        log_interval=5,
        eval_interval=100,
        save_interval=100,
        checkpoint_dir=f"checkpoints/smoke_{arch}",
    )

    logs = train(model, loader, config, train_config)
    print(f"  Final loss: {logs[-1]['loss']:.4f}")
    print("  Smoke test passed!")


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    if args.smoke:
        run_smoke_test(args.arch)
        return

    # Create model config
    model_config = make_config(arch=args.arch, depth=args.depth)

    # Create model
    model = make_model(model_config)
    import mlx.utils
    n_params = sum(v.size for _, v in mlx.utils.tree_flatten(model.trainable_parameters()))
    print(f"Model: {args.arch}, depth={args.depth}, params={n_params:,}")

    # Load and prepare data
    print("Loading FineWeb-Edu dataset...")
    texts = load_fineweb_edu()

    # We need a tokenizer — for now use a simple one
    # TODO: integrate proper BPE tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(f"Tokenizing and packing (max {args.max_tokens:,} tokens)...")
    sequences = prepare_pretraining_data(
        texts=texts,
        encode_fn=tokenizer.encode,
        seq_len=model_config.max_seq_len,
        max_tokens=args.max_tokens,
    )
    print(f"  Packed {len(sequences):,} sequences of length {model_config.max_seq_len}")

    # Split train/eval (99/1)
    n_eval = max(len(sequences) // 100, 1)
    eval_sequences = sequences[:n_eval]
    train_sequences = sequences[n_eval:]

    train_loader = PretrainDataLoader(train_sequences, batch_size=args.batch_size)
    eval_loader = PretrainDataLoader(eval_sequences, batch_size=args.batch_size, shuffle=False)

    # Train
    train_config = TrainConfig(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )

    train(model, train_loader, model_config, train_config, eval_loader=eval_loader)


if __name__ == "__main__":
    main()

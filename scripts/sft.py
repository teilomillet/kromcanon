"""Safety fine-tuning CLI for KromCanon models.

Usage:
    python -m scripts.sft --arch=vanilla --depth=12
    python -m scripts.sft --arch=canon --depth=12 --max-steps=500
    python -m scripts.sft --arch=kromcanon --depth=12
"""

import argparse
from pathlib import Path

from kromcanon.config import TrainConfig, make_config
from kromcanon.model import make_model
from kromcanon.safety_data import (
    iter_safety_batches,
    load_beavertails,
    load_hh_rlhf,
    tokenize_conversations,
)
from kromcanon.sft import sft_train
from kromcanon.train import load_checkpoint


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Safety fine-tune KromCanon models")
    parser.add_argument(
        "--arch", type=str, default="vanilla",
        choices=["vanilla", "canon", "kromcanon"],
        help="Architecture variant",
    )
    parser.add_argument("--depth", type=int, default=12, help="Number of layers")
    parser.add_argument("--max-steps", type=int, default=1000, help="SFT steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to pretrained checkpoint to load",
    )
    parser.add_argument(
        "--max-examples", type=int, default=10000,
        help="Max examples per dataset",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Checkpoint directory",
    )
    return parser.parse_args()


def main() -> None:
    """Main SFT entry point."""
    args = parse_args()

    # Create model
    model_config = make_config(arch=args.arch, depth=args.depth)
    model = make_model(model_config)

    # Load pretrained checkpoint if specified
    if args.checkpoint:
        step = load_checkpoint(model, Path(args.checkpoint))
        print(f"Loaded checkpoint from step {step}")

    # Load safety data
    print("Loading HH-RLHF dataset...")
    hh_pairs = load_hh_rlhf(max_examples=args.max_examples)
    print(f"  HH-RLHF: {len(hh_pairs)} pairs")

    print("Loading BeaverTails dataset...")
    bt_pairs = load_beavertails(max_examples=args.max_examples)
    print(f"  BeaverTails: {len(bt_pairs)} pairs")

    all_pairs = hh_pairs + bt_pairs
    print(f"  Total: {len(all_pairs)} pairs")

    # Tokenize — use GPT-2 tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    sequences = tokenize_conversations(
        all_pairs, encode_fn=tokenizer.encode, max_len=512
    )
    print(f"  Tokenized: {len(sequences)} sequences")

    # Create data iterator
    train_data = iter_safety_batches(
        sequences, batch_size=args.batch_size, seq_len=512
    )

    # Run SFT
    train_config = TrainConfig(
        batch_size=args.batch_size,
        lr=6e-4,
        checkpoint_dir=args.checkpoint_dir,
    )

    sft_train(
        model, train_data, model_config, train_config,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()

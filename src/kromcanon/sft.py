"""Supervised fine-tuning loop for safety training.

Fine-tunes pretrained models on safety contrast pairs to create
measurable refusal behavior for interpretability analysis.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from kromcanon.config import ModelConfig, TrainConfig
from kromcanon.model import GPT2
from kromcanon.train import compute_loss, save_checkpoint

if TYPE_CHECKING:
    from collections.abc import Iterator


def sft_train(
    model: GPT2,
    train_data: Iterator[tuple[mx.array, mx.array]],
    model_config: ModelConfig,
    train_config: TrainConfig,
    max_steps: int = 1000,
) -> list[dict[str, float]]:
    """Run supervised fine-tuning on safety data.

    Uses a lower learning rate than pretraining (1/10th default).

    Args:
        model: Pretrained GPT-2 model.
        train_data: Iterator yielding (input_ids, target_ids) batches.
        model_config: Model configuration.
        train_config: Training configuration.
        max_steps: Maximum SFT steps.

    Returns:
        List of training log entries.
    """
    # SFT uses lower LR
    sft_lr = train_config.lr / 10.0
    optimizer = optim.AdamW(learning_rate=sft_lr, weight_decay=0.01)

    checkpoint_dir = Path(train_config.checkpoint_dir) / f"{model_config.arch}_sft"
    logs: list[dict[str, float]] = []

    print(f"SFT: {model_config.arch} for {max_steps} steps (lr={sft_lr:.1e})")

    step = 0
    for input_ids, target_ids in train_data:
        if step >= max_steps:
            break

        t0 = time.perf_counter()

        loss_and_grad_fn = nn.value_and_grad(
            model, lambda m, x, y: compute_loss(m, x, y)
        )
        loss, grads = loss_and_grad_fn(model, input_ids, target_ids)
        grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        dt = time.perf_counter() - t0
        step += 1
        loss_val = loss.item()

        if step % 10 == 0:
            log_entry = {"step": step, "loss": loss_val, "time_ms": dt * 1000}
            logs.append(log_entry)
            print(f"  sft step {step:>4d} | loss {loss_val:.4f} | {dt*1000:.0f}ms")

    # Save SFT checkpoint
    save_checkpoint(
        model, optimizer, step, logs[-1]["loss"] if logs else 0.0,
        model_config, checkpoint_dir / "final",
    )
    print(f"SFT complete. Checkpoint saved to {checkpoint_dir}/final")

    return logs

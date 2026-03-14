"""Training loop for KromCanon models.

Supports all three architecture variants with appropriate optimizer grouping:
- Vanilla/Canon: Muon (2D weights) + AdamW (embeddings, 1D params)
- KromCanon: Muon (2D weights) + AdamW-HC (KromHC params) + AdamW (rest)

Uses mx.compile for fused GPU kernels in the training step.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import numpy as np

from kromcanon.config import ModelConfig, TrainConfig
from kromcanon.kromhc import extract_hres_metrics
from kromcanon.model import GPT2

if TYPE_CHECKING:
    from kromcanon.data import PretrainDataLoader


def compute_loss(
    model: GPT2, input_ids: mx.array, target_ids: mx.array
) -> mx.array:
    """Compute cross-entropy loss for next-token prediction.

    Args:
        model: The GPT-2 model.
        input_ids: Input token IDs, shape (batch, seq_len).
        target_ids: Target token IDs, shape (batch, seq_len).

    Returns:
        Scalar loss value.
    """
    logits = model(input_ids)  # (batch, seq_len, vocab_size)
    # Reshape for cross-entropy: (batch * seq_len, vocab_size)
    # Cast to float32 for numerical stability (model may be bfloat16)
    b, t, v = logits.shape
    loss = nn.losses.cross_entropy(
        logits.reshape(b * t, v).astype(mx.float32),
        target_ids.reshape(b * t),
        reduction="mean",
    )
    return loss


def _create_schedule(
    config: TrainConfig, lr: float, min_lr: float | None = None,
) -> optim.schedulers.cosine_decay:
    """Create cosine decay learning rate schedule with optional warmup.

    Args:
        config: Training configuration (uses warmup_steps and max_steps).
        lr: Peak learning rate.
        min_lr: End learning rate. Defaults to lr / 10.

    Returns:
        MLX learning rate schedule.
    """
    if min_lr is None:
        min_lr = lr / 10
    decay_steps = max(config.max_steps - config.warmup_steps, 1)
    cosine = optim.schedulers.cosine_decay(
        init=lr, decay_steps=decay_steps, end=min_lr,
    )
    if config.warmup_steps > 0:
        warmup = optim.schedulers.linear_schedule(
            init=1e-7, end=lr, steps=config.warmup_steps,
        )
        return optim.schedulers.join_schedules(
            schedules=[warmup, cosine],
            boundaries=[config.warmup_steps],
        )
    return cosine


def create_lr_schedule(
    config: TrainConfig,
) -> optim.schedulers.cosine_decay:
    """Create cosine decay learning rate schedule with warmup.

    Args:
        config: Training configuration.

    Returns:
        MLX learning rate schedule.
    """
    return _create_schedule(config, config.lr, config.min_lr)


def _is_muon_param(path: str, param: mx.array) -> bool:
    """Return True for 2D weight matrices suited for Muon.

    Excludes embeddings and LM head — those train better with AdamW.

    Args:
        path: Parameter path in the model tree.
        param: Parameter array.

    Returns:
        True if this parameter should use Muon.
    """
    if param.ndim != 2:
        return False
    return not any(name in path for name in ("wte", "wpe"))


def _is_hc_param(path: str, _param: mx.array) -> bool:
    """Return True for KromHC parameters.

    Args:
        path: Parameter path in the model tree.
        _param: Parameter array (unused).

    Returns:
        True if this is a KromHC parameter.
    """
    return "kromhc" in path


def create_optimizer(
    model: GPT2, model_config: ModelConfig, train_config: TrainConfig,
) -> optim.Optimizer:
    """Create optimizer with appropriate parameter grouping.

    When use_muon is True:
    - Muon for 2D weight matrices (Q/K/V/O projections, FFN layers)
    - AdamW for embeddings, 1D params, and LM head
    - For KromCanon: separate AdamW group for HC params with custom LR/betas

    Args:
        model: The GPT-2 model.
        model_config: Model configuration.
        train_config: Training configuration.

    Returns:
        Configured optimizer.
    """
    adamw_schedule = create_lr_schedule(train_config)

    if not train_config.use_muon:
        return optim.AdamW(
            learning_rate=adamw_schedule,
            weight_decay=train_config.weight_decay,
        )

    muon_schedule = _create_schedule(config=train_config, lr=train_config.muon_lr)
    muon = optim.Muon(
        learning_rate=muon_schedule,
        momentum=0.95,
        weight_decay=0.01,
        nesterov=True,
    )
    adamw = optim.AdamW(
        learning_rate=adamw_schedule,
        weight_decay=train_config.weight_decay,
    )

    if model_config.kromhc.enabled:
        # Three groups: Muon → 2D weights, HC-AdamW → KromHC params, AdamW → rest
        hc_schedule = _create_schedule(config=train_config, lr=train_config.hc_lr)
        hc_adamw = optim.AdamW(
            learning_rate=hc_schedule,
            betas=train_config.hc_betas,
            weight_decay=train_config.hc_weight_decay,
        )
        return optim.MultiOptimizer(
            optimizers=[muon, hc_adamw, adamw],
            filters=[
                lambda path, param: _is_muon_param(path, param) and not _is_hc_param(path, param),
                _is_hc_param,
            ],
        )

    # Two groups: Muon → 2D weights, AdamW → rest
    return optim.MultiOptimizer(
        optimizers=[muon, adamw],
        filters=[_is_muon_param],
    )


def save_checkpoint(
    model: GPT2,
    optimizer: optim.Optimizer,
    step: int,
    loss: float,
    config: ModelConfig,
    path: Path,
) -> None:
    """Save model checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer state to save.
        step: Current training step.
        loss: Current loss value.
        config: Model configuration.
        path: Directory to save checkpoint to.
    """
    path.mkdir(parents=True, exist_ok=True)
    # Save model weights
    weights = dict(mlx.utils.tree_flatten(model.trainable_parameters()))
    mx.savez(str(path / "model.npz"), **weights)
    # Save metadata
    meta = {
        "step": step,
        "loss": loss,
        "arch": config.arch,
        "n_layers": config.n_layers,
        "d_model": config.d_model,
    }
    np.save(str(path / "meta.npy"), meta)


def load_checkpoint(model: GPT2, path: Path) -> int:
    """Load model weights from checkpoint.

    Args:
        model: The model to load weights into.
        path: Directory containing the checkpoint.

    Returns:
        The training step at which the checkpoint was saved.
    """
    weights = dict(mx.load(str(path / "model.npz")))
    model.load_weights(list(weights.items()))
    meta = np.load(str(path / "meta.npy"), allow_pickle=True).item()
    return meta["step"]


def train_step(
    model: GPT2,
    input_ids: mx.array,
    target_ids: mx.array,
    optimizer: optim.Optimizer,
    grad_clip: float = 1.0,
) -> mx.array:
    """Execute a single training step (uncompiled, for tests and simple use).

    For production training, use ``train()`` which compiles the step.

    Args:
        model: The GPT-2 model.
        input_ids: Input token IDs, shape (batch, seq_len).
        target_ids: Target token IDs, shape (batch, seq_len).
        optimizer: The optimizer.
        grad_clip: Maximum gradient norm.

    Returns:
        Loss value for this step.
    """
    loss_and_grad_fn = nn.value_and_grad(model, compute_loss)
    loss, grads = loss_and_grad_fn(model, input_ids, target_ids)

    # Gradient clipping
    grads, _ = optim.clip_grad_norm(grads, max_norm=grad_clip)

    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    return loss


def train(
    model: GPT2,
    train_loader: PretrainDataLoader,
    model_config: ModelConfig,
    train_config: TrainConfig,
    eval_loader: PretrainDataLoader | None = None,
) -> list[dict[str, float]]:
    """Full training loop with compiled step and Muon optimizer.

    Args:
        model: The GPT-2 model.
        train_loader: Training data loader.
        model_config: Model configuration.
        train_config: Training configuration.
        eval_loader: Optional evaluation data loader.

    Returns:
        List of log dictionaries (step, loss, lr, time, etc.).
    """
    optimizer = create_optimizer(model, model_config, train_config)
    checkpoint_dir = Path(train_config.checkpoint_dir) / model_config.arch

    # Compiled training step — fuses GPU kernels for ~15-30% speedup
    loss_and_grad_fn = nn.value_and_grad(model, compute_loss)
    state = [model.state, optimizer.state]

    def _step(input_ids: mx.array, target_ids: mx.array) -> mx.array:
        loss, grads = loss_and_grad_fn(model, input_ids, target_ids)
        grads, _ = optim.clip_grad_norm(grads, max_norm=train_config.grad_clip)
        optimizer.update(model, grads)
        return loss

    compiled_step = mx.compile(_step, inputs=state, outputs=state)

    logs: list[dict[str, float]] = []
    step = 0
    epoch = 0

    optimizer_name = type(optimizer).__name__
    print(f"Training {model_config.arch} model for {train_config.max_steps} steps")
    print(f"  Optimizer: {optimizer_name}")
    print(f"  Batch size: {train_config.batch_size}")
    print(f"  Learning rate: {train_config.lr} (Muon: {train_config.muon_lr})")
    print(f"  Grad clip: {train_config.grad_clip}")

    while step < train_config.max_steps:
        epoch += 1
        for input_ids, target_ids in train_loader:
            if step >= train_config.max_steps:
                break

            t0 = time.perf_counter()
            loss = compiled_step(input_ids, target_ids)
            mx.eval(state)
            dt = time.perf_counter() - t0

            step += 1
            loss_val = loss.item()

            if step % train_config.log_interval == 0:
                log_entry: dict[str, float] = {
                    "step": step,
                    "loss": loss_val,
                    "time_ms": dt * 1000,
                    "epoch": epoch,
                }
                # Log H^res metrics for KromCanon models
                if model_config.kromhc.enabled:
                    hres_metrics = extract_hres_metrics(model)
                    log_entry.update(hres_metrics)
                logs.append(log_entry)
                tokens_per_sec = (
                    train_config.batch_size
                    * (model_config.max_seq_len - 1)
                    / dt
                )
                print(
                    f"  step {step:>5d} | loss {loss_val:.4f} | "
                    f"{dt*1000:.0f}ms | {tokens_per_sec:.0f} tok/s"
                )

            if step % train_config.save_interval == 0:
                save_checkpoint(
                    model, optimizer, step, loss_val,
                    model_config, checkpoint_dir / f"step_{step}",
                )
                print(f"  Saved checkpoint at step {step}")

            if (
                eval_loader is not None
                and step % train_config.eval_interval == 0
            ):
                eval_loss = evaluate(model, eval_loader)
                print(f"  eval loss: {eval_loss:.4f}")
                if logs:
                    logs[-1]["eval_loss"] = eval_loss

    # Final checkpoint
    save_checkpoint(
        model, optimizer, step, logs[-1]["loss"] if logs else 0.0,
        model_config, checkpoint_dir / "final",
    )
    print("Training complete. Final checkpoint saved.")

    return logs


def evaluate(model: GPT2, eval_loader: PretrainDataLoader) -> float:
    """Evaluate model on a data loader.

    Args:
        model: The GPT-2 model.
        eval_loader: Evaluation data loader.

    Returns:
        Mean loss over the evaluation set.
    """
    total_loss = 0.0
    n_batches = 0

    for input_ids, target_ids in eval_loader:
        loss = compute_loss(model, input_ids, target_ids)
        total_loss += loss.item()
        n_batches += 1
        if n_batches >= 50:  # Cap evaluation at 50 batches
            break

    return total_loss / max(n_batches, 1)

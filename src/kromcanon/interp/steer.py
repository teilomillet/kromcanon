"""Activation steering: add/subtract directions during forward pass.

Injects a scaled direction vector into activations at specified layers
during the forward pass to steer model behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from kromcanon.model import GPT2


@dataclass
class SteeringConfig:
    """Configuration for activation steering.

    Attributes:
        direction: Unit direction vector, shape (d_model,).
        alpha: Scaling factor. Positive = add direction, negative = subtract.
        layers: Which layers to apply steering at (None = all).
    """

    direction: mx.array
    alpha: float
    layers: list[int] | None = None


def steer_forward(
    model: GPT2,
    input_ids: mx.array,
    steering: SteeringConfig,
) -> mx.array:
    """Forward pass with activation steering.

    At specified layers, adds `alpha * direction` to the hidden state.

    Args:
        model: The GPT-2 model.
        input_ids: Input token IDs, shape (batch, seq_len).
        steering: Steering configuration.

    Returns:
        Logits, shape (batch, seq_len, vocab_size).
    """
    steer_layers = set(
        steering.layers if steering.layers is not None
        else range(model.config.n_layers)
    )
    scaled_dir = steering.alpha * steering.direction  # (d_model,)

    b, t = input_ids.shape
    positions = mx.arange(t)
    x = model.wte(input_ids) + model.wpe(positions)

    residuals: mx.array | None = None
    if model.kromhc_init is not None:
        residuals = model.kromhc_init(x)

    for i, block in enumerate(model.blocks):
        x, residuals = block(x, residuals=residuals)

        if i in steer_layers:
            if residuals is not None:
                # KromCanon: add to all streams
                residuals = residuals + scaled_dir
            else:
                x = x + scaled_dir

    if model.kromhc_reduce is not None and residuals is not None:
        x = model.kromhc_reduce(residuals)

    x = model.ln_f(x)
    return x @ model.wte.weight.T


def steer_generate(
    model: GPT2,
    input_ids: mx.array,
    steering: SteeringConfig,
    max_new_tokens: int = 50,
) -> list[int]:
    """Generate tokens with activation steering applied.

    Args:
        model: The GPT-2 model.
        input_ids: Input token IDs, shape (1, seq_len).
        steering: Steering configuration.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        List of generated token IDs.
    """
    generated: list[int] = []
    current = input_ids

    for _ in range(max_new_tokens):
        logits = steer_forward(model, current, steering)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        generated.append(next_token.item())
        current = mx.concatenate(
            [current, next_token.reshape(1, 1)], axis=1
        )

    return generated


def sweep_alpha(
    model: GPT2,
    input_ids: mx.array,
    direction: mx.array,
    alphas: list[float],
    layers: list[int] | None = None,
) -> dict[float, mx.array]:
    """Sweep over alpha values and collect logits.

    Useful for analyzing the effect of steering strength on output distribution.

    Args:
        model: The GPT-2 model.
        input_ids: Input token IDs, shape (1, seq_len).
        direction: Unit direction vector.
        alphas: List of alpha values to try.
        layers: Which layers to steer.

    Returns:
        Dict mapping alpha → logits at last position, shape (vocab_size,).
    """
    results: dict[float, mx.array] = {}
    for alpha in alphas:
        steering = SteeringConfig(
            direction=direction,
            alpha=alpha,
            layers=layers,
        )
        logits = steer_forward(model, input_ids, steering)
        results[alpha] = logits[0, -1, :]  # Last position logits
    return results

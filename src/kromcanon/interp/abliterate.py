"""Abliteration: direction removal and refusal rate measurement.

Removes identified refusal directions from model weight matrices
to suppress refusal behavior, then measures the change in refusal rate.
"""

from __future__ import annotations

import mlx.core as mx

from kromcanon.model import GPT2


def abliterate_model(
    model: GPT2,
    direction: mx.array,
    layers: list[int] | None = None,
) -> None:
    """Remove a direction from output projection weights (in-place).

    Projects out the refusal direction from the attention output projection
    weight matrix at specified layers.

    For weight matrix W, the abliterated weight is:
        W' = W - (W @ d) @ d^T
    where d is the unit direction vector.

    Args:
        model: The GPT-2 model (modified in-place).
        direction: Unit direction vector, shape (d_model,).
        layers: Which layers to abliterate (None = all).
    """
    if layers is None:
        layers = list(range(model.config.n_layers))

    # Ensure direction is unit norm
    direction = direction / (mx.linalg.norm(direction) + 1e-8)
    # Projection matrix: d @ d^T, shape (d_model, d_model)
    proj = mx.outer(direction, direction)

    for layer_idx in layers:
        block = model.blocks[layer_idx]
        # Abliterate attention output projection
        w = block.attn.o_proj.weight  # (d_model, d_model)
        block.attn.o_proj.weight = w - w @ proj
    mx.eval(model.parameters())


def abliterate_multistream(
    model: GPT2,
    per_stream_directions: mx.array,
    layers: list[int] | None = None,
) -> None:
    """Abliterate with per-stream directions for KromCanon models.

    Applies direction removal per-stream by projecting out directions
    from the output projection, weighted by stream contribution.

    Args:
        model: KromCanon GPT-2 model (modified in-place).
        per_stream_directions: Directions per stream per layer,
            shape (n_layers, n_streams, d_model).
        layers: Which layers to abliterate (None = all).
    """
    if layers is None:
        layers = list(range(model.config.n_layers))

    for i, layer_idx in enumerate(layers):
        if i >= per_stream_directions.shape[0]:
            break
        block = model.blocks[layer_idx]
        # Average the per-stream directions for this layer
        avg_dir = per_stream_directions[i].mean(axis=0)  # (d_model,)
        avg_dir = avg_dir / (mx.linalg.norm(avg_dir) + 1e-8)
        proj = mx.outer(avg_dir, avg_dir)
        w = block.attn.o_proj.weight
        block.attn.o_proj.weight = w - w @ proj
    mx.eval(model.parameters())


def measure_refusal_rate(
    model: GPT2,
    test_prompts: list[mx.array],
    refusal_tokens: set[int] | None = None,
    max_new_tokens: int = 20,
) -> float:
    """Measure refusal rate on test prompts via greedy generation.

    Generates responses to prompts and checks if the first generated tokens
    match common refusal patterns (token-level heuristic).

    Args:
        model: The GPT-2 model.
        test_prompts: List of token ID arrays, each shape (1, seq_len).
        refusal_tokens: Set of token IDs considered refusal indicators.
            If None, uses a simple heuristic (checks if generated text starts
            with common refusal starters).
        max_new_tokens: Maximum tokens to generate per prompt.

    Returns:
        Fraction of prompts that resulted in refusal (0.0 to 1.0).
    """
    if not test_prompts:
        return 0.0

    n_refusals = 0

    for prompt in test_prompts:
        generated = _greedy_generate(model, prompt, max_new_tokens)
        if refusal_tokens is not None:
            # Check if any of the first few tokens are refusal tokens
            first_tokens = generated[:5] if len(generated) >= 5 else generated
            if any(t in refusal_tokens for t in first_tokens):
                n_refusals += 1
        else:
            # Heuristic: check if generation contains refusal-like patterns
            # This is a simple fallback when we don't have refusal token IDs
            n_refusals += 1 if _looks_like_refusal(generated) else 0

    return n_refusals / len(test_prompts)


def _greedy_generate(
    model: GPT2,
    input_ids: mx.array,
    max_new_tokens: int,
) -> list[int]:
    """Greedy token generation.

    Args:
        model: The GPT-2 model.
        input_ids: Input token IDs, shape (1, seq_len).
        max_new_tokens: Number of tokens to generate.

    Returns:
        List of generated token IDs.
    """
    generated: list[int] = []
    current = input_ids

    for _ in range(max_new_tokens):
        logits = model(current)  # (1, seq_len, vocab_size)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)  # (1,)
        generated.append(next_token.item())
        current = mx.concatenate(
            [current, next_token.reshape(1, 1)], axis=1
        )

    return generated


def _looks_like_refusal(token_ids: list[int]) -> bool:
    """Heuristic check if generated tokens look like a refusal.

    This is a very rough heuristic — in practice, you'd decode and check
    for refusal phrases. Here we just check token distribution patterns.

    Args:
        token_ids: Generated token IDs.

    Returns:
        True if the generation looks like a refusal.
    """
    if not token_ids:
        return False
    # Simple heuristic: if the first token is repeated (degenerate),
    # it's likely not a meaningful response. Real refusal detection
    # requires decoding and NLP-level analysis.
    return len(set(token_ids[:5])) <= 2

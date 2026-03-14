"""Direction extraction: mean-diff and SVD for refusal direction identification.

Extracts linear behavioral directions from model activations by contrasting
harmful vs helpful prompts. Supports per-layer extraction and, for KromCanon,
per-stream and joint extraction across multi-stream residuals.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from kromcanon.model import GPT2


@dataclass
class ExtractionResult:
    """Result of direction extraction.

    Attributes:
        directions: Per-layer direction vectors, shape (n_layers, d_model).
        method: Extraction method used ("mean_diff" or "svd").
        layer_norms: Per-layer direction magnitude, shape (n_layers,).
        subspace: Optional top-k SVD subspace, shape (n_layers, k, d_model).
    """

    directions: mx.array
    method: str
    layer_norms: mx.array
    subspace: mx.array | None = None


@dataclass
class MultiStreamExtractionResult:
    """Direction extraction result for KromCanon multi-stream models.

    Attributes:
        per_stream: Per-stream directions, shape (n_layers, n_streams, d_model).
        joint: Joint direction across concatenated streams, shape (n_layers, n_streams * d_model).
        stream_norms: Per-stream direction magnitudes, shape (n_layers, n_streams).
        joint_norms: Joint direction magnitudes, shape (n_layers,).
    """

    per_stream: mx.array
    joint: mx.array
    stream_norms: mx.array
    joint_norms: mx.array


def collect_activations(
    model: GPT2,
    input_ids_list: list[mx.array],
    layers: list[int] | None = None,
) -> dict[int, list[mx.array]]:
    """Collect per-layer activations for a list of inputs.

    Captures the output of each transformer block's residual stream
    (after the block, before the next norm).

    Args:
        model: The GPT-2 model.
        input_ids_list: List of input token ID arrays, each shape (1, seq_len).
        layers: Which layers to collect from (None = all).

    Returns:
        Dict mapping layer_index → list of activation tensors.
        Each activation is the mean over sequence positions, shape (d_model,).
    """
    if layers is None:
        layers = list(range(model.config.n_layers))

    activations: dict[int, list[mx.array]] = {layer: [] for layer in layers}

    for input_ids in input_ids_list:
        # Forward pass collecting intermediate activations
        layer_outputs = _forward_with_activations(model, input_ids, layers)
        for layer_idx, act in layer_outputs.items():
            # Mean over sequence positions → single vector
            activations[layer_idx].append(act.mean(axis=1).squeeze(0))

    return activations


def collect_multistream_activations(
    model: GPT2,
    input_ids_list: list[mx.array],
    layers: list[int] | None = None,
) -> dict[int, list[mx.array]]:
    """Collect per-layer multi-stream activations for KromCanon models.

    For each layer, captures the full multi-stream residual tensor.

    Args:
        model: KromCanon GPT-2 model.
        input_ids_list: List of input token ID arrays.
        layers: Which layers to collect from.

    Returns:
        Dict mapping layer_index → list of activations.
        Each activation shape: (n_streams, d_model) (mean over seq positions).
    """
    if layers is None:
        layers = list(range(model.config.n_layers))

    activations: dict[int, list[mx.array]] = {layer: [] for layer in layers}

    for input_ids in input_ids_list:
        layer_outputs = _forward_with_multistream_activations(
            model, input_ids, layers
        )
        for layer_idx, act in layer_outputs.items():
            # act: (1, n_streams, seq_len, d_model) → mean over seq → (n_streams, d_model)
            activations[layer_idx].append(act.mean(axis=2).squeeze(0))

    return activations


def extract_mean_diff(
    harmful_acts: dict[int, list[mx.array]],
    harmless_acts: dict[int, list[mx.array]],
) -> ExtractionResult:
    """Extract refusal direction via mean difference.

    direction = mean(harmful_activations) - mean(harmless_activations)
    Normalized to unit length per layer.

    Args:
        harmful_acts: Per-layer activations for harmful prompts.
        harmless_acts: Per-layer activations for harmless prompts.

    Returns:
        ExtractionResult with per-layer directions.
    """
    layers = sorted(harmful_acts.keys())
    directions: list[mx.array] = []
    norms: list[float] = []

    for layer in layers:
        harmful_mean = mx.stack(harmful_acts[layer]).mean(axis=0)
        harmless_mean = mx.stack(harmless_acts[layer]).mean(axis=0)
        diff = harmful_mean - harmless_mean
        norm = mx.linalg.norm(diff).item()
        norms.append(norm)
        if norm > 1e-8:
            directions.append(diff / norm)
        else:
            directions.append(diff)

    return ExtractionResult(
        directions=mx.stack(directions),
        method="mean_diff",
        layer_norms=mx.array(norms),
    )


def extract_svd(
    harmful_acts: dict[int, list[mx.array]],
    harmless_acts: dict[int, list[mx.array]],
    top_k: int = 3,
) -> ExtractionResult:
    """Extract refusal subspace via SVD of activation differences.

    Takes the top-k singular vectors of the difference matrix.

    Args:
        harmful_acts: Per-layer activations for harmful prompts.
        harmless_acts: Per-layer activations for harmless prompts.
        top_k: Number of top singular vectors to keep.

    Returns:
        ExtractionResult with per-layer directions (top-1) and subspace (top-k).
    """
    layers = sorted(harmful_acts.keys())
    directions: list[mx.array] = []
    subspaces: list[mx.array] = []
    norms: list[float] = []

    for layer in layers:
        harmful_stack = mx.stack(harmful_acts[layer])  # (n_harmful, d_model)
        harmless_stack = mx.stack(harmless_acts[layer])  # (n_harmless, d_model)

        # Difference matrix: each harmful paired with mean harmless
        harmless_mean = harmless_stack.mean(axis=0)
        diff_matrix = harmful_stack - harmless_mean  # (n_harmful, d_model)

        # SVD (requires float32 — model may produce bfloat16)
        u, s, vt = mx.linalg.svd(diff_matrix.astype(mx.float32), stream=mx.cpu)

        # Top-1 direction (first right singular vector)
        top_dir = vt[0]
        norms.append(s[0].item())
        directions.append(top_dir)

        # Top-k subspace
        k = min(top_k, vt.shape[0])
        subspaces.append(vt[:k])

    return ExtractionResult(
        directions=mx.stack(directions),
        method="svd",
        layer_norms=mx.array(norms),
        subspace=mx.stack(subspaces),
    )


def extract_multistream_directions(
    harmful_acts: dict[int, list[mx.array]],
    harmless_acts: dict[int, list[mx.array]],
    n_streams: int = 4,
) -> MultiStreamExtractionResult:
    """Extract directions from KromCanon multi-stream activations.

    Performs both per-stream and joint extraction.

    Args:
        harmful_acts: Per-layer multi-stream activations for harmful prompts.
            Each activation shape: (n_streams, d_model).
        harmless_acts: Per-layer multi-stream activations for harmless prompts.
        n_streams: Number of residual streams.

    Returns:
        MultiStreamExtractionResult with per-stream and joint directions.
    """
    layers = sorted(harmful_acts.keys())
    per_stream_dirs: list[mx.array] = []
    joint_dirs: list[mx.array] = []
    stream_norms: list[mx.array] = []
    joint_norms: list[float] = []

    for layer in layers:
        harmful_stack = mx.stack(harmful_acts[layer])  # (n, n_streams, d_model)
        harmless_stack = mx.stack(harmless_acts[layer])

        harmful_mean = harmful_stack.mean(axis=0)  # (n_streams, d_model)
        harmless_mean = harmless_stack.mean(axis=0)

        # Per-stream directions
        diff = harmful_mean - harmless_mean  # (n_streams, d_model)
        s_norms = mx.linalg.norm(diff, axis=-1)  # (n_streams,)
        stream_norms.append(s_norms)

        # Normalize per-stream
        normalized = diff / (s_norms[..., None] + 1e-8)
        per_stream_dirs.append(normalized)

        # Joint direction: concatenate streams
        joint_diff = diff.reshape(-1)  # (n_streams * d_model,)
        j_norm = mx.linalg.norm(joint_diff).item()
        joint_norms.append(j_norm)
        joint_dirs.append(joint_diff / (j_norm + 1e-8))

    return MultiStreamExtractionResult(
        per_stream=mx.stack(per_stream_dirs),
        joint=mx.stack(joint_dirs),
        stream_norms=mx.stack(stream_norms),
        joint_norms=mx.array(joint_norms),
    )


# --- Internal helpers ---


def _forward_with_activations(
    model: GPT2,
    input_ids: mx.array,
    layers: list[int],
) -> dict[int, mx.array]:
    """Forward pass that captures intermediate activations.

    Args:
        model: GPT-2 model.
        input_ids: Input token IDs, shape (1, seq_len).
        layers: Which layers to capture.

    Returns:
        Dict mapping layer_index → activation tensor (1, seq_len, d_model).
    """
    b, t = input_ids.shape
    positions = mx.arange(t)
    x = model.wte(input_ids) + model.wpe(positions)
    results: dict[int, mx.array] = {}

    # Handle KromCanon
    residuals: mx.array | None = None
    if model.kromhc_init is not None:
        residuals = model.kromhc_init(x)

    for i, block in enumerate(model.blocks):
        x, residuals = block(x, residuals=residuals)
        if i in layers:
            if residuals is not None:
                # KromCanon: mean over streams
                results[i] = residuals.mean(axis=1)
            else:
                results[i] = x

    return results


def _forward_with_multistream_activations(
    model: GPT2,
    input_ids: mx.array,
    layers: list[int],
) -> dict[int, mx.array]:
    """Forward pass capturing full multi-stream residuals for KromCanon.

    Args:
        model: KromCanon GPT-2 model.
        input_ids: Input token IDs, shape (1, seq_len).
        layers: Which layers to capture.

    Returns:
        Dict mapping layer_index → residuals (1, n_streams, seq_len, d_model).
    """
    b, t = input_ids.shape
    positions = mx.arange(t)
    x = model.wte(input_ids) + model.wpe(positions)

    results: dict[int, mx.array] = {}

    residuals: mx.array | None = None
    if model.kromhc_init is not None:
        residuals = model.kromhc_init(x)

    for i, block in enumerate(model.blocks):
        x, residuals = block(x, residuals=residuals)
        if i in layers:
            if residuals is not None:
                results[i] = residuals
            else:
                # Fallback for non-KromCanon: wrap as single stream
                results[i] = x[:, None, :, :]

    return results

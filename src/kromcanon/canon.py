"""Canon layers: trainable 1-D causal depthwise convolutions for local token mixing.

Based on "Physics of Language Models: Part 4.1" (Allen-Zhu, 2025).
Canon-A: applied to residual stream before attention.
Canon-B: applied to Q, K, V projections after linear projection.
"""

import mlx.core as mx
import mlx.nn as nn

from kromcanon.config import CanonConfig


class DepthwiseCausalConv(nn.Module):
    """Depthwise causal 1-D convolution.

    Each channel has its own independent kernel of size `kernel_size`.
    Left-padded for causal masking (no future information leakage).

    Args:
        d_model: Number of input/output channels.
        kernel_size: Convolution kernel size (looks at current + kernel_size-1 previous tokens).
        bias: Whether to include a bias term.
    """

    def __init__(self, d_model: int, kernel_size: int = 4, bias: bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        # Weight: (channels, kernel_size) — each channel gets its own kernel
        self.weight = mx.random.normal((d_model, kernel_size)) * 0.02
        if bias:
            self.bias = mx.zeros((d_model,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """Apply depthwise causal convolution.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        b, t, d = x.shape
        # Pad left for causal: (batch, kernel_size - 1 + seq_len, d_model)
        x_padded = mx.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0)])
        # Build sliding windows: stack shifted views → (batch, seq_len, d_model, kernel_size)
        windows = mx.stack(
            [x_padded[:, i : i + t, :] for i in range(self.kernel_size)], axis=-1
        )
        # Element-wise multiply with kernel and sum over kernel dimension
        # weight: (d_model, kernel_size) → broadcast over (batch, seq_len)
        out = (windows * self.weight).sum(axis=-1)
        if self.bias is not None:
            out = out + self.bias
        return out


class CanonLayer(nn.Module):
    """Canon layer with optional residual connection.

    Wraps DepthwiseCausalConv with an optional residual bypass.

    Args:
        d_model: Number of channels.
        config: Canon configuration.
    """

    def __init__(self, d_model: int, config: CanonConfig) -> None:
        super().__init__()
        self.conv = DepthwiseCausalConv(
            d_model=d_model,
            kernel_size=config.kernel_size,
            bias=config.bias,
        )
        self.residual = config.residual

    def __call__(self, x: mx.array) -> mx.array:
        """Apply Canon layer.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        out = self.conv(x)
        if self.residual:
            out = out + x
        return out


def apply_canon_a(canon_a: CanonLayer, x: mx.array) -> mx.array:
    """Apply Canon-A (pre-attention local mixing) to normalized hidden states.

    Args:
        canon_a: The Canon-A layer.
        x: Normalized hidden states, shape (batch, seq_len, d_model).

    Returns:
        Locally mixed hidden states, same shape.
    """
    return canon_a(x)


def apply_canon_b(
    canon_b: CanonLayer,
    q: mx.array,
    k: mx.array,
    v: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Apply Canon-B to Q, K, V projections.

    Concatenates Q, K, V along the feature dimension, applies Canon convolution,
    then splits back. This allows local mixing across the combined QKV space.

    Args:
        canon_b: The Canon-B layer (d_model = num_heads * head_dim * 3).
        q: Query tensor, shape (batch, seq_len, n_heads * head_dim).
        k: Key tensor, shape (batch, seq_len, n_heads * head_dim).
        v: Value tensor, shape (batch, seq_len, n_heads * head_dim).

    Returns:
        Tuple of (q, k, v) after Canon-B mixing, same shapes as input.
    """
    d = q.shape[-1]
    qkv = mx.concatenate([q, k, v], axis=-1)  # (batch, seq_len, 3 * d)
    qkv = canon_b(qkv)
    return qkv[..., :d], qkv[..., d : 2 * d], qkv[..., 2 * d :]

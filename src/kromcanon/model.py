"""GPT-2 transformer with pluggable architecture: Vanilla, Canon, KromCanon.

Supports three modes:
- vanilla: Standard GPT-2 decoder-only transformer
- canon: GPT-2 + Canon layers (A: pre-attention, B: on QKV, C: pre-MLP, D: inside MLP)
- kromcanon: GPT-2 + Canon layers + KromHC multi-stream residual connections
"""

import mlx.core as mx
import mlx.nn as nn

from kromcanon.canon import CanonLayer, apply_canon_b
from kromcanon.config import ModelConfig
from kromcanon.kromhc import KromHCInit, KromHCLayer, KromHCReduce


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional Canon-B.

    Args:
        config: Model configuration.
        layer_index: Layer index (for Canon-B creation).
    """

    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.scale = self.head_dim**-0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Canon-B: applied to concatenated Q, K, V
        self.canon_b: CanonLayer | None = None
        if config.canon.enabled and "B" in config.canon.canon_set:
            self.canon_b = CanonLayer(d_model=config.d_model * 3, config=config.canon)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        """Forward pass for causal self-attention.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).
            mask: Optional attention mask, shape (1, 1, seq_len, seq_len).

        Returns:
            Output tensor, shape (batch, seq_len, d_model).
        """
        b, t, d = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Apply Canon-B if enabled
        if self.canon_b is not None:
            q, k, v = apply_canon_b(self.canon_b, q, k, v)

        # Reshape to multi-head: (batch, seq_len, n_heads, head_dim)
        q = q.reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        # Now: (batch, n_heads, seq_len, head_dim)

        # Fused Flash-style attention (Metal kernel)
        sdpa_mask = mask if mask is not None else "causal"
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=sdpa_mask,
        )
        out = out.transpose(0, 2, 1, 3).reshape(b, t, d)  # (batch, seq_len, d_model)

        return self.o_proj(out)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation and optional Canon-D.

    Canon-D is applied after the first linear projection (up-projection),
    before the activation function. This enables local token mixing in the
    expanded MLP space (d_ff dimensions).

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=False)

        # Canon-D: applied after up-projection, before activation
        self.canon_d: CanonLayer | None = None
        if config.canon.enabled and "D" in config.canon.canon_set:
            self.canon_d = CanonLayer(d_model=config.d_ff, config=config.canon)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).

        Returns:
            Output tensor, shape (batch, seq_len, d_model).
        """
        h = self.fc1(x)
        if self.canon_d is not None:
            h = self.canon_d(h)
        return self.fc2(nn.gelu(h))


class TransformerBlock(nn.Module):
    """Single transformer block supporting vanilla/canon/kromcanon modes.

    Vanilla:
        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))

    Canon (with canon_set="ABCD"):
        h = canon_a(norm1(x))     (Canon-A: pre-attention mixing)
        x = x + attn(h)          (attn internally applies Canon-B to QKV)
        h = canon_c(norm2(x))    (Canon-C: pre-MLP mixing)
        x = x + ffn(h)           (ffn internally applies Canon-D before activation)

    KromCanon:
        Uses KromHC residual connections for both attn and ffn branches.
        Each branch goes through width_connection → branch → depth_connection.

    Args:
        config: Model configuration.
        layer_index: Index of this layer.
    """

    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.config = config
        self.layer_index = layer_index

        # Core components
        self.norm1 = nn.RMSNorm(config.d_model, eps=config.norm_eps)
        self.norm2 = nn.RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config, layer_index)
        self.ffn = FeedForward(config)

        # Canon-A: pre-attention local mixing
        self.canon_a: CanonLayer | None = None
        if config.canon.enabled and "A" in config.canon.canon_set:
            self.canon_a = CanonLayer(d_model=config.d_model, config=config.canon)

        # Canon-C: pre-MLP local mixing (after norm2, before FFN)
        self.canon_c: CanonLayer | None = None
        if config.canon.enabled and "C" in config.canon.canon_set:
            self.canon_c = CanonLayer(d_model=config.d_model, config=config.canon)

        # KromHC: one layer per branch (attn and ffn)
        self.kromhc_attn: KromHCLayer | None = None
        self.kromhc_ffn: KromHCLayer | None = None
        if config.kromhc.enabled:
            # Two KromHC layers per block: one for attention, one for FFN
            # Layer indices: even for attn, odd for ffn
            self.kromhc_attn = KromHCLayer(
                d_model=config.d_model,
                n_streams=config.kromhc.n_streams,
                layer_index=layer_index * 2,
                config=config.kromhc,
            )
            self.kromhc_ffn = KromHCLayer(
                d_model=config.d_model,
                n_streams=config.kromhc.n_streams,
                layer_index=layer_index * 2 + 1,
                config=config.kromhc,
            )

    def _attn_branch(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        """Attention branch: norm → optional Canon-A → attention.

        Args:
            x: Input, shape (batch, seq_len, d_model).
            mask: Optional attention mask.

        Returns:
            Attention output, shape (batch, seq_len, d_model).
        """
        h = self.norm1(x)
        if self.canon_a is not None:
            h = self.canon_a(h)
        return self.attn(h, mask=mask)

    def _ffn_branch(self, x: mx.array) -> mx.array:
        """FFN branch: norm → optional Canon-C → feed-forward.

        Args:
            x: Input, shape (batch, seq_len, d_model).

        Returns:
            FFN output, shape (batch, seq_len, d_model).
        """
        h = self.norm2(x)
        if self.canon_c is not None:
            h = self.canon_c(h)
        return self.ffn(h)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        residuals: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        """Forward pass for transformer block.

        For vanilla/canon: uses standard residual connections.
        For kromcanon: uses KromHC multi-stream residuals.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model). Used for vanilla/canon.
            mask: Optional causal attention mask.
            residuals: Multi-stream residuals for KromCanon mode,
                      shape (batch, n_streams, seq_len, d_model).

        Returns:
            Tuple of:
                - x: Updated hidden state (batch, seq_len, d_model). For vanilla/canon.
                - residuals: Updated multi-stream residuals. None for vanilla/canon.
        """
        if self.kromhc_attn is not None and residuals is not None:
            # KromCanon mode: use KromHC for both branches
            residuals = self.kromhc_attn(
                residuals, branch_fn=lambda h: self._attn_branch(h, mask=mask)
            )
            residuals = self.kromhc_ffn(residuals, branch_fn=self._ffn_branch)
            return x, residuals  # x is not used in KromCanon mode
        else:
            # Vanilla / Canon mode: standard residual
            x = x + self._attn_branch(x, mask=mask)
            x = x + self._ffn_branch(x)
            return x, None


class GPT2(nn.Module):
    """GPT-2 model with pluggable architecture.

    Supports vanilla, canon, and kromcanon modes via ModelConfig.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer blocks
        self.blocks = [TransformerBlock(config, i) for i in range(config.n_layers)]

        # Final layer norm
        self.ln_f = nn.RMSNorm(config.d_model, eps=config.norm_eps)

        # LM head: weight-tied with wte (standard GPT-2)
        # No separate nn.Linear — forward uses x @ wte.weight.T directly.

        # KromHC init/reduce
        self.kromhc_init: KromHCInit | None = None
        self.kromhc_reduce: KromHCReduce | None = None
        if config.kromhc.enabled:
            self.kromhc_init = KromHCInit(config.kromhc.n_streams)
            self.kromhc_reduce = KromHCReduce(config.kromhc.n_streams)

    def __call__(self, input_ids: mx.array) -> mx.array:
        """Forward pass: input token IDs → logits.

        Args:
            input_ids: Token IDs, shape (batch, seq_len).

        Returns:
            Logits, shape (batch, seq_len, vocab_size).
        """
        b, t = input_ids.shape

        # Token + position embeddings
        positions = mx.arange(t)
        x = self.wte(input_ids) + self.wpe(positions)

        # KromCanon: initialize multi-stream residuals
        residuals: mx.array | None = None
        if self.kromhc_init is not None:
            residuals = self.kromhc_init(x)

        # Transformer blocks (causal masking handled by SDPA)
        for block in self.blocks:
            x, residuals = block(x, residuals=residuals)

        # KromCanon: reduce multi-stream to single stream
        if self.kromhc_reduce is not None and residuals is not None:
            x = self.kromhc_reduce(residuals)

        # Final norm + LM head (weight-tied with wte)
        x = self.ln_f(x)
        return x @ self.wte.weight.T

    def count_parameters(self) -> int:
        """Count total trainable parameters.

        Returns:
            Total number of parameters.
        """
        total = 0
        for _k, v in self.parameters().items():
            if isinstance(v, mx.array):
                total += v.size
            elif isinstance(v, dict):
                total += sum(p.size for p in v.values() if isinstance(p, mx.array))
        return total


def make_model(config: ModelConfig) -> GPT2:
    """Create a GPT-2 model from config, cast to bfloat16.

    Uses bfloat16 for all parameters — halves memory bandwidth
    on Apple Silicon for ~1.5-2x throughput. Loss computation
    is done in float32 for numerical stability (see compute_loss).

    Args:
        config: Model configuration.

    Returns:
        Initialized GPT-2 model in bfloat16.
    """
    model = GPT2(config)
    model.set_dtype(mx.bfloat16)
    return model

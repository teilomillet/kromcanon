"""Tests for GPT-2 model with pluggable architecture."""

import mlx.core as mx

from kromcanon.config import ModelConfig
from kromcanon.model import GPT2, CausalSelfAttention, FeedForward, TransformerBlock


def _small_config(arch: str = "vanilla") -> ModelConfig:
    """Create a small config for testing."""
    return ModelConfig(
        arch=arch,
        vocab_size=256,
        n_layers=2,
        n_heads=4,
        d_model=64,
        d_ff=256,
        max_seq_len=32,
    )


class TestCausalSelfAttention:
    """Tests for CausalSelfAttention."""

    def test_output_shape(self) -> None:
        """Output shape matches input shape."""
        config = _small_config()
        attn = CausalSelfAttention(config, layer_index=0)
        x = mx.random.normal((2, 16, 64))
        out = attn(x)
        assert out.shape == (2, 16, 64)

    def test_with_canon_b(self) -> None:
        """Canon-B is applied when config enables it."""
        config = _small_config("canon")
        attn = CausalSelfAttention(config, layer_index=0)
        assert attn.canon_b is not None
        x = mx.random.normal((2, 16, 64))
        out = attn(x)
        assert out.shape == (2, 16, 64)

    def test_without_canon_b(self) -> None:
        """No Canon-B in vanilla mode."""
        config = _small_config("vanilla")
        attn = CausalSelfAttention(config, layer_index=0)
        assert attn.canon_b is None


class TestFeedForward:
    """Tests for FeedForward."""

    def test_output_shape(self) -> None:
        """Output shape matches input shape."""
        config = _small_config()
        ffn = FeedForward(config)
        x = mx.random.normal((2, 16, 64))
        out = ffn(x)
        assert out.shape == (2, 16, 64)


class TestTransformerBlock:
    """Tests for TransformerBlock."""

    def test_vanilla_forward(self) -> None:
        """Vanilla block forward pass."""
        config = _small_config("vanilla")
        block = TransformerBlock(config, layer_index=0)
        x = mx.random.normal((2, 16, 64))
        out, residuals = block(x)
        assert out.shape == (2, 16, 64)
        assert residuals is None

    def test_canon_forward(self) -> None:
        """Canon block forward pass."""
        config = _small_config("canon")
        block = TransformerBlock(config, layer_index=0)
        assert block.canon_a is not None
        x = mx.random.normal((2, 16, 64))
        out, residuals = block(x)
        assert out.shape == (2, 16, 64)
        assert residuals is None

    def test_kromcanon_forward(self) -> None:
        """KromCanon block forward pass with multi-stream residuals."""
        config = _small_config("kromcanon")
        block = TransformerBlock(config, layer_index=0)
        assert block.kromhc_attn is not None
        assert block.kromhc_ffn is not None
        x = mx.random.normal((2, 16, 64))
        residuals = mx.random.normal((2, 4, 16, 64))
        _, residuals_out = block(x, residuals=residuals)
        assert residuals_out is not None
        assert residuals_out.shape == (2, 4, 16, 64)


class TestGPT2:
    """Tests for full GPT-2 model."""

    def test_vanilla_forward(self) -> None:
        """Vanilla GPT-2 forward pass produces correct logit shape."""
        config = _small_config("vanilla")
        model = GPT2(config)
        input_ids = mx.random.randint(0, 256, (2, 16))
        logits = model(input_ids)
        assert logits.shape == (2, 16, 256)

    def test_canon_forward(self) -> None:
        """Canon GPT-2 forward pass produces correct logit shape."""
        config = _small_config("canon")
        model = GPT2(config)
        input_ids = mx.random.randint(0, 256, (2, 16))
        logits = model(input_ids)
        assert logits.shape == (2, 16, 256)

    def test_kromcanon_forward(self) -> None:
        """KromCanon GPT-2 forward pass produces correct logit shape."""
        config = _small_config("kromcanon")
        model = GPT2(config)
        input_ids = mx.random.randint(0, 256, (2, 16))
        logits = model(input_ids)
        assert logits.shape == (2, 16, 256)

    def test_canon_has_more_params_than_vanilla(self) -> None:
        """Canon should have slightly more parameters than vanilla (~0.5% overhead)."""
        vanilla = GPT2(_small_config("vanilla"))
        canon = GPT2(_small_config("canon"))
        p_vanilla = _count_params(vanilla)
        p_canon = _count_params(canon)
        assert p_canon > p_vanilla, (
            f"Canon ({p_canon}) should have more params than vanilla ({p_vanilla})"
        )
        overhead = (p_canon - p_vanilla) / p_vanilla * 100
        assert overhead < 5.0, f"Canon overhead too high: {overhead:.1f}%"

    def test_kromcanon_has_more_params_than_canon(self) -> None:
        """KromCanon should have more parameters than Canon (KromHC adds params)."""
        canon = GPT2(_small_config("canon"))
        kromcanon = GPT2(_small_config("kromcanon"))
        p_canon = _count_params(canon)
        p_kromcanon = _count_params(kromcanon)
        assert p_kromcanon > p_canon, (
            f"KromCanon ({p_kromcanon}) should have more params than canon ({p_canon})"
        )

    def test_single_token(self) -> None:
        """Model works with single token input."""
        config = _small_config("vanilla")
        model = GPT2(config)
        input_ids = mx.array([[42]])
        logits = model(input_ids)
        assert logits.shape == (1, 1, 256)

    def test_max_seq_len(self) -> None:
        """Model works at max sequence length."""
        config = _small_config("vanilla")
        model = GPT2(config)
        input_ids = mx.random.randint(0, 256, (1, 32))
        logits = model(input_ids)
        assert logits.shape == (1, 32, 256)


def _count_params(model: GPT2) -> int:
    """Count total parameters in a model by traversing leaf arrays."""
    import mlx.utils
    total = 0
    leaves = mlx.utils.tree_flatten(model.parameters())
    for _, v in leaves:
        total += v.size
    return total

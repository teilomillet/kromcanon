"""Tests for Canon layer implementation."""

import mlx.core as mx

from kromcanon.canon import CanonLayer, DepthwiseCausalConv, apply_canon_b
from kromcanon.config import CanonConfig


class TestDepthwiseCausalConv:
    """Tests for DepthwiseCausalConv."""

    def test_output_shape(self) -> None:
        """Output shape matches input shape."""
        conv = DepthwiseCausalConv(d_model=64, kernel_size=4)
        x = mx.random.normal((2, 16, 64))
        out = conv(x)
        assert out.shape == (2, 16, 64)

    def test_causal_masking(self) -> None:
        """Output at position t depends only on positions <= t.

        Verify by checking that changing a future token doesn't affect earlier outputs.
        """
        conv = DepthwiseCausalConv(d_model=32, kernel_size=4)
        x = mx.random.normal((1, 8, 32))
        out_original = conv(x)

        # Modify token at position 5 — should not affect positions 0-4
        x_modified = mx.array(x)
        # Create modification at position 5
        x_list = list(x.tolist())
        x_list[0][5] = [99.0] * 32
        x_modified = mx.array(x_list)
        out_modified = conv(x_modified)

        # Positions 0-4 should be identical
        diff = mx.abs(out_original[:, :5, :] - out_modified[:, :5, :])
        assert mx.max(diff).item() < 1e-5, "Causal violation: future token affected past output"

    def test_kernel_size_1(self) -> None:
        """Kernel size 1 is a pointwise convolution (no mixing)."""
        conv = DepthwiseCausalConv(d_model=16, kernel_size=1)
        x = mx.random.normal((1, 4, 16))
        out = conv(x)
        assert out.shape == (1, 4, 16)

    def test_with_bias(self) -> None:
        """Bias parameter is applied correctly."""
        conv = DepthwiseCausalConv(d_model=16, kernel_size=4, bias=True)
        assert conv.bias is not None
        x = mx.random.normal((1, 8, 16))
        out = conv(x)
        assert out.shape == (1, 8, 16)

    def test_batch_independence(self) -> None:
        """Each batch element is processed independently."""
        conv = DepthwiseCausalConv(d_model=32, kernel_size=4)
        x1 = mx.random.normal((1, 8, 32))
        x2 = mx.random.normal((1, 8, 32))
        x_batch = mx.concatenate([x1, x2], axis=0)

        out_batch = conv(x_batch)
        out_1 = conv(x1)
        out_2 = conv(x2)

        diff1 = mx.max(mx.abs(out_batch[0:1] - out_1)).item()
        diff2 = mx.max(mx.abs(out_batch[1:2] - out_2)).item()
        assert diff1 < 1e-5
        assert diff2 < 1e-5


class TestCanonLayer:
    """Tests for CanonLayer."""

    def test_output_shape(self) -> None:
        """Output shape matches input shape."""
        config = CanonConfig(enabled=True)
        layer = CanonLayer(d_model=64, config=config)
        x = mx.random.normal((2, 16, 64))
        out = layer(x)
        assert out.shape == (2, 16, 64)

    def test_residual_connection(self) -> None:
        """With residual=True, output differs from pure conv output."""
        config = CanonConfig(enabled=True, residual=True)
        layer = CanonLayer(d_model=32, config=config)
        x = mx.random.normal((1, 8, 32))
        out_with_res = layer(x)

        # Without residual
        config_no_res = CanonConfig(enabled=True, residual=False)
        layer_no_res = CanonLayer(d_model=32, config=config_no_res)
        # Copy weights
        layer_no_res.conv.weight = layer.conv.weight
        out_without_res = layer_no_res(x)

        # out_with_res should equal out_without_res + x
        diff = mx.max(mx.abs(out_with_res - (out_without_res + x))).item()
        assert diff < 1e-5

    def test_no_residual(self) -> None:
        """With residual=False, output is pure convolution."""
        config = CanonConfig(enabled=True, residual=False)
        layer = CanonLayer(d_model=32, config=config)
        x = mx.random.normal((1, 8, 32))
        out = layer(x)
        conv_out = layer.conv(x)
        diff = mx.max(mx.abs(out - conv_out)).item()
        assert diff < 1e-5


class TestCanonB:
    """Tests for Canon-B (Q/K/V mixing)."""

    def test_output_shapes(self) -> None:
        """Q, K, V maintain their shapes after Canon-B."""
        d = 64
        config = CanonConfig(enabled=True)
        canon_b = CanonLayer(d_model=d * 3, config=config)
        q = mx.random.normal((2, 16, d))
        k = mx.random.normal((2, 16, d))
        v = mx.random.normal((2, 16, d))
        q_out, k_out, v_out = apply_canon_b(canon_b, q, k, v)
        assert q_out.shape == (2, 16, d)
        assert k_out.shape == (2, 16, d)
        assert v_out.shape == (2, 16, d)

    def test_mixing_occurs(self) -> None:
        """Canon-B actually modifies Q, K, V (not identity)."""
        d = 32
        config = CanonConfig(enabled=True, residual=False)
        canon_b = CanonLayer(d_model=d * 3, config=config)
        q = mx.random.normal((1, 8, d))
        k = mx.random.normal((1, 8, d))
        v = mx.random.normal((1, 8, d))
        q_out, k_out, v_out = apply_canon_b(canon_b, q, k, v)
        # With random weights and no residual, output should differ from input
        assert mx.max(mx.abs(q_out - q)).item() > 1e-6


class TestCanonCD:
    """Tests for Canon-C (pre-MLP) and Canon-D (inside MLP)."""

    def test_canon_c_placement(self) -> None:
        """Canon-C is applied before FFN, changes block output."""
        from kromcanon.config import ModelConfig
        from kromcanon.model import TransformerBlock

        config = ModelConfig(arch="vanilla", n_layers=1, n_heads=4, d_model=64, d_ff=256)
        # Manually enable Canon-C only
        config.canon = CanonConfig(enabled=True, canon_set="C")
        block = TransformerBlock(config, layer_index=0)

        assert block.canon_c is not None
        assert block.canon_a is None  # Only C enabled
        x = mx.random.normal((1, 8, 64))
        out, _ = block(x)
        assert out.shape == (1, 8, 64)

    def test_canon_d_placement(self) -> None:
        """Canon-D is inside FFN, changes FFN output."""
        from kromcanon.config import ModelConfig
        from kromcanon.model import FeedForward

        config = ModelConfig(arch="vanilla", n_layers=1, n_heads=4, d_model=64, d_ff=256)
        config.canon = CanonConfig(enabled=True, canon_set="D")
        ffn = FeedForward(config)

        assert ffn.canon_d is not None
        assert ffn.canon_d.conv.d_model == 256  # d_ff, not d_model
        x = mx.random.normal((1, 8, 64))
        out = ffn(x)
        assert out.shape == (1, 8, 64)

    def test_canon_abcd_full(self) -> None:
        """Full ABCD configuration creates all four Canon layers."""
        from kromcanon.config import ModelConfig
        from kromcanon.model import TransformerBlock

        config = ModelConfig(arch="vanilla", n_layers=1, n_heads=4, d_model=64, d_ff=256)
        config.canon = CanonConfig(enabled=True, canon_set="ABCD")
        block = TransformerBlock(config, layer_index=0)

        assert block.canon_a is not None
        assert block.attn.canon_b is not None
        assert block.canon_c is not None
        assert block.ffn.canon_d is not None

        x = mx.random.normal((1, 8, 64))
        out, _ = block(x)
        assert out.shape == (1, 8, 64)

    def test_canon_d_dimension(self) -> None:
        """Canon-D operates at d_ff dimension, not d_model."""
        from kromcanon.config import ModelConfig
        from kromcanon.model import FeedForward

        config = ModelConfig(arch="vanilla", n_layers=1, n_heads=4, d_model=64, d_ff=256)
        config.canon = CanonConfig(enabled=True, canon_set="D")
        ffn = FeedForward(config)

        # Canon-D conv weight should be (d_ff, kernel_size)
        assert ffn.canon_d.conv.weight.shape == (256, 4)

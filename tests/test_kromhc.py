"""Tests for KromHC residual connection implementation."""

import mlx.core as mx

from kromcanon.config import KromHCConfig
from kromcanon.kromhc import (
    KromHCInit,
    KromHCLayer,
    KromHCReduce,
    _build_2x2_factor,
    _build_doubly_stochastic_factor,
    _kronecker_product,
)


class TestDoublyStochastic:
    """Tests for doubly stochastic matrix construction."""

    def test_2x2_identity(self) -> None:
        """Weights [1, 0] should produce identity matrix."""
        weights = mx.array([1.0, 0.0])
        factor = _build_2x2_factor(weights)
        expected = mx.array([[1.0, 0.0], [0.0, 1.0]])
        diff = mx.max(mx.abs(factor - expected)).item()
        assert diff < 1e-6

    def test_2x2_swap(self) -> None:
        """Weights [0, 1] should produce swap matrix."""
        weights = mx.array([0.0, 1.0])
        factor = _build_2x2_factor(weights)
        expected = mx.array([[0.0, 1.0], [1.0, 0.0]])
        diff = mx.max(mx.abs(factor - expected)).item()
        assert diff < 1e-6

    def test_2x2_doubly_stochastic_property(self) -> None:
        """Any convex combination should produce a doubly stochastic matrix."""
        weights = mx.softmax(mx.random.normal((2,)), axis=-1)
        factor = _build_2x2_factor(weights)
        # Row sums should be 1
        row_sums = factor.sum(axis=-1)
        assert mx.max(mx.abs(row_sums - 1.0)).item() < 1e-5
        # Column sums should be 1
        col_sums = factor.sum(axis=-2)
        assert mx.max(mx.abs(col_sums - 1.0)).item() < 1e-5
        # All entries should be non-negative
        assert mx.min(factor).item() >= -1e-7

    def test_batched_2x2(self) -> None:
        """2x2 factor works with batch dimensions."""
        weights = mx.softmax(mx.random.normal((4, 8, 2)), axis=-1)
        factor = _build_2x2_factor(weights)
        assert factor.shape == (4, 8, 2, 2)
        # Check doubly stochastic for all elements
        row_sums = factor.sum(axis=-1)
        col_sums = factor.sum(axis=-2)
        assert mx.max(mx.abs(row_sums - 1.0)).item() < 1e-5
        assert mx.max(mx.abs(col_sums - 1.0)).item() < 1e-5

    def test_general_factor_3x3(self) -> None:
        """General factor for 3x3 (6 permutations) is doubly stochastic."""
        weights = mx.softmax(mx.random.normal((6,)), axis=-1)
        factor = _build_doubly_stochastic_factor(weights, factor_size=3)
        assert factor.shape == (3, 3)
        row_sums = factor.sum(axis=-1)
        col_sums = factor.sum(axis=-2)
        assert mx.max(mx.abs(row_sums - 1.0)).item() < 1e-5
        assert mx.max(mx.abs(col_sums - 1.0)).item() < 1e-5


class TestKroneckerProduct:
    """Tests for Kronecker product."""

    def test_identity_kronecker_identity(self) -> None:
        """I_2 ⊗ I_2 = I_4."""
        i2 = mx.eye(2)
        result = _kronecker_product(i2, i2)
        expected = mx.eye(4)
        diff = mx.max(mx.abs(result - expected)).item()
        assert diff < 1e-6

    def test_output_shape(self) -> None:
        """Kronecker product of (m,m) and (n,n) gives (m*n, m*n)."""
        a = mx.random.normal((2, 2))
        b = mx.random.normal((3, 3))
        result = _kronecker_product(a, b)
        assert result.shape == (6, 6)

    def test_batched_kronecker(self) -> None:
        """Kronecker product works with batch dimensions."""
        a = mx.random.normal((4, 8, 2, 2))
        b = mx.random.normal((4, 8, 2, 2))
        result = _kronecker_product(a, b)
        assert result.shape == (4, 8, 4, 4)

    def test_doubly_stochastic_closure(self) -> None:
        """Kronecker product of doubly stochastic matrices is doubly stochastic.

        This is Theorem 4.2 from the KromHC paper.
        """
        w1 = mx.softmax(mx.random.normal((2,)), axis=-1)
        w2 = mx.softmax(mx.random.normal((2,)), axis=-1)
        u1 = _build_2x2_factor(w1)
        u2 = _build_2x2_factor(w2)
        result = _kronecker_product(u1, u2)
        assert result.shape == (4, 4)
        row_sums = result.sum(axis=-1)
        col_sums = result.sum(axis=-2)
        assert mx.max(mx.abs(row_sums - 1.0)).item() < 1e-5
        assert mx.max(mx.abs(col_sums - 1.0)).item() < 1e-5
        assert mx.min(result).item() >= -1e-7


class TestKromHCLayer:
    """Tests for the full KromHCLayer."""

    def _make_config(self, dynamic: bool = True) -> KromHCConfig:
        return KromHCConfig(enabled=True, n_streams=4, dynamic=dynamic)

    def test_width_connection_shapes(self) -> None:
        """Width connection produces correct output shapes."""
        config = self._make_config()
        layer = KromHCLayer(d_model=64, n_streams=4, layer_index=0, config=config)
        residuals = mx.random.normal((2, 4, 16, 64))
        branch_input, residuals_mixed = layer.width_connection(residuals)
        assert branch_input.shape == (2, 16, 64)
        assert residuals_mixed.shape == (2, 4, 16, 64)

    def test_depth_connection_shapes(self) -> None:
        """Depth connection produces correct output shapes."""
        config = self._make_config()
        layer = KromHCLayer(d_model=64, n_streams=4, layer_index=0, config=config)
        branch_output = mx.random.normal((2, 16, 64))
        residuals = mx.random.normal((2, 4, 16, 64))
        result = layer.depth_connection(branch_output, residuals)
        assert result.shape == (2, 4, 16, 64)

    def test_full_forward_shapes(self) -> None:
        """Full forward pass with identity branch preserves shapes."""
        config = self._make_config()
        layer = KromHCLayer(d_model=64, n_streams=4, layer_index=0, config=config)
        residuals = mx.random.normal((2, 4, 16, 64))
        result = layer(residuals, branch_fn=lambda x: x)
        assert result.shape == (2, 4, 16, 64)

    def test_init_near_identity(self) -> None:
        """At initialization, H^res should be approximately identity.

        With b^res = [0, -8] → softmax ≈ [1, 0] → U ≈ I → H^res ≈ I_4.
        """
        config = self._make_config(dynamic=False)
        layer = KromHCLayer(d_model=32, n_streams=4, layer_index=0, config=config)
        residuals = mx.random.normal((1, 4, 8, 32))
        _, residuals_mixed = layer.width_connection(residuals)
        # H^res ≈ I → residuals_mixed ≈ residuals
        diff = mx.max(mx.abs(residuals_mixed - residuals)).item()
        assert diff < 0.01, f"H^res not near identity at init: max diff = {diff}"

    def test_static_mode(self) -> None:
        """Static mode (dynamic=False) works correctly."""
        config = self._make_config(dynamic=False)
        layer = KromHCLayer(d_model=32, n_streams=4, layer_index=0, config=config)
        residuals = mx.random.normal((1, 4, 8, 32))
        result = layer(residuals, branch_fn=lambda x: x * 0.1)
        assert result.shape == (1, 4, 8, 32)

    def test_different_layer_indices(self) -> None:
        """Different layer indices produce different stream selection patterns."""
        config = self._make_config(dynamic=False)
        layer0 = KromHCLayer(d_model=32, n_streams=4, layer_index=0, config=config)
        layer1 = KromHCLayer(d_model=32, n_streams=4, layer_index=1, config=config)
        # b_pre should have the 1.0 at different positions
        assert layer0.b_pre[0].item() > 0, "Layer 0 should select stream 0"
        assert layer1.b_pre[1].item() > 0, "Layer 1 should select stream 1"


class TestKromHCInitReduce:
    """Tests for stream initialization and reduction."""

    def test_init_shape(self) -> None:
        """KromHCInit expands single stream to multi-stream."""
        init = KromHCInit(n_streams=4)
        x = mx.random.normal((2, 16, 64))
        result = init(x)
        assert result.shape == (2, 4, 16, 64)

    def test_init_content(self) -> None:
        """KromHCInit replicates input across all streams."""
        init = KromHCInit(n_streams=4)
        x = mx.random.normal((1, 8, 32))
        result = init(x)
        for s in range(4):
            diff = mx.max(mx.abs(result[:, s, :, :] - x)).item()
            assert diff < 1e-6, f"Stream {s} doesn't match input"

    def test_reduce_shape(self) -> None:
        """KromHCReduce averages multi-stream to single stream."""
        reduce_mod = KromHCReduce(n_streams=4)
        residuals = mx.random.normal((2, 4, 16, 64))
        result = reduce_mod(residuals)
        assert result.shape == (2, 16, 64)

    def test_reduce_is_mean(self) -> None:
        """KromHCReduce computes mean across streams."""
        reduce_mod = KromHCReduce(n_streams=4)
        residuals = mx.random.normal((1, 4, 8, 32))
        result = reduce_mod(residuals)
        expected = residuals.mean(axis=1)
        diff = mx.max(mx.abs(result - expected)).item()
        assert diff < 1e-6

    def test_init_then_reduce_preserves(self) -> None:
        """Init → Reduce should return the original input."""
        init = KromHCInit(n_streams=4)
        reduce_mod = KromHCReduce(n_streams=4)
        x = mx.random.normal((2, 16, 64))
        result = reduce_mod(init(x))
        diff = mx.max(mx.abs(result - x)).item()
        assert diff < 1e-6

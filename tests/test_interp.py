"""Tests for interpretability tooling."""

import mlx.core as mx

from kromcanon.config import ModelConfig
from kromcanon.interp.abliterate import abliterate_model
from kromcanon.interp.compare import (
    analyze_stream_distribution,
    compare_directions,
)
from kromcanon.interp.extract import (
    ExtractionResult,
    MultiStreamExtractionResult,
    collect_activations,
    extract_mean_diff,
    extract_multistream_directions,
    extract_svd,
)
from kromcanon.interp.steer import SteeringConfig, steer_forward, sweep_alpha
from kromcanon.model import GPT2


def _small_config(arch: str = "vanilla") -> ModelConfig:
    return ModelConfig(
        arch=arch, vocab_size=256, n_layers=2,
        n_heads=4, d_model=64, d_ff=256, max_seq_len=32,
    )


class TestCollectActivations:
    """Tests for activation collection."""

    def test_collect_vanilla(self) -> None:
        """Collect activations from vanilla model."""
        model = GPT2(_small_config())
        inputs = [mx.random.randint(0, 256, (1, 8)) for _ in range(4)]
        acts = collect_activations(model, inputs)
        assert len(acts) == 2  # 2 layers
        assert len(acts[0]) == 4  # 4 inputs
        assert acts[0][0].shape == (64,)  # d_model

    def test_collect_specific_layers(self) -> None:
        """Collect from specific layers only."""
        model = GPT2(_small_config())
        inputs = [mx.random.randint(0, 256, (1, 8)) for _ in range(2)]
        acts = collect_activations(model, inputs, layers=[0])
        assert 0 in acts
        assert 1 not in acts


class TestExtractMeanDiff:
    """Tests for mean-diff extraction."""

    def test_directions_shape(self) -> None:
        """Extracted directions have correct shape."""
        model = GPT2(_small_config())
        harmful = [mx.random.randint(0, 256, (1, 8)) for _ in range(5)]
        harmless = [mx.random.randint(0, 256, (1, 8)) for _ in range(5)]
        harmful_acts = collect_activations(model, harmful)
        harmless_acts = collect_activations(model, harmless)
        result = extract_mean_diff(harmful_acts, harmless_acts)
        assert result.directions.shape == (2, 64)  # (n_layers, d_model)
        assert result.method == "mean_diff"

    def test_directions_unit_norm(self) -> None:
        """Extracted directions are approximately unit norm."""
        model = GPT2(_small_config())
        harmful = [mx.random.randint(0, 256, (1, 8)) for _ in range(10)]
        harmless = [mx.random.randint(0, 256, (1, 8)) for _ in range(10)]
        harmful_acts = collect_activations(model, harmful)
        harmless_acts = collect_activations(model, harmless)
        result = extract_mean_diff(harmful_acts, harmless_acts)
        for i in range(result.directions.shape[0]):
            norm = mx.linalg.norm(result.directions[i]).item()
            assert abs(norm - 1.0) < 0.01, f"Layer {i} norm = {norm}"


class TestExtractSVD:
    """Tests for SVD extraction."""

    def test_svd_directions(self) -> None:
        """SVD extraction produces directions and subspace."""
        model = GPT2(_small_config())
        harmful = [mx.random.randint(0, 256, (1, 8)) for _ in range(10)]
        harmless = [mx.random.randint(0, 256, (1, 8)) for _ in range(10)]
        harmful_acts = collect_activations(model, harmful)
        harmless_acts = collect_activations(model, harmless)
        result = extract_svd(harmful_acts, harmless_acts, top_k=3)
        assert result.directions.shape == (2, 64)
        assert result.subspace is not None
        assert result.subspace.shape[1] <= 3  # top_k


class TestMultistreamExtraction:
    """Tests for KromCanon multi-stream extraction."""

    def test_multistream_shapes(self) -> None:
        """Multi-stream extraction produces correct shapes."""
        harmful_acts = {
            0: [mx.random.normal((4, 64)) for _ in range(5)],
            1: [mx.random.normal((4, 64)) for _ in range(5)],
        }
        harmless_acts = {
            0: [mx.random.normal((4, 64)) for _ in range(5)],
            1: [mx.random.normal((4, 64)) for _ in range(5)],
        }
        result = extract_multistream_directions(harmful_acts, harmless_acts)
        assert result.per_stream.shape == (2, 4, 64)
        assert result.joint.shape == (2, 256)  # 4 * 64
        assert result.stream_norms.shape == (2, 4)


class TestAbliterate:
    """Tests for abliteration."""

    def test_abliterate_modifies_weights(self) -> None:
        """Abliteration changes model weights."""
        model = GPT2(_small_config())
        original_w = mx.array(model.blocks[0].attn.o_proj.weight)
        direction = mx.random.normal((64,))
        direction = direction / mx.linalg.norm(direction)
        abliterate_model(model, direction, layers=[0])
        modified_w = model.blocks[0].attn.o_proj.weight
        diff = mx.max(mx.abs(original_w - modified_w)).item()
        assert diff > 1e-4, "Abliteration should modify weights"

    def test_abliterate_removes_direction(self) -> None:
        """After abliteration, projection onto direction is near zero."""
        model = GPT2(_small_config())
        direction = mx.random.normal((64,))
        direction = direction / mx.linalg.norm(direction)
        abliterate_model(model, direction, layers=[0])
        w = model.blocks[0].attn.o_proj.weight
        projection = w @ direction
        proj_norm = mx.linalg.norm(projection).item()
        assert proj_norm < 0.01, f"Direction not removed: proj norm = {proj_norm}"


class TestSteering:
    """Tests for activation steering."""

    def test_steer_produces_logits(self) -> None:
        """Steering forward pass produces valid logits."""
        model = GPT2(_small_config())
        input_ids = mx.random.randint(0, 256, (1, 8))
        direction = mx.random.normal((64,))
        direction = direction / mx.linalg.norm(direction)
        steering = SteeringConfig(direction=direction, alpha=1.0)
        logits = steer_forward(model, input_ids, steering)
        assert logits.shape == (1, 8, 256)

    def test_alpha_zero_matches_normal(self) -> None:
        """Steering with alpha=0 should match normal forward pass."""
        model = GPT2(_small_config())
        input_ids = mx.random.randint(0, 256, (1, 8))
        direction = mx.random.normal((64,))
        steering = SteeringConfig(direction=direction, alpha=0.0)
        logits_steered = steer_forward(model, input_ids, steering)
        logits_normal = model(input_ids)
        diff = mx.max(mx.abs(logits_steered - logits_normal)).item()
        assert diff < 1e-4, f"alpha=0 should match normal: diff={diff}"

    def test_sweep_alpha(self) -> None:
        """Alpha sweep returns results for each alpha."""
        model = GPT2(_small_config())
        input_ids = mx.random.randint(0, 256, (1, 8))
        direction = mx.random.normal((64,))
        results = sweep_alpha(model, input_ids, direction, [-1.0, 0.0, 1.0])
        assert len(results) == 3
        for alpha in [-1.0, 0.0, 1.0]:
            assert results[alpha].shape == (256,)


class TestCompare:
    """Tests for cross-architecture comparison."""

    def test_compare_directions(self) -> None:
        """Comparison produces valid cosine similarities."""
        result_a = ExtractionResult(
            directions=mx.random.normal((2, 64)),
            method="test",
            layer_norms=mx.array([1.0, 0.5]),
        )
        result_b = ExtractionResult(
            directions=mx.random.normal((2, 64)),
            method="test",
            layer_norms=mx.array([0.8, 0.6]),
        )
        comp = compare_directions(result_a, result_b, "a", "b")
        assert comp.cosine_sims.shape == (2,)
        # Cosine similarity should be between -1 and 1
        assert mx.min(comp.cosine_sims).item() >= -1.01
        assert mx.max(comp.cosine_sims).item() <= 1.01

    def test_stream_analysis(self) -> None:
        """Stream analysis produces valid concentration metrics."""
        result = MultiStreamExtractionResult(
            per_stream=mx.random.normal((2, 4, 64)),
            joint=mx.random.normal((2, 256)),
            stream_norms=mx.abs(mx.random.normal((2, 4))),
            joint_norms=mx.array([1.0, 0.5]),
        )
        analysis = analyze_stream_distribution(result)
        assert analysis.concentration.shape == (2,)
        assert analysis.dominant_stream.shape == (2,)
        assert analysis.norm_ratios.shape == (2, 4)
        # Concentration should be between 0 and 1
        assert mx.min(analysis.concentration).item() >= -0.01
        assert mx.max(analysis.concentration).item() <= 1.01

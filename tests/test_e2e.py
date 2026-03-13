"""End-to-end smoke tests: tiny model, few steps, all three architectures.

Verifies the full pipeline works: model creation → training → direction extraction
→ abliteration → steering → comparison.
"""

import mlx.core as mx

from kromcanon.config import ModelConfig, TrainConfig
from kromcanon.data import PretrainDataLoader
from kromcanon.interp.abliterate import abliterate_model
from kromcanon.interp.compare import compare_directions
from kromcanon.interp.extract import (
    collect_activations,
    extract_mean_diff,
)
from kromcanon.interp.steer import SteeringConfig, steer_forward
from kromcanon.model import GPT2
from kromcanon.train import create_optimizer, train_step


def _tiny_config(arch: str) -> ModelConfig:
    """Minimal config for smoke testing."""
    return ModelConfig(
        arch=arch,
        vocab_size=128,
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ff=128,
        max_seq_len=16,
    )


class TestEndToEnd:
    """End-to-end pipeline tests."""

    def test_vanilla_pipeline(self) -> None:
        """Full pipeline for vanilla architecture."""
        self._run_pipeline("vanilla")

    def test_canon_pipeline(self) -> None:
        """Full pipeline for canon architecture."""
        self._run_pipeline("canon")

    def test_kromcanon_pipeline(self) -> None:
        """Full pipeline for kromcanon architecture."""
        self._run_pipeline("kromcanon")

    def _run_pipeline(self, arch: str) -> None:
        """Run full pipeline for a given architecture.

        Steps:
        1. Create model
        2. Train for 3 steps
        3. Extract directions
        4. Abliterate
        5. Steer
        """
        # 1. Create model
        config = _tiny_config(arch)
        model = GPT2(config)

        # 2. Train for a few steps
        train_config = TrainConfig(lr=1e-3, warmup_steps=0, max_steps=3)
        optimizer = create_optimizer(model, config, train_config)
        for _ in range(3):
            input_ids = mx.random.randint(0, 128, (2, 15))
            target_ids = mx.random.randint(0, 128, (2, 15))
            loss = train_step(model, input_ids, target_ids, optimizer)
            assert loss.item() > 0

        # 3. Extract directions
        harmful = [mx.random.randint(0, 128, (1, 8)) for _ in range(5)]
        harmless = [mx.random.randint(0, 128, (1, 8)) for _ in range(5)]
        harmful_acts = collect_activations(model, harmful)
        harmless_acts = collect_activations(model, harmless)
        result = extract_mean_diff(harmful_acts, harmless_acts)
        assert result.directions.shape == (2, 32)

        # 4. Abliterate (use direction from layer 0)
        direction = result.directions[0]
        abliterate_model(model, direction, layers=[0])

        # 5. Steer
        steering = SteeringConfig(direction=direction, alpha=1.0, layers=[0])
        input_ids = mx.random.randint(0, 128, (1, 8))
        logits = steer_forward(model, input_ids, steering)
        assert logits.shape == (1, 8, 128)

    def test_cross_architecture_comparison(self) -> None:
        """Compare directions across all three architectures."""
        results = {}
        for arch in ["vanilla", "canon", "kromcanon"]:
            config = _tiny_config(arch)
            model = GPT2(config)
            harmful = [mx.random.randint(0, 128, (1, 8)) for _ in range(5)]
            harmless = [mx.random.randint(0, 128, (1, 8)) for _ in range(5)]
            harmful_acts = collect_activations(model, harmful)
            harmless_acts = collect_activations(model, harmless)
            results[arch] = extract_mean_diff(harmful_acts, harmless_acts)

        # Compare vanilla vs canon
        comp = compare_directions(
            results["vanilla"], results["canon"], "vanilla", "canon"
        )
        assert comp.cosine_sims.shape == (2,)

        # Compare vanilla vs kromcanon
        comp = compare_directions(
            results["vanilla"], results["kromcanon"], "vanilla", "kromcanon"
        )
        assert comp.cosine_sims.shape == (2,)

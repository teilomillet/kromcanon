"""Tests for training loop."""

import mlx.core as mx

from kromcanon.config import ModelConfig, TrainConfig
from kromcanon.data import PretrainDataLoader, pack_sequences
from kromcanon.model import GPT2
from kromcanon.train import compute_loss, create_optimizer, evaluate, train_step


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


def _random_loader(
    n_sequences: int = 32, seq_len: int = 32, batch_size: int = 4
) -> PretrainDataLoader:
    """Create a data loader with random data."""
    sequences = [mx.random.randint(0, 256, (seq_len,)).tolist() for _ in range(n_sequences)]
    return PretrainDataLoader(sequences, batch_size=batch_size, shuffle=False)


class TestComputeLoss:
    """Tests for loss computation."""

    def test_loss_is_scalar(self) -> None:
        """Loss should be a scalar."""
        config = _small_config()
        model = GPT2(config)
        input_ids = mx.random.randint(0, 256, (2, 16))
        target_ids = mx.random.randint(0, 256, (2, 16))
        loss = compute_loss(model, input_ids, target_ids)
        assert loss.shape == ()

    def test_loss_is_positive(self) -> None:
        """Cross-entropy loss should be positive."""
        config = _small_config()
        model = GPT2(config)
        input_ids = mx.random.randint(0, 256, (2, 16))
        target_ids = mx.random.randint(0, 256, (2, 16))
        loss = compute_loss(model, input_ids, target_ids)
        assert loss.item() > 0


class TestTrainStep:
    """Tests for single training step."""

    def test_loss_decreases_over_steps(self) -> None:
        """Loss should generally decrease over a few training steps on fixed data."""
        config = _small_config()
        model = GPT2(config)
        train_config = TrainConfig(lr=1e-3, warmup_steps=0, max_steps=10)
        optimizer = create_optimizer(model, config, train_config)

        input_ids = mx.random.randint(0, 256, (4, 16))
        target_ids = mx.random.randint(0, 256, (4, 16))

        losses: list[float] = []
        for _ in range(10):
            loss = train_step(model, input_ids, target_ids, optimizer)
            losses.append(loss.item())

        # Loss at end should be lower than at start (on same data)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_all_architectures_train(self) -> None:
        """All three architectures can execute a training step."""
        for arch in ["vanilla", "canon", "kromcanon"]:
            config = _small_config(arch)
            model = GPT2(config)
            train_config = TrainConfig(lr=1e-3, warmup_steps=0, max_steps=1)
            optimizer = create_optimizer(model, config, train_config)

            input_ids = mx.random.randint(0, 256, (2, 16))
            target_ids = mx.random.randint(0, 256, (2, 16))

            loss = train_step(model, input_ids, target_ids, optimizer)
            assert loss.item() > 0, f"{arch} training step failed"


class TestDataLoader:
    """Tests for data loading."""

    def test_pack_sequences(self) -> None:
        """Packing produces correct length sequences."""
        docs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        packed = pack_sequences(docs, seq_len=4, bos_token=0)
        assert all(len(s) == 4 for s in packed)

    def test_dataloader_shapes(self) -> None:
        """DataLoader yields correct shapes."""
        loader = _random_loader(n_sequences=16, seq_len=32, batch_size=4)
        for input_ids, target_ids in loader:
            assert input_ids.shape == (4, 31)  # seq_len - 1
            assert target_ids.shape == (4, 31)
            break

    def test_dataloader_len(self) -> None:
        """DataLoader reports correct number of batches."""
        loader = _random_loader(n_sequences=16, seq_len=32, batch_size=4)
        assert len(loader) == 4


class TestEvaluate:
    """Tests for evaluation."""

    def test_evaluate_returns_float(self) -> None:
        """Evaluate returns a float loss."""
        config = _small_config()
        model = GPT2(config)
        loader = _random_loader(n_sequences=8, seq_len=32, batch_size=4)
        loss = evaluate(model, loader)
        assert isinstance(loss, float)
        assert loss > 0

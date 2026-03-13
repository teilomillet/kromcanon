"""Tests for safety data loading and SFT."""

import mlx.core as mx

from kromcanon.config import ModelConfig, TrainConfig
from kromcanon.model import GPT2
from kromcanon.safety_data import (
    ConversationPair,
    _parse_hh_conversation,
    format_for_sft,
    iter_safety_batches,
    tokenize_conversations,
)
from kromcanon.sft import sft_train


class TestParseHHConversation:
    """Tests for HH-RLHF conversation parsing."""

    def test_basic_parse(self) -> None:
        """Parse a standard HH-RLHF conversation."""
        text = "\n\nHuman: Hello\n\nAssistant: Hi there!"
        result = _parse_hh_conversation(text)
        assert result is not None
        user, assistant = result
        assert user == "Hello"
        assert assistant == "Hi there!"

    def test_multi_turn(self) -> None:
        """Parse first turn of multi-turn conversation."""
        text = (
            "\n\nHuman: First question\n\nAssistant: First answer"
            "\n\nHuman: Second question\n\nAssistant: Second answer"
        )
        result = _parse_hh_conversation(text)
        assert result is not None
        user, assistant = result
        assert user == "First question"
        assert assistant == "First answer"

    def test_invalid_format(self) -> None:
        """Return None for unparseable text."""
        result = _parse_hh_conversation("just some random text")
        assert result is None


class TestFormatForSFT:
    """Tests for SFT formatting."""

    def test_format(self) -> None:
        """Format produces correct structure."""
        pairs = [
            ConversationPair("Hello", "Hi!", is_harmful=False),
            ConversationPair("Bad request", "I can't help with that.", is_harmful=True),
        ]
        formatted = format_for_sft(pairs)
        assert len(formatted) == 2
        assert formatted[0]["messages"][0]["role"] == "user"
        assert formatted[0]["messages"][1]["role"] == "assistant"


class TestTokenizeConversations:
    """Tests for conversation tokenization."""

    def test_tokenize(self) -> None:
        """Tokenize produces token sequences."""
        pairs = [
            ConversationPair(
                "What is the capital of France?",
                "The capital of France is Paris.",
                is_harmful=False,
            ),
        ]

        def mock_encode(text: str) -> list[int]:
            return list(range(len(text)))

        sequences = tokenize_conversations(pairs, encode_fn=mock_encode, max_len=256)
        assert len(sequences) == 1
        assert len(sequences[0]) <= 256

    def test_truncation(self) -> None:
        """Long sequences are truncated."""
        pairs = [
            ConversationPair("x" * 1000, "y" * 1000, is_harmful=False),
        ]

        def mock_encode(text: str) -> list[int]:
            return list(range(len(text)))

        sequences = tokenize_conversations(pairs, encode_fn=mock_encode, max_len=100)
        assert len(sequences) == 1
        assert len(sequences[0]) == 100


class TestIterSafetyBatches:
    """Tests for safety batch iteration."""

    def test_batch_shapes(self) -> None:
        """Batches have correct shapes."""
        sequences = [list(range(32)) for _ in range(16)]
        batches = list(iter_safety_batches(sequences, batch_size=4, seq_len=32))
        assert len(batches) == 4
        for input_ids, target_ids in batches:
            assert input_ids.shape == (4, 31)
            assert target_ids.shape == (4, 31)


class TestSFTTrain:
    """Tests for SFT training loop."""

    def test_sft_runs(self) -> None:
        """SFT training loop executes without errors."""
        config = ModelConfig(
            arch="vanilla", vocab_size=256, n_layers=2,
            n_heads=4, d_model=64, d_ff=256, max_seq_len=32,
        )
        model = GPT2(config)
        train_config = TrainConfig(lr=1e-3, checkpoint_dir="checkpoints/test_sft")

        # Generate fake data
        sequences = [mx.random.randint(0, 256, (32,)).tolist() for _ in range(32)]
        train_data = iter_safety_batches(sequences, batch_size=8, seq_len=32)

        logs = sft_train(
            model, train_data, config, train_config, max_steps=5
        )
        # Should have at least logged once (at step 10, but we only do 5 steps)
        # No error = success
        assert isinstance(logs, list)

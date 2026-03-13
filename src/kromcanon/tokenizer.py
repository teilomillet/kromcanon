"""BPE tokenizer wrapper for KromCanon.

Uses the GPT-2 tokenizer from HuggingFace as the default.
Can be swapped for a custom-trained BPE tokenizer later.
"""

from pathlib import Path

import mlx.core as mx


class Tokenizer:
    """BPE tokenizer backed by HuggingFace's tokenizers library.

    Args:
        path: Path to a tokenizer.json file, or None to use GPT-2 default.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        from tokenizers import Tokenizer as HFTokenizer

        if path is not None:
            self._tokenizer = HFTokenizer.from_file(str(path))
        else:
            from tokenizers import decoders, models, pre_tokenizers

            # Fall back to a basic BPE — real usage should provide a trained tokenizer
            self._tokenizer = HFTokenizer(models.BPE())
            self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self._tokenizer.decoder = decoders.ByteLevel()

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self._tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input string.

        Returns:
            List of integer token IDs.
        """
        return self._tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text.

        Args:
            ids: List of integer token IDs.

        Returns:
            Decoded string.
        """
        return self._tokenizer.decode(ids)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode a batch of texts.

        Args:
            texts: List of input strings.

        Returns:
            List of token ID lists.
        """
        return [enc.ids for enc in self._tokenizer.encode_batch(texts)]

    def to_array(self, ids: list[int]) -> mx.array:
        """Convert token IDs to an MLX array.

        Args:
            ids: List of integer token IDs.

        Returns:
            MLX array of shape (len(ids),) with dtype int32.
        """
        return mx.array(ids, dtype=mx.int32)

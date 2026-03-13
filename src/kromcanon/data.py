"""Data loading for pretraining on FineWeb-Edu.

Handles streaming from HuggingFace datasets, tokenization, and sequence packing.
"""

from collections.abc import Iterator

import mlx.core as mx
import numpy as np


def load_fineweb_edu(
    split: str = "train",
    name: str = "sample-10BT",
    streaming: bool = True,
) -> Iterator[str]:
    """Load FineWeb-Edu dataset as a text iterator.

    Args:
        split: Dataset split ("train").
        name: FineWeb-Edu subset name.
        streaming: Whether to stream (recommended for large datasets).

    Yields:
        Text strings from the dataset.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=name,
        split=split,
        streaming=streaming,
    )
    for example in ds:
        text = example.get("text", "")
        if text:
            yield text


def pack_sequences(
    token_ids: list[list[int]],
    seq_len: int,
    bos_token: int = 0,
) -> list[list[int]]:
    """Pack tokenized documents into fixed-length sequences with BOS alignment.

    Concatenates documents separated by BOS tokens, then splits into
    fixed-length chunks. Incomplete final chunk is discarded.

    Args:
        token_ids: List of tokenized documents (each is a list of int).
        seq_len: Target sequence length.
        bos_token: BOS token ID to insert between documents.

    Returns:
        List of packed sequences, each of length seq_len.
    """
    # Concatenate all documents with BOS separators
    flat: list[int] = []
    for doc in token_ids:
        flat.append(bos_token)
        flat.extend(doc)

    # Split into fixed-length chunks
    n_sequences = len(flat) // seq_len
    sequences: list[list[int]] = []
    for i in range(n_sequences):
        start = i * seq_len
        sequences.append(flat[start : start + seq_len])

    return sequences


class PretrainDataLoader:
    """Data loader for pretraining that yields batches of packed sequences.

    Args:
        sequences: List of packed token sequences (each length seq_len).
        batch_size: Number of sequences per batch.
        shuffle: Whether to shuffle sequences each epoch.
    """

    def __init__(
        self,
        sequences: list[list[int]],
        batch_size: int,
        shuffle: bool = True,
    ) -> None:
        self.data = np.array(sequences, dtype=np.int32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_sequences = len(sequences)

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return self.n_sequences // self.batch_size

    def __iter__(self) -> Iterator[tuple[mx.array, mx.array]]:
        """Yield (input, target) batches.

        Input is tokens[:-1], target is tokens[1:] (next-token prediction).

        Yields:
            Tuple of (input_ids, target_ids), each shape (batch_size, seq_len - 1).
        """
        indices = np.arange(self.n_sequences)
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, self.n_sequences - self.batch_size + 1, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch = self.data[batch_indices]  # (batch_size, seq_len)
            input_ids = mx.array(batch[:, :-1])
            target_ids = mx.array(batch[:, 1:])
            yield input_ids, target_ids


def prepare_pretraining_data(
    texts: Iterator[str],
    encode_fn: callable,
    seq_len: int,
    max_tokens: int = 1_200_000_000,
    bos_token: int = 0,
) -> list[list[int]]:
    """Tokenize and pack texts into training sequences.

    Args:
        texts: Iterator of text strings.
        encode_fn: Tokenizer encode function (str → list[int]).
        seq_len: Target sequence length.
        max_tokens: Maximum total tokens to process.
        bos_token: BOS token ID.

    Returns:
        List of packed sequences.
    """
    all_tokens: list[list[int]] = []
    total = 0

    for text in texts:
        ids = encode_fn(text)
        all_tokens.append(ids)
        total += len(ids)
        if total >= max_tokens:
            break

    return pack_sequences(all_tokens, seq_len=seq_len, bos_token=bos_token)

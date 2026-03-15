"""Data loading for pretraining on FineWeb-Edu.

Handles streaming from HuggingFace datasets, tokenization, and sequence packing.
"""

import hashlib
from collections.abc import Iterator
from pathlib import Path

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
        sequences: Packed token sequences as a numpy array (n_seqs, seq_len)
            or a list of lists (legacy, will be converted).
        batch_size: Number of sequences per batch.
        shuffle: Whether to shuffle sequences each epoch.
    """

    def __init__(
        self,
        sequences: np.ndarray | list[list[int]],
        batch_size: int,
        shuffle: bool = True,
    ) -> None:
        if isinstance(sequences, np.ndarray):
            self.data = sequences
        else:
            self.data = np.array(sequences, dtype=np.int32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_sequences = self.data.shape[0]

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


def _cache_path(
    seq_len: int,
    max_tokens: int,
    cache_dir: Path | None,
) -> Path | None:
    """Build deterministic cache path from data parameters."""
    if cache_dir is None:
        return None
    key = f"fineweb_edu_seq{seq_len}_tok{max_tokens}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]  # noqa: S324
    return cache_dir / f"packed_{h}.npy"


def prepare_pretraining_data(
    texts: Iterator[str],
    encode_fn: callable,
    seq_len: int,
    max_tokens: int = 1_200_000_000,
    bos_token: int = 0,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Tokenize and pack texts into training sequences.

    If cache_dir is provided and a cache exists, loads from disk instead
    of re-tokenizing. This saves ~8GB peak memory on repeated runs.

    Uses streaming tokenization to keep peak memory under ~1GB: tokens
    are packed into a pre-allocated numpy array as they arrive, avoiding
    the 2.8GB overhead of accumulating Python int lists.

    Args:
        texts: Iterator of text strings.
        encode_fn: Tokenizer encode function (str → list[int]).
        seq_len: Target sequence length.
        max_tokens: Maximum total tokens to process.
        bos_token: BOS token ID.
        cache_dir: Directory for caching packed sequences. None to disable.

    Returns:
        Packed sequences as numpy array, shape (n_sequences, seq_len).
    """
    cp = _cache_path(seq_len, max_tokens, cache_dir)
    if cp is not None and cp.exists():
        print(f"  Loading cached sequences from {cp}")
        return np.load(cp)

    # Stream tokenize directly into a flat numpy buffer to avoid
    # accumulating Python int lists (saves ~7GB peak memory).
    max_seqs = max_tokens // seq_len + 1
    buf = np.empty((max_seqs, seq_len), dtype=np.int32)
    pos = 0  # position in current sequence
    seq_idx = 0  # current sequence index
    total = 0

    for text in texts:
        ids = encode_fn(text)
        # Prepend BOS separator
        tokens = [bos_token, *ids]
        i = 0
        while i < len(tokens):
            space = seq_len - pos
            chunk = tokens[i : i + space]
            buf[seq_idx, pos : pos + len(chunk)] = chunk
            pos += len(chunk)
            i += len(chunk)
            if pos >= seq_len:
                seq_idx += 1
                pos = 0
                if seq_idx >= max_seqs:
                    break
        total += len(ids)
        if total >= max_tokens or seq_idx >= max_seqs:
            break

    # Trim to actual number of complete sequences
    buf = buf[:seq_idx]
    print(f"  Tokenized {total:,} tokens into {seq_idx:,} sequences")

    if cp is not None:
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.save(cp, buf)
        print(f"  Cached to {cp} ({buf.nbytes / 1e6:.0f} MB)")

    return buf

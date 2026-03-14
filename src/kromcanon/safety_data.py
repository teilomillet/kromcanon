"""Safety fine-tuning data: load and format HH-RLHF and BeaverTails datasets.

Converts safety-relevant datasets into conversation-format SFT examples:
- Harmful prompts → refusal responses
- Helpful prompts → helpful responses
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class ConversationPair:
    """A single conversation turn for SFT.

    Attributes:
        user_message: The user's prompt.
        assistant_message: The desired assistant response.
        is_harmful: Whether the user message is harmful (for analysis).
    """

    user_message: str
    assistant_message: str
    is_harmful: bool


def load_hh_rlhf(
    split: str = "train",
    max_examples: int | None = None,
) -> list[ConversationPair]:
    """Load Anthropic HH-RLHF dataset as conversation pairs.

    Uses the 'chosen' responses as training targets. Classifies examples
    based on which split they come from (helpful-base vs harmless-base).

    Args:
        split: Dataset split ("train" or "test").
        max_examples: Maximum number of examples to load (None for all).

    Returns:
        List of ConversationPair instances.
    """
    from datasets import load_dataset

    pairs: list[ConversationPair] = []

    for subset in ["helpful-base", "harmless-base"]:
        ds = load_dataset("Anthropic/hh-rlhf", data_dir=subset, split=split)
        is_harmful = subset == "harmless-base"  # harmless-base has harmful prompts + refusals

        for _i, example in enumerate(ds):
            if max_examples is not None and len(pairs) >= max_examples:
                break

            chosen = example["chosen"]
            parsed = _parse_hh_conversation(chosen)
            if parsed is not None:
                user_msg, assistant_msg = parsed
                pairs.append(ConversationPair(
                    user_message=user_msg,
                    assistant_message=assistant_msg,
                    is_harmful=is_harmful,
                ))

        if max_examples is not None and len(pairs) >= max_examples:
            break

    return pairs[:max_examples] if max_examples else pairs


def load_beavertails(
    split: str = "train",
    max_examples: int | None = None,
) -> list[ConversationPair]:
    """Load BeaverTails dataset as conversation pairs.

    Uses safety labels to classify harmful vs safe examples.

    Args:
        split: Dataset split.
        max_examples: Maximum number of examples to load.

    Returns:
        List of ConversationPair instances.
    """
    from datasets import load_dataset

    # BeaverTails uses '330k_train'/'30k_train' instead of 'train'
    bt_split = f"330k_{split}" if split in ("train", "test") else split
    ds = load_dataset("PKU-Alignment/BeaverTails", split=bt_split)
    pairs: list[ConversationPair] = []

    for i, example in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            break

        prompt = example.get("prompt", "")
        response = example.get("response", "")
        is_safe = example.get("is_safe", True)

        if not prompt or not response:
            continue

        pairs.append(ConversationPair(
            user_message=prompt,
            assistant_message=response,
            is_harmful=not is_safe,
        ))

    return pairs


def format_for_sft(
    pairs: list[ConversationPair],
) -> list[dict[str, list[dict[str, str]]]]:
    """Format conversation pairs into SFT training format.

    Args:
        pairs: List of conversation pairs.

    Returns:
        List of dicts with "messages" key containing role/content dicts.
    """
    formatted: list[dict[str, list[dict[str, str]]]] = []
    for pair in pairs:
        formatted.append({
            "messages": [
                {"role": "user", "content": pair.user_message},
                {"role": "assistant", "content": pair.assistant_message},
            ]
        })
    return formatted


def tokenize_conversations(
    pairs: list[ConversationPair],
    encode_fn: callable,
    max_len: int = 512,
) -> list[list[int]]:
    """Tokenize conversation pairs into token sequences.

    Format: <user_tokens> <sep> <assistant_tokens>
    Truncated to max_len.

    Args:
        pairs: Conversation pairs.
        encode_fn: Tokenizer encode function.
        max_len: Maximum sequence length.

    Returns:
        List of token sequences.
    """
    sequences: list[list[int]] = []
    for pair in pairs:
        prompt_text = f"User: {pair.user_message}\nAssistant: {pair.assistant_message}"
        tokens = encode_fn(prompt_text)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        if len(tokens) > 10:  # Skip very short sequences
            sequences.append(tokens)
    return sequences


def iter_safety_batches(
    sequences: list[list[int]],
    batch_size: int,
    seq_len: int,
) -> Iterator[tuple[list[list[int]], list[list[int]]]]:
    """Yield padded batches of (input, target) for SFT.

    Args:
        sequences: List of token sequences (variable length).
        batch_size: Batch size.
        seq_len: Pad/truncate to this length.

    Yields:
        Tuples of (input_ids, target_ids) as nested lists.
    """
    import mlx.core as mx
    import numpy as np

    # Pad sequences to seq_len
    padded = np.zeros((len(sequences), seq_len), dtype=np.int32)
    for i, seq in enumerate(sequences):
        length = min(len(seq), seq_len)
        padded[i, :length] = seq[:length]

    # Shuffle
    indices = np.random.permutation(len(sequences))
    padded = padded[indices]

    for i in range(0, len(sequences) - batch_size + 1, batch_size):
        batch = padded[i : i + batch_size]
        input_ids = mx.array(batch[:, :-1])
        target_ids = mx.array(batch[:, 1:])
        yield input_ids, target_ids


def load_test_prompts(
    max_examples: int = 100,
) -> tuple[list[str], list[str]]:
    """Load harmful and harmless test prompts from HH-RLHF test split.

    Uses the test split to avoid train/test data leakage during direction
    extraction. Returns raw prompt strings (not tokenized).

    Args:
        max_examples: Maximum number of prompts per category.

    Returns:
        Tuple of (harmful_prompts, harmless_prompts) as string lists.
    """
    from datasets import load_dataset

    harmful: list[str] = []
    harmless: list[str] = []

    for subset, target in [("harmless-base", harmful), ("helpful-base", harmless)]:
        ds = load_dataset("Anthropic/hh-rlhf", data_dir=subset, split="test")
        for example in ds:
            if len(target) >= max_examples:
                break
            parsed = _parse_hh_conversation(example["chosen"])
            if parsed is not None:
                target.append(parsed[0])

    return harmful, harmless


def _parse_hh_conversation(text: str) -> tuple[str, str] | None:
    """Parse HH-RLHF conversation format into (user, assistant) pair.

    HH-RLHF format: "\\n\\nHuman: <text>\\n\\nAssistant: <text>..."

    Args:
        text: Raw conversation text.

    Returns:
        Tuple of (user_message, assistant_message) or None if parsing fails.
    """
    parts = text.split("\n\nHuman: ")
    if len(parts) < 2:
        return None

    # Take the first human-assistant exchange
    rest = parts[1]
    assistant_parts = rest.split("\n\nAssistant: ")
    if len(assistant_parts) < 2:
        return None

    user_msg = assistant_parts[0].strip()
    assistant_msg = assistant_parts[1].split("\n\nHuman:")[0].strip()

    if not user_msg or not assistant_msg:
        return None

    return user_msg, assistant_msg

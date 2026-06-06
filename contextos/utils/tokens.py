"""
ContextOS — tiktoken-based token counter.

Wraps tiktoken with:
- Caching of encoder objects (expensive to construct)
- Fallback word-count estimate if tiktoken unavailable
- Batch counting
"""
from __future__ import annotations

import functools
import re


@functools.lru_cache(maxsize=8)
def _get_encoder(model: str):
    """Cache encoder objects — construction is slow (~100ms)."""
    try:
        import tiktoken
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        return None


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count the number of tokens in `text` for the given model.

    Falls back to a word-count approximation (~1.3 tokens/word) if
    tiktoken is not installed.
    """
    if not text:
        return 0

    encoder = _get_encoder(model)
    if encoder is not None:
        return len(encoder.encode(text))

    # Fallback: rough approximation
    words = len(re.split(r"\s+", text.strip()))
    return max(1, int(words * 1.3))


def count_tokens_batch(texts: list[str], model: str = "gpt-4o-mini") -> list[int]:
    """Count tokens for a list of texts. Uses batch encoding when possible."""
    if not texts:
        return []

    encoder = _get_encoder(model)
    if encoder is not None:
        return [len(enc) for enc in encoder.encode_batch(texts)]

    return [count_tokens(t, model) for t in texts]


def fits_in_budget(text: str, budget: int, model: str = "gpt-4o-mini") -> bool:
    """Quick check: does `text` fit within `budget` tokens?"""
    return count_tokens(text, model) <= budget

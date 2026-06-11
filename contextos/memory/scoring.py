"""
ContextOS — importance scoring engine.

Every MemoryBlock has a composite importance score that drives scheduling
and eviction decisions. This module computes and updates those scores.

Formula:
    score = α·similarity + β·recency + γ·frequency + δ·user_weight

    α = 0.40  (semantic similarity to the current query)
    β = 0.30  (how recently was this block accessed — exponential decay)
    γ = 0.20  (how often was this block accessed — log-normalized)
    δ = 0.10  (user-assigned importance hint)
"""
from __future__ import annotations

import math
from datetime import datetime

from contextos.models.block import MemoryBlock
from contextos.utils.embedder import Embedder


def recency_score(last_accessed: datetime, half_life_hours: float = 24.0) -> float:
    """
    Exponential decay: score = 2^(-hours_elapsed / half_life).

    Returns 1.0 for a block accessed right now, 0.5 after half_life_hours,
    0.25 after 2*half_life_hours, etc.
    """
    elapsed_seconds = (datetime.utcnow() - last_accessed).total_seconds()
    elapsed_hours = elapsed_seconds / 3600.0
    return 2.0 ** (-elapsed_hours / half_life_hours)


def frequency_score(frequency: int, max_frequency: int = 1) -> float:
    """
    Log-normalised frequency score in [0, 1].

    log1p(freq) / log1p(max_freq) so that the most-accessed block scores 1.0.
    When max_frequency == 0, returns 0.0 (no access data yet).
    """
    if max_frequency <= 0:
        return 0.0
    return math.log1p(frequency) / math.log1p(max(max_frequency, 1))


def compute_importance(
    block: MemoryBlock,
    query_embedding: list[float],
    max_frequency: int = 1,
    half_life_hours: float = 24.0,
    # weights
    w_similarity: float = 0.40,
    w_recency: float = 0.30,
    w_frequency: float = 0.20,
    w_user: float = 0.10,
) -> float:
    """
    Compute the composite importance score for a single MemoryBlock.

    Args:
        block:           The block to score.
        query_embedding: Embedding of the current query. Pass an empty list
                         when there is no query (all similarity weight goes to 0).
        max_frequency:   The highest frequency seen across all blocks in the store.
                         Used to normalise the frequency component.
        half_life_hours: Recency decay half-life.
        w_*:             Component weights. Must sum to 1.0.

    Returns:
        float in [0.0, 1.0]
    """
    # Similarity — 0 if no query or no embedding stored
    if query_embedding and block.embedding:
        sim = Embedder.cosine_similarity(block.embedding, query_embedding)
        sim = (sim + 1.0) / 2.0  # remap [-1, 1] → [0, 1]
    else:
        sim = 0.0

    rec = recency_score(block.last_accessed, half_life_hours)
    freq = frequency_score(block.frequency, max_frequency)
    user = min(1.0, max(0.0, block.user_weight))

    return w_similarity * sim + w_recency * rec + w_frequency * freq + w_user * user


def recompute_scores_for_blocks(
    blocks: list[MemoryBlock],
    query_embedding: list[float],
    half_life_hours: float = 24.0,
    w_similarity: float = 0.40,
    w_recency: float = 0.30,
    w_frequency: float = 0.20,
    w_user: float = 0.10,
) -> None:
    """
    Recompute and mutate importance_score and recency_score on every block.
    Called by ContextScheduler before scheduling.
    """
    max_freq = max((b.frequency for b in blocks), default=0)

    for block in blocks:
        block.recency_score = recency_score(block.last_accessed, half_life_hours)
        block.importance_score = compute_importance(
            block,
            query_embedding,
            max_frequency=max_freq,
            half_life_hours=half_life_hours,
            w_similarity=w_similarity,
            w_recency=w_recency,
            w_frequency=w_frequency,
            w_user=w_user,
        )

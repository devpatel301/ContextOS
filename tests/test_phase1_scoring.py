"""
Phase 1 — Importance Scoring tests.

Tests scoring functions in isolation (pure math, no DB or LLM).
"""
import math
from datetime import datetime, timedelta

import pytest

from contextos.memory.scoring import (
    compute_importance,
    frequency_score,
    recency_score,
    recompute_scores_for_blocks,
)
from contextos.models.block import MemoryBlock
from contextos.models.tier import MemoryTier
from contextos.utils.embedder import Embedder


class TestRecencyScore:
    def test_fresh_access_is_one(self):
        score = recency_score(datetime.utcnow(), half_life_hours=24.0)
        assert abs(score - 1.0) < 0.01  # within 1% of 1.0

    def test_after_half_life_is_half(self):
        past = datetime.utcnow() - timedelta(hours=24)
        score = recency_score(past, half_life_hours=24.0)
        assert abs(score - 0.5) < 0.01

    def test_after_two_half_lives_is_quarter(self):
        past = datetime.utcnow() - timedelta(hours=48)
        score = recency_score(past, half_life_hours=24.0)
        assert abs(score - 0.25) < 0.01

    def test_score_bounded_zero_to_one(self):
        very_old = datetime.utcnow() - timedelta(days=365)
        score = recency_score(very_old)
        assert 0.0 <= score <= 1.0

    def test_shorter_half_life_decays_faster(self):
        past = datetime.utcnow() - timedelta(hours=12)
        fast = recency_score(past, half_life_hours=6.0)
        slow = recency_score(past, half_life_hours=48.0)
        assert fast < slow


class TestFrequencyScore:
    def test_zero_frequency_is_zero(self):
        assert frequency_score(0, max_frequency=10) == 0.0

    def test_max_frequency_is_one(self):
        assert abs(frequency_score(100, max_frequency=100) - 1.0) < 1e-9

    def test_monotonically_increasing(self):
        scores = [frequency_score(i, max_frequency=100) for i in range(0, 101, 10)]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1]

    def test_zero_max_frequency_returns_zero(self):
        assert frequency_score(5, max_frequency=0) == 0.0


class TestComputeImportance:
    def _make_block(self, frequency=0, user_weight=0.5, hours_old=0) -> MemoryBlock:
        block = MemoryBlock(
            content="test",
            frequency=frequency,
            user_weight=user_weight,
            last_accessed=datetime.utcnow() - timedelta(hours=hours_old),
        )
        emb = Embedder(stub=True)
        block.embedding = emb.embed("test")
        return block

    def test_returns_value_in_zero_one(self):
        block = self._make_block()
        query_emb = Embedder(stub=True).embed("query")
        score = compute_importance(block, query_emb)
        assert 0.0 <= score <= 1.0

    def test_fresh_block_scores_higher_recency(self):
        fresh = self._make_block(hours_old=0)
        old = self._make_block(hours_old=72)
        query_emb = []  # no query context

        fresh_score = compute_importance(fresh, query_emb)
        old_score = compute_importance(old, query_emb)
        assert fresh_score > old_score

    def test_high_frequency_scores_higher(self):
        low_freq = self._make_block(frequency=0)
        high_freq = self._make_block(frequency=50)
        query_emb = []

        low_score = compute_importance(low_freq, query_emb, max_frequency=50)
        high_score = compute_importance(high_freq, query_emb, max_frequency=50)
        assert high_score > low_score

    def test_pinned_block_user_weight_contributes(self):
        normal = self._make_block(user_weight=0.5, hours_old=24)
        pinned = self._make_block(user_weight=1.0, hours_old=24)
        query_emb = []

        normal_score = compute_importance(normal, query_emb)
        pinned_score = compute_importance(pinned, query_emb)
        assert pinned_score > normal_score

    def test_no_query_embedding_still_works(self):
        block = self._make_block()
        score = compute_importance(block, query_embedding=[])
        assert 0.0 <= score <= 1.0

    def test_weights_sum_sanity(self):
        """With a query that's identical to block content, similarity ~1."""
        emb = Embedder(stub=True)
        block = self._make_block(frequency=10, user_weight=1.0, hours_old=0)
        text = "test"
        block.embedding = emb.embed(text)
        query_emb = emb.embed(text)  # identical → similarity ≈ 1

        score = compute_importance(
            block,
            query_emb,
            max_frequency=10,
            w_similarity=0.40,
            w_recency=0.30,
            w_frequency=0.20,
            w_user=0.10,
        )
        # Should be very high: similarity=1, recency≈1, frequency=1, user=1
        assert score > 0.9


class TestRecomputeScoresForBlocks:
    def test_mutates_blocks_in_place(self):
        emb = Embedder(stub=True)
        blocks = [
            MemoryBlock(
                content="block one",
                embedding=emb.embed("block one"),
                importance_score=0.0,
                recency_score=0.0,
            ),
            MemoryBlock(
                content="block two",
                embedding=emb.embed("block two"),
                importance_score=0.0,
                recency_score=0.0,
            ),
        ]
        query_emb = emb.embed("block one")
        recompute_scores_for_blocks(blocks, query_emb)

        # Scores should be non-zero after recompute
        for b in blocks:
            assert b.importance_score > 0.0
            assert b.recency_score > 0.0

    def test_relevant_block_scores_higher(self):
        emb = Embedder(stub=True)
        relevant = MemoryBlock(
            content="the cat sat on the mat",
            embedding=emb.embed("the cat sat on the mat"),
        )
        irrelevant = MemoryBlock(
            content="quantum physics superposition",
            embedding=emb.embed("quantum physics superposition"),
        )
        query_emb = emb.embed("the cat sat on the mat")

        # Note: with stub embedder, similarity is hash-based not semantic.
        # Identical text should get highest similarity.
        recompute_scores_for_blocks([relevant, irrelevant], query_emb)
        # The relevant block should score at least as high
        assert relevant.importance_score >= irrelevant.importance_score - 0.01

    def test_empty_block_list(self):
        """Should not raise."""
        recompute_scores_for_blocks([], [])

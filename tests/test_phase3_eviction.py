"""
Phase 3 — Eviction Policies + Garbage Collector tests.

Tests all 5 eviction policies and the 3-pass GC.
Uses stub embedder (no GPU needed).
"""
from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from contextos import ContextConfig, ContextOS, MemoryTier
from contextos.eviction.policies import (
    FIFOEviction,
    HybridEviction,
    LFUEviction,
    LRUEviction,
    PriorityEviction,
    get_eviction_policy,
)
from contextos.models.block import MemoryBlock
from contextos.utils.embedder import Embedder


@pytest.fixture
def cos(tmp_path):
    config = ContextConfig(
        db_path=str(tmp_path / "evict.db"),
        embedding_stub=True,
        llm_provider="none",
        token_budget=8_000,
        episodic_tier_budget=100,  # tiny budget to trigger eviction
    )
    with ContextOS(config) as c:
        yield c


# ── Eviction Policy Unit Tests ────────────────────────────────────────────────

def make_blocks(count=5, tokens_each=20):
    """Create blocks with varying ages, frequencies, and scores."""
    blocks = []
    for i in range(count):
        blocks.append(MemoryBlock(
            content=f"block {i}",
            token_count=tokens_each,
            created_at=datetime(2025, 1, 1 + i),
            last_accessed=datetime(2025, 6, 1 + i),
            frequency=i,
            importance_score=i * 0.2,
            recency_score=i * 0.2,
        ))
    return blocks


class TestFIFOEviction:
    def test_evicts_oldest_first(self):
        blocks = make_blocks(5, tokens_each=20)
        policy = FIFOEviction()
        victims = policy.select_victims(blocks, tokens_to_free=40)
        # Oldest = lowest created_at index
        assert len(victims) == 2
        assert victims[0].content == "block 0"
        assert victims[1].content == "block 1"


class TestLRUEviction:
    def test_evicts_least_recently_accessed(self):
        blocks = make_blocks(5, tokens_each=20)
        policy = LRUEviction()
        victims = policy.select_victims(blocks, tokens_to_free=20)
        assert len(victims) == 1
        assert victims[0].content == "block 0"  # oldest last_accessed


class TestLFUEviction:
    def test_evicts_least_frequently_used(self):
        blocks = make_blocks(5, tokens_each=20)
        policy = LFUEviction()
        victims = policy.select_victims(blocks, tokens_to_free=20)
        assert len(victims) == 1
        assert victims[0].content == "block 0"  # frequency=0


class TestPriorityEviction:
    def test_evicts_lowest_importance(self):
        blocks = make_blocks(5, tokens_each=20)
        policy = PriorityEviction()
        victims = policy.select_victims(blocks, tokens_to_free=20)
        assert len(victims) == 1
        assert victims[0].content == "block 0"  # importance=0.0


class TestHybridEviction:
    def test_evicts_hybrid_score(self):
        blocks = make_blocks(5, tokens_each=20)
        policy = HybridEviction()
        victims = policy.select_victims(blocks, tokens_to_free=20)
        # Block 0 has worst recency, frequency, and importance → evicted first
        assert len(victims) == 1
        assert victims[0].content == "block 0"

    def test_frees_enough_tokens(self):
        blocks = make_blocks(5, tokens_each=20)
        policy = HybridEviction()
        victims = policy.select_victims(blocks, tokens_to_free=60)
        total = sum(b.effective_token_count() for b in victims)
        assert total >= 60


class TestEvictionPolicyRegistry:
    def test_all_policies_instantiate(self):
        for name in ["fifo", "lru", "lfu", "priority", "hybrid"]:
            p = get_eviction_policy(name)
            assert p.name == name

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            get_eviction_policy("does_not_exist")


# ── EvictionEngine Integration Tests ─────────────────────────────────────────

class TestEvictionEngine:
    def test_evict_when_over_budget(self, cos):
        # Budget is 100 tokens. Store blocks with long content to clearly exceed it.
        for i in range(10):
            cos.store(
                f"Block number {i} contains a long explanation about topic {i} "
                f"covering multiple aspects including history background context "
                f"implementation details and future directions for research area {i}",
                tier=MemoryTier.EPISODIC,
            )
        usage_before = cos.status()["tiers"]["episodic"]["tokens"]
        assert usage_before > 100, f"Expected > 100 tokens, got {usage_before}"
        evicted = cos.evict(tier=MemoryTier.EPISODIC, budget=100)
        assert len(evicted) > 0
        # After eviction, episodic should be near or under budget
        remaining = cos.status()["tiers"]["episodic"]["tokens"]
        assert remaining <= 100

    def test_evict_under_budget_does_nothing(self, cos):
        cos.store("small", tier=MemoryTier.EPISODIC)
        evicted = cos.evict(tier=MemoryTier.EPISODIC, budget=10_000)
        assert len(evicted) == 0

    def test_evict_never_touches_working(self, cos):
        for i in range(5):
            cos.store(f"working block {i}", tier=MemoryTier.WORKING)
        evicted = cos.evict(tier=MemoryTier.WORKING, budget=0)
        assert len(evicted) == 0

    def test_evict_pinned_blocks_protected(self, cos):
        cos.store("pinned content padding words", tier=MemoryTier.EPISODIC, user_weight=1.0)
        cos.store("evictable content padding words", tier=MemoryTier.EPISODIC)
        evicted = cos.evict(tier=MemoryTier.EPISODIC, budget=10)
        for v in evicted:
            assert not v.is_pinned()

    def test_evicted_blocks_move_to_archived(self, cos):
        b = cos.store("will be evicted padding", tier=MemoryTier.EPISODIC)
        cos.evict(tier=MemoryTier.EPISODIC, budget=0)
        updated = cos.get(b.id)
        # Block should now be in ARCHIVED (not deleted)
        assert updated.tier == MemoryTier.ARCHIVED

    def test_evict_default_budget_from_config(self, cos):
        # episodic_tier_budget is set to 100 in the fixture
        for i in range(10):
            cos.store(
                f"Block number {i} contains a long explanation about topic {i} "
                f"covering multiple aspects including history background context "
                f"implementation details and future directions for research area {i}",
                tier=MemoryTier.EPISODIC,
            )
        evicted = cos.evict()  # uses default tier and budget
        assert len(evicted) > 0


# ── Garbage Collector Tests ───────────────────────────────────────────────────

class TestGarbageCollector:
    def test_gc_removes_expired_blocks(self, cos):
        b = cos.store("will expire", expires_in_hours=0.001)  # expires in ~3 seconds
        import time
        time.sleep(0.1)
        # Manually set the expires_at to the past to test
        from contextos.models.block import MemoryBlock
        block = cos.get(b.id)
        block.expires_at = datetime.utcnow() - timedelta(hours=1)
        cos._store.save(block)

        report = cos.gc_run()
        assert report["expired"] >= 1
        assert cos.get(b.id) is None  # block is gone

    def test_gc_deduplicates_identical_blocks(self, cos):
        # Store same content twice → identical embedding → should be deduped
        cos.store("exact same content here")
        cos.store("exact same content here")
        assert cos.status()["total_blocks"] == 2

        report = cos.gc_run()
        assert report["deduplicated"] >= 1
        assert cos.status()["total_blocks"] == 1

    def test_gc_preserves_pinned_duplicates(self, cos):
        cos.store("pinned duplicate", user_weight=1.0)
        cos.store("pinned duplicate", user_weight=1.0)
        report = cos.gc_run()
        # Both pinned → neither should be deleted
        assert cos.status()["total_blocks"] == 2

    def test_gc_idempotent(self, cos):
        cos.store("block one")
        cos.store("block one")  # duplicate
        r1 = cos.gc_run()
        r2 = cos.gc_run()
        assert r2["total_freed"] == 0  # second run does nothing

    def test_gc_returns_report(self, cos):
        report = cos.gc_run()
        assert "expired" in report
        assert "deduplicated" in report
        assert "pruned" in report
        assert "total_freed" in report

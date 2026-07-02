"""
Phase 1 — MemoryStore tests.

Tests the SQLite persistence layer in isolation.
All tests use an in-memory SQLite database (no files written).
"""
from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from contextos.memory.store import MemoryStore
from contextos.models.block import MemoryBlock
from contextos.models.tier import MemoryTier


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def make_block(content: str, tier: MemoryTier = MemoryTier.EPISODIC, **kwargs) -> MemoryBlock:
    """Helper: create a MemoryBlock with minimal required fields."""
    # Compute token_count only if not provided in kwargs to avoid duplicate keyword error
    if "token_count" not in kwargs:
        kwargs["token_count"] = len(content.split())
    return MemoryBlock(content=content, tier=tier, **kwargs)


class TestMemoryStoreCRUD:
    def test_save_and_get(self, store):
        block = make_block("hello world")
        store.save(block)
        retrieved = store.get(block.id)
        assert retrieved is not None
        assert retrieved.content == "hello world"
        assert retrieved.id == block.id

    def test_get_nonexistent_returns_none(self, store):
        assert store.get(uuid4()) is None

    def test_save_overwrites_existing(self, store):
        block = make_block("original")
        store.save(block)
        block.content = "updated"
        store.save(block)
        retrieved = store.get(block.id)
        assert retrieved.content == "updated"

    def test_delete_existing(self, store):
        block = make_block("to delete")
        store.save(block)
        deleted = store.delete(block.id)
        assert deleted is True
        assert store.get(block.id) is None

    def test_delete_nonexistent_returns_false(self, store):
        assert store.delete(uuid4()) is False

    def test_delete_many(self, store):
        blocks = [make_block(f"block {i}") for i in range(5)]
        for b in blocks:
            store.save(b)
        count = store.delete_many([b.id for b in blocks[:3]])
        assert count == 3
        assert store.count() == 2

    def test_tier_roundtrip(self, store):
        for tier in MemoryTier:
            block = make_block(f"in {tier.value}", tier=tier)
            store.save(block)
            retrieved = store.get(block.id)
            assert retrieved.tier == tier

    def test_tags_roundtrip(self, store):
        block = make_block("tagged", tags=["user_profile", "pinned"])
        store.save(block)
        retrieved = store.get(block.id)
        assert "user_profile" in retrieved.tags
        assert "pinned" in retrieved.tags

    def test_metadata_roundtrip(self, store):
        block = make_block("with meta", metadata={"page": 3, "doc": "manual.pdf"})
        store.save(block)
        retrieved = store.get(block.id)
        assert retrieved.metadata["page"] == 3
        assert retrieved.metadata["doc"] == "manual.pdf"

    def test_embedding_roundtrip(self, store):
        embedding = [0.1, 0.2, -0.5, 0.8]
        block = make_block("with embedding", embedding=embedding)
        store.save(block)
        retrieved = store.get(block.id)
        assert len(retrieved.embedding) == len(embedding)
        for a, b in zip(retrieved.embedding, embedding):
            assert abs(a - b) < 1e-5

    def test_expires_at_roundtrip(self, store):
        expiry = datetime.utcnow() + timedelta(hours=2)
        block = make_block("expires", expires_at=expiry)
        store.save(block)
        retrieved = store.get(block.id)
        assert retrieved.expires_at is not None
        assert abs((retrieved.expires_at - expiry).total_seconds()) < 1


class TestMemoryStoreQueries:
    def test_get_by_tier(self, store):
        store.save(make_block("working", MemoryTier.WORKING))
        store.save(make_block("episodic", MemoryTier.EPISODIC))
        store.save(make_block("episodic 2", MemoryTier.EPISODIC))
        store.save(make_block("semantic", MemoryTier.SEMANTIC))

        working = store.get_by_tier(MemoryTier.WORKING)
        episodic = store.get_by_tier(MemoryTier.EPISODIC)
        semantic = store.get_by_tier(MemoryTier.SEMANTIC)
        archived = store.get_by_tier(MemoryTier.ARCHIVED)

        assert len(working) == 1
        assert len(episodic) == 2
        assert len(semantic) == 1
        assert len(archived) == 0

    def test_get_all(self, store):
        for i in range(5):
            store.save(make_block(f"block {i}"))
        assert len(store.get_all()) == 5

    def test_count(self, store):
        for i in range(3):
            store.save(make_block(f"e{i}", MemoryTier.EPISODIC))
        store.save(make_block("s", MemoryTier.SEMANTIC))
        assert store.count() == 4
        assert store.count(MemoryTier.EPISODIC) == 3
        assert store.count(MemoryTier.SEMANTIC) == 1
        assert store.count(MemoryTier.WORKING) == 0

    def test_get_by_tag(self, store):
        store.save(make_block("tagged", tags=["user_profile"]))
        store.save(make_block("also tagged", tags=["user_profile", "recent"]))
        store.save(make_block("not tagged"))
        results = store.get_by_tag("user_profile")
        assert len(results) == 2

    def test_token_usage(self, store):
        store.save(make_block("short", token_count=10))
        store.save(make_block("medium", token_count=50))
        store.save(make_block("long", token_count=200))
        assert store.token_usage() == 260

    def test_token_usage_with_summary(self, store):
        block = make_block("long content", token_count=500)
        block.summary = "short"
        block.token_count_summary = 20
        store.save(block)
        # Should count summary tokens, not content tokens
        assert store.token_usage() == 20

    def test_get_expired(self, store):
        past = datetime.utcnow() - timedelta(hours=1)
        future = datetime.utcnow() + timedelta(hours=1)
        store.save(make_block("expired", expires_at=past))
        store.save(make_block("not expired", expires_at=future))
        store.save(make_block("no expiry"))
        expired = store.get_expired()
        assert len(expired) == 1
        assert expired[0].content == "expired"

    def test_get_idle_since(self, store):
        old = make_block("old memory")
        old.last_accessed = datetime.utcnow() - timedelta(hours=48)
        store.save(old)

        fresh = make_block("fresh memory")
        store.save(fresh)

        idle = store.get_idle_since(hours=24)
        assert len(idle) == 1
        assert idle[0].content == "old memory"


class TestMemoryStoreUpdates:
    def test_update_tier(self, store):
        block = make_block("x", MemoryTier.EPISODIC)
        store.save(block)
        store.update_tier(block.id, MemoryTier.SEMANTIC)
        assert store.get(block.id).tier == MemoryTier.SEMANTIC

    def test_update_access(self, store):
        block = make_block("x")
        store.save(block)
        new_time = datetime.utcnow()
        store.update_access(block.id, new_time, frequency=5)
        updated = store.get(block.id)
        assert updated.frequency == 5

    def test_update_scores(self, store):
        block = make_block("x")
        store.save(block)
        store.update_scores(block.id, importance=0.95, recency=0.7)
        updated = store.get(block.id)
        assert abs(updated.importance_score - 0.95) < 1e-6
        assert abs(updated.recency_score - 0.7) < 1e-6

    def test_flush_tier_scores_batch(self, store):
        blocks = [make_block(f"b{i}") for i in range(10)]
        for b in blocks:
            store.save(b)
        for i, b in enumerate(blocks):
            b.importance_score = i * 0.1
            b.recency_score = 1.0 - i * 0.1
        store.flush_tier_scores(blocks)
        for i, b in enumerate(blocks):
            updated = store.get(b.id)
            assert abs(updated.importance_score - i * 0.1) < 1e-6

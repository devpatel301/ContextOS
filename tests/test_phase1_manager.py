"""
Phase 1 — MemoryManager + ContextOS integration tests.

Tests the full Phase 1 API:
    cos.store()          ← create and persist a memory
    cos.get()            ← retrieve by ID
    cos.get_tier()       ← list blocks in a tier
    cos.touch()          ← record access
    cos.promote()        ← move to hotter tier
    cos.demote()         ← move to colder tier
    cos.run_transitions() ← fire automatic tier rules
    cos.status()         ← get usage snapshot

Uses the `cos` fixture from conftest.py (stub embedder + temp SQLite).
No LLM or API keys needed.
"""
from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from contextos.models.tier import MemoryTier


class TestStoreAndRetrieve:
    def test_store_returns_block(self, cos):
        block = cos.store("User's name is Alex")
        assert block.content == "User's name is Alex"
        assert block.id is not None

    def test_store_auto_counts_tokens(self, cos):
        block = cos.store("hello world this is a test")
        assert block.token_count > 0

    def test_store_auto_embeds(self, cos):
        block = cos.store("test content")
        assert len(block.embedding) == 384

    def test_store_default_tier_is_episodic(self, cos):
        block = cos.store("default tier")
        assert block.tier == MemoryTier.EPISODIC

    def test_store_custom_tier(self, cos):
        block = cos.store("working memory", tier=MemoryTier.WORKING)
        assert block.tier == MemoryTier.WORKING

    def test_store_with_tags(self, cos):
        block = cos.store("tagged content", tags=["user_profile", "important"])
        assert "user_profile" in block.tags
        assert "important" in block.tags

    def test_store_with_metadata(self, cos):
        block = cos.store("with meta", metadata={"source_doc": "notes.txt", "page": 7})
        assert block.metadata["source_doc"] == "notes.txt"

    def test_store_pinned_user_weight(self, cos):
        block = cos.store("critical memory", user_weight=1.0)
        assert block.is_pinned()

    def test_store_with_expiry(self, cos):
        block = cos.store("short-lived", expires_in_hours=1)
        assert block.expires_at is not None
        assert block.expires_at > datetime.utcnow()

    def test_get_existing(self, cos):
        block = cos.store("findable")
        retrieved = cos.get(block.id)
        assert retrieved is not None
        assert retrieved.content == "findable"

    def test_get_by_string_uuid(self, cos):
        block = cos.store("findable by string id")
        retrieved = cos.get(str(block.id))
        assert retrieved is not None

    def test_get_nonexistent_returns_none(self, cos):
        assert cos.get(uuid4()) is None

    def test_get_tier_returns_correct_blocks(self, cos):
        cos.store("e1", tier=MemoryTier.EPISODIC)
        cos.store("e2", tier=MemoryTier.EPISODIC)
        cos.store("w1", tier=MemoryTier.WORKING)
        episodic = cos.get_tier(MemoryTier.EPISODIC)
        working = cos.get_tier(MemoryTier.WORKING)
        assert len(episodic) == 2
        assert len(working) == 1

    def test_get_all_returns_all(self, cos):
        for tier in MemoryTier:
            cos.store(f"block in {tier.value}", tier=tier)
        all_blocks = cos.get_all()
        assert len(all_blocks) == len(MemoryTier)


class TestAccessTracking:
    def test_touch_increments_frequency(self, cos):
        block = cos.store("track me")
        assert block.frequency == 0
        cos.touch(block.id)
        updated = cos.get(block.id)
        assert updated.frequency == 1

    def test_touch_multiple_times(self, cos):
        block = cos.store("often accessed")
        for _ in range(5):
            cos.touch(block.id)
        updated = cos.get(block.id)
        assert updated.frequency == 5

    def test_touch_nonexistent_returns_none(self, cos):
        result = cos.touch(uuid4())
        assert result is None

    def test_touch_by_string_uuid(self, cos):
        block = cos.store("touch by string")
        result = cos.touch(str(block.id))
        assert result is not None
        assert result.frequency == 1


class TestTierTransitions:
    def test_promote_episodic_to_working(self, cos):
        block = cos.store("promote me", tier=MemoryTier.EPISODIC)
        success = cos.promote(block.id, MemoryTier.WORKING)
        assert success is True
        updated = cos.get(block.id)
        assert updated.tier == MemoryTier.WORKING

    def test_promote_to_same_tier_returns_false(self, cos):
        block = cos.store("same tier", tier=MemoryTier.EPISODIC)
        success = cos.promote(block.id, MemoryTier.EPISODIC)
        assert success is False

    def test_promote_to_colder_tier_returns_false(self, cos):
        block = cos.store("going wrong way", tier=MemoryTier.EPISODIC)
        success = cos.promote(block.id, MemoryTier.ARCHIVED)
        assert success is False

    def test_demote_episodic_to_semantic(self, cos):
        block = cos.store("demote me", tier=MemoryTier.EPISODIC)
        success = cos.demote(block.id, MemoryTier.SEMANTIC)
        assert success is True
        updated = cos.get(block.id)
        assert updated.tier == MemoryTier.SEMANTIC

    def test_demote_pinned_block_fails(self, cos):
        block = cos.store("pinned", tier=MemoryTier.EPISODIC, user_weight=1.0)
        success = cos.demote(block.id, MemoryTier.SEMANTIC)
        assert success is False
        updated = cos.get(block.id)
        assert updated.tier == MemoryTier.EPISODIC  # unchanged

    def test_demote_to_hotter_tier_returns_false(self, cos):
        block = cos.store("wrong direction", tier=MemoryTier.SEMANTIC)
        success = cos.demote(block.id, MemoryTier.WORKING)
        assert success is False

    def test_run_transitions_high_frequency(self, cos):
        """
        A block accessed >= 3 times should transition EPISODIC → SEMANTIC.
        """
        block = cos.store("frequently accessed", tier=MemoryTier.EPISODIC)
        for _ in range(3):
            cos.touch(block.id)
        cos.run_transitions()
        updated = cos.get(block.id)
        assert updated.tier == MemoryTier.SEMANTIC

    def test_run_transitions_returns_counts(self, cos):
        counts = cos.run_transitions()
        assert isinstance(counts, dict)
        assert "episodic_to_semantic" in counts

    def test_run_transitions_working_tier_never_demoted(self, cos):
        """WORKING tier blocks should NEVER be touched by auto-transitions."""
        block = cos.store("system prompt", tier=MemoryTier.WORKING)
        cos.run_transitions()
        updated = cos.get(block.id)
        assert updated.tier == MemoryTier.WORKING


class TestStatusSnapshot:
    def test_empty_status(self, cos):
        status = cos.status()
        assert status["total_blocks"] == 0
        for tier in MemoryTier:
            assert status["tiers"][tier.value]["blocks"] == 0
            assert status["tiers"][tier.value]["tokens"] == 0

    def test_status_counts_blocks(self, cos):
        cos.store("a", tier=MemoryTier.EPISODIC)
        cos.store("b", tier=MemoryTier.EPISODIC)
        cos.store("c", tier=MemoryTier.WORKING)
        status = cos.status()
        assert status["total_blocks"] == 3
        assert status["tiers"]["episodic"]["blocks"] == 2
        assert status["tiers"]["working"]["blocks"] == 1

    def test_status_tracks_tokens(self, cos):
        cos.store("hello world")  # ~2 tokens
        status = cos.status()
        assert status["tiers"]["episodic"]["tokens"] > 0


class TestContextManagerProtocol:
    def test_with_statement(self, stub_config):
        from contextos import ContextOS
        with ContextOS(stub_config) as c:
            block = c.store("inside context manager")
            assert block.content == "inside context manager"
        # After __exit__, connection is closed. No error should have raised.

    def test_repr_shows_stats(self, cos):
        cos.store("one")
        cos.store("two")
        r = repr(cos)
        assert "ContextOS(" in r

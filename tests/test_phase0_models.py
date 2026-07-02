"""
Phase 0 — Data model tests.

These tests verify MemoryBlock, MemoryTier, and ContextWindow
with zero external dependencies. No SQLite, no embeddings, no LLM.
"""
import math
import struct
from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from contextos.models.block import MemoryBlock
from contextos.models.context import ContextWindow
from contextos.models.tier import MemoryTier


# ── MemoryTier ────────────────────────────────────────────────────────────────

class TestMemoryTier:
    def test_tier_values_exist(self):
        assert MemoryTier.WORKING.value == "working"
        assert MemoryTier.EPISODIC.value == "episodic"
        assert MemoryTier.SEMANTIC.value == "semantic"
        assert MemoryTier.ARCHIVED.value == "archived"

    def test_ordered_returns_four_tiers(self):
        order = MemoryTier.ordered()
        assert len(order) == 4
        assert order[0] == MemoryTier.WORKING

    def test_hotter_than(self):
        assert MemoryTier.WORKING.is_hotter_than(MemoryTier.EPISODIC)
        assert MemoryTier.EPISODIC.is_hotter_than(MemoryTier.SEMANTIC)
        assert MemoryTier.SEMANTIC.is_hotter_than(MemoryTier.ARCHIVED)
        assert not MemoryTier.ARCHIVED.is_hotter_than(MemoryTier.WORKING)

    def test_colder_than(self):
        assert MemoryTier.ARCHIVED.is_colder_than(MemoryTier.SEMANTIC)
        assert MemoryTier.SEMANTIC.is_colder_than(MemoryTier.EPISODIC)
        assert not MemoryTier.WORKING.is_colder_than(MemoryTier.EPISODIC)

    def test_tier_from_string(self):
        """Enum should be constructable from its string value."""
        assert MemoryTier("episodic") == MemoryTier.EPISODIC


# ── MemoryBlock ───────────────────────────────────────────────────────────────

class TestMemoryBlock:
    def test_default_construction(self):
        block = MemoryBlock(content="hello world")
        assert block.content == "hello world"
        assert block.tier == MemoryTier.WORKING
        assert block.frequency == 0
        assert block.user_weight == 0.5
        assert block.summary is None
        assert not block.is_expired()

    def test_unique_ids(self):
        a = MemoryBlock(content="a")
        b = MemoryBlock(content="b")
        assert a.id != b.id

    def test_effective_content_no_summary(self):
        block = MemoryBlock(content="full content")
        assert block.effective_content() == "full content"

    def test_effective_content_with_summary(self):
        block = MemoryBlock(content="very long content", summary="short")
        assert block.effective_content() == "short"

    def test_effective_token_count_no_summary(self):
        block = MemoryBlock(content="test", token_count=5)
        assert block.effective_token_count() == 5

    def test_effective_token_count_with_summary(self):
        block = MemoryBlock(content="test", token_count=100, summary="short", token_count_summary=3)
        assert block.effective_token_count() == 3

    def test_is_pinned_by_user_weight(self):
        block = MemoryBlock(content="x", user_weight=1.0)
        assert block.is_pinned()

    def test_is_pinned_by_tag(self):
        block = MemoryBlock(content="x", tags=["pinned"])
        assert block.is_pinned()

    def test_is_not_pinned_by_default(self):
        block = MemoryBlock(content="x")
        assert not block.is_pinned()

    def test_touch_increments_frequency(self):
        block = MemoryBlock(content="x")
        original_time = block.last_accessed
        block.touch()
        assert block.frequency == 1
        block.touch()
        assert block.frequency == 2

    def test_is_not_expired_by_default(self):
        block = MemoryBlock(content="x")
        assert not block.is_expired()

    def test_is_expired_when_past_expiry(self):
        block = MemoryBlock(
            content="x",
            expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        assert block.is_expired()

    def test_is_not_expired_when_future(self):
        block = MemoryBlock(
            content="x",
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        assert not block.is_expired()

    def test_repr_contains_key_info(self):
        block = MemoryBlock(content="hello world", tier=MemoryTier.EPISODIC)
        r = repr(block)
        assert "episodic" in r
        assert "hello" in r

    def test_tags_serialisation_roundtrip(self):
        block = MemoryBlock(content="x", tags=["a", "b", "system"])
        serialised = block.tags_to_json()
        import json
        assert json.loads(serialised) == ["a", "b", "system"]

    def test_metadata_serialisation_roundtrip(self):
        block = MemoryBlock(content="x", metadata={"page": 3, "doc": "manual.pdf"})
        serialised = block.metadata_to_json()
        import json
        assert json.loads(serialised)["page"] == 3

    def test_embedding_bytes_roundtrip(self):
        embedding = [0.1, 0.2, -0.5, 1.0, 0.0]
        block = MemoryBlock(content="x", embedding=embedding)
        raw = block.embedding_to_bytes()
        recovered = MemoryBlock.embedding_from_bytes(raw)
        assert len(recovered) == len(embedding)
        for a, b in zip(recovered, embedding):
            assert abs(a - b) < 1e-5

    def test_empty_embedding_roundtrip(self):
        block = MemoryBlock(content="x", embedding=[])
        assert block.embedding_to_bytes() == b""
        assert MemoryBlock.embedding_from_bytes(b"") == []


# ── ContextWindow ─────────────────────────────────────────────────────────────

class TestContextWindow:
    def _make_block(self, content: str, tier: MemoryTier, tokens: int = 10) -> MemoryBlock:
        return MemoryBlock(content=content, tier=tier, token_count=tokens)

    def test_empty_window(self):
        cw = ContextWindow(blocks=[], token_budget=1000, tokens_used=0)
        assert cw.utilization() == 0.0
        assert cw.as_prompt().startswith("[CONTEXT WINDOW")

    def test_utilization(self):
        cw = ContextWindow(blocks=[], token_budget=1000, tokens_used=500)
        assert cw.utilization() == 0.5

    def test_blocks_by_tier(self):
        blocks = [
            self._make_block("w1", MemoryTier.WORKING),
            self._make_block("w2", MemoryTier.WORKING),
            self._make_block("e1", MemoryTier.EPISODIC),
        ]
        cw = ContextWindow(blocks=blocks, token_budget=1000, tokens_used=30)
        assert len(cw.blocks_by_tier(MemoryTier.WORKING)) == 2
        assert len(cw.blocks_by_tier(MemoryTier.EPISODIC)) == 1
        assert len(cw.blocks_by_tier(MemoryTier.SEMANTIC)) == 0

    def test_as_prompt_contains_sections(self):
        blocks = [
            self._make_block("sys", MemoryTier.WORKING, 5),
            self._make_block("recent", MemoryTier.EPISODIC, 10),
            self._make_block("fact", MemoryTier.SEMANTIC, 8),
        ]
        # Tag first block as system
        blocks[0].tags = ["system"]
        cw = ContextWindow(blocks=blocks, token_budget=1000, tokens_used=23)
        prompt = cw.as_prompt()
        assert "[SYSTEM INSTRUCTIONS]" in prompt
        assert "[RECENT CONTEXT]" in prompt
        assert "[RETRIEVED MEMORIES]" in prompt
        assert "sys" in prompt
        assert "recent" in prompt
        assert "fact" in prompt

    def test_token_usage_by_tier(self):
        blocks = [
            self._make_block("w", MemoryTier.WORKING, 100),
            self._make_block("e", MemoryTier.EPISODIC, 200),
        ]
        cw = ContextWindow(blocks=blocks, token_budget=1000, tokens_used=300)
        usage = cw.token_usage_by_tier()
        assert usage["working"] == 100
        assert usage["episodic"] == 200
        assert usage["semantic"] == 0

    def test_summary_repr(self):
        blocks = [self._make_block("x", MemoryTier.WORKING, 50)]
        cw = ContextWindow(blocks=blocks, token_budget=1000, tokens_used=50)
        s = cw.summary()
        assert "50/1000" in s
        assert "blocks=1" in s

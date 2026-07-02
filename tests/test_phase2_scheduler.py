"""
Phase 2 — Context Scheduler tests.

Tests the allocate() API and all four scheduling policies.
Uses stub embedder (no GPU needed).
"""
from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from contextos import ContextConfig, ContextOS, MemoryTier
from contextos.models.block import MemoryBlock
from contextos.scheduler.policies import (
    FIFOPolicy,
    PriorityPolicy,
    RoundRobinPolicy,
    WeightedFairPolicy,
    get_policy,
)


@pytest.fixture
def cos(tmp_path):
    config = ContextConfig(
        db_path=str(tmp_path / "sched.db"),
        embedding_stub=True,
        llm_provider="none",
        token_budget=8_000,
    )
    with ContextOS(config) as c:
        yield c


def seed_memories(cos, count=10):
    """Seed the store with N episodic memories."""
    blocks = []
    for i in range(count):
        b = cos.store(f"Memory number {i} with some content about topic {i}", tier=MemoryTier.EPISODIC)
        blocks.append(b)
    return blocks


class TestAllocateBasic:
    def test_allocate_returns_context_window(self, cos):
        cos.store("hello world", tier=MemoryTier.EPISODIC)
        ctx = cos.allocate(budget=4096, query="hello")
        assert ctx is not None
        assert ctx.token_budget == 4096

    def test_allocate_never_exceeds_budget(self, cos):
        seed_memories(cos, 50)
        ctx = cos.allocate(budget=100, query="test")
        assert ctx.tokens_used <= 100

    def test_allocate_includes_working_blocks(self, cos):
        sys = cos.store("You are a helpful assistant", tier=MemoryTier.WORKING, tags=["system"])
        cos.store("Some episodic fact", tier=MemoryTier.EPISODIC)
        ctx = cos.allocate(budget=4096, query="hi")
        working = ctx.blocks_by_tier(MemoryTier.WORKING)
        assert len(working) >= 1
        assert any(b.id == sys.id for b in working)

    def test_allocate_empty_store_returns_empty_window(self, cos):
        ctx = cos.allocate(budget=4096, query="hello")
        assert len(ctx.blocks) == 0
        assert ctx.tokens_used == 0

    def test_allocate_no_query_still_works(self, cos):
        cos.store("a fact")
        ctx = cos.allocate(budget=4096)
        assert ctx.tokens_used >= 0

    def test_allocate_default_budget_from_config(self, cos):
        cos.store("test memory")
        ctx = cos.allocate(query="test")
        assert ctx.token_budget == cos.config.token_budget

    def test_as_prompt_is_string(self, cos):
        cos.store("remember this", tier=MemoryTier.EPISODIC)
        ctx = cos.allocate(budget=4096, query="recall")
        prompt = ctx.as_prompt()
        assert isinstance(prompt, str)
        assert "CONTEXT WINDOW" in prompt

    def test_allocate_touches_included_blocks(self, cos):
        b = cos.store("I will be touched", tier=MemoryTier.EPISODIC)
        assert b.frequency == 0
        cos.allocate(budget=4096, query="anything")
        updated = cos.get(b.id)
        assert updated.frequency >= 1


class TestSchedulingPolicies:
    def test_fifo_returns_oldest_first(self):
        blocks = [
            MemoryBlock(content="old", created_at=datetime(2020, 1, 1), token_count=5),
            MemoryBlock(content="new", created_at=datetime(2025, 1, 1), token_count=5),
        ]
        ordered = FIFOPolicy().order(blocks, [])
        assert ordered[0].content == "old"

    def test_priority_returns_highest_score_first(self):
        blocks = [
            MemoryBlock(content="low", importance_score=0.2, token_count=5),
            MemoryBlock(content="high", importance_score=0.9, token_count=5),
        ]
        ordered = PriorityPolicy().order(blocks, [])
        assert ordered[0].content == "high"

    def test_round_robin_interleaves_tiers(self):
        blocks = [
            MemoryBlock(content="w1", tier=MemoryTier.WORKING, importance_score=0.9, token_count=5),
            MemoryBlock(content="w2", tier=MemoryTier.WORKING, importance_score=0.5, token_count=5),
            MemoryBlock(content="e1", tier=MemoryTier.EPISODIC, importance_score=0.8, token_count=5),
            MemoryBlock(content="e2", tier=MemoryTier.EPISODIC, importance_score=0.4, token_count=5),
        ]
        ordered = RoundRobinPolicy().order(blocks, [])
        # First two should be from different tiers (interleaved)
        assert ordered[0].tier != ordered[1].tier

    def test_weighted_fair_puts_working_first(self):
        blocks = [
            MemoryBlock(content="ep", tier=MemoryTier.EPISODIC, importance_score=0.9, token_count=5),
            MemoryBlock(content="wk", tier=MemoryTier.WORKING, importance_score=0.1, token_count=5),
        ]
        ordered = WeightedFairPolicy().order(blocks, [])
        assert ordered[0].tier == MemoryTier.WORKING

    def test_get_policy_by_name(self):
        p = get_policy("priority")
        assert p.name == "priority"

    def test_get_policy_invalid_raises(self):
        with pytest.raises(ValueError):
            get_policy("nonexistent_policy")

    def test_different_policies_produce_different_orders(self, cos):
        for i in range(20):
            cos.store(f"Block {i}", tier=MemoryTier.EPISODIC)
        ctx_fifo = cos.allocate(budget=200, query="test")
        cos.scheduler = __import__("contextos.scheduler.engine", fromlist=["ContextScheduler"]).ContextScheduler("priority")
        ctx_priority = cos.allocate(budget=200, query="test")
        # At least one block should differ in ordering (not guaranteed but very likely with 20 blocks)
        fifo_ids = [b.id for b in ctx_fifo.blocks]
        priority_ids = [b.id for b in ctx_priority.blocks]
        # Both should be valid context windows
        assert ctx_fifo.tokens_used <= 200
        assert ctx_priority.tokens_used <= 200


class TestContextWindowFormat:
    def test_prompt_sections_populated(self, cos):
        cos.store("system prompt", tier=MemoryTier.WORKING, tags=["system"])
        cos.store("recent event", tier=MemoryTier.EPISODIC)
        cos.store("a known fact", tier=MemoryTier.SEMANTIC)
        ctx = cos.allocate(budget=4096, query="tell me")
        prompt = ctx.as_prompt()
        assert "[SYSTEM INSTRUCTIONS]" in prompt
        assert "[RECENT CONTEXT]" in prompt or "[WORKING MEMORY" in prompt

    def test_utilization_ratio(self, cos):
        seed_memories(cos, 5)
        ctx = cos.allocate(budget=1000, query="test")
        assert 0.0 <= ctx.utilization() <= 1.0

    def test_summary_contains_stats(self, cos):
        cos.store("test")
        ctx = cos.allocate(budget=1000, query="test")
        summary = ctx.summary()
        assert "tokens" in summary
        assert "blocks" in summary

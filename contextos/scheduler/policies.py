"""
ContextOS — built-in scheduling policies.

FIFO:          Oldest memories first (by created_at).
Priority:      Highest importance_score first. Pure greedy.
RoundRobin:    Interleaves one block from each tier in rotation.
WeightedFair:  Allocates token budget proportionally across tiers,
               then fills each slice with highest-importance blocks.
"""
from __future__ import annotations

from contextos.models.block import MemoryBlock
from contextos.models.tier import MemoryTier
from contextos.scheduler.base import SchedulingPolicy


class FIFOPolicy(SchedulingPolicy):
    """Oldest memories first (by created_at)."""

    name = "fifo"

    def order(
        self,
        candidates: list[MemoryBlock],
        query_embedding: list[float],
    ) -> list[MemoryBlock]:
        return sorted(candidates, key=lambda b: b.created_at)


class PriorityPolicy(SchedulingPolicy):
    """Highest importance_score first. Pure greedy."""

    name = "priority"

    def order(
        self,
        candidates: list[MemoryBlock],
        query_embedding: list[float],
    ) -> list[MemoryBlock]:
        return sorted(candidates, key=lambda b: b.importance_score, reverse=True)


class RoundRobinPolicy(SchedulingPolicy):
    """
    Interleave one block from each tier in rotation.

    Order within each tier is by importance_score (highest first).
    Ensures no tier starves, even if one tier has far more blocks.
    """

    name = "round_robin"

    def order(
        self,
        candidates: list[MemoryBlock],
        query_embedding: list[float],
    ) -> list[MemoryBlock]:
        # Group by tier, each group sorted by importance desc
        tier_queues: dict[MemoryTier, list[MemoryBlock]] = {}
        for tier in MemoryTier.ordered():
            tier_blocks = [b for b in candidates if b.tier == tier]
            tier_blocks.sort(key=lambda b: b.importance_score, reverse=True)
            if tier_blocks:
                tier_queues[tier] = tier_blocks

        result: list[MemoryBlock] = []
        while tier_queues:
            empty_tiers: list[MemoryTier] = []
            for tier, blocks in tier_queues.items():
                result.append(blocks.pop(0))
                if not blocks:
                    empty_tiers.append(tier)
            for t in empty_tiers:
                del tier_queues[t]

        return result


class WeightedFairPolicy(SchedulingPolicy):
    """
    Allocate token budget proportionally across tiers, then fill each
    tier's slice with its highest-importance blocks.

    Default weights:
        WORKING  = 0.20
        EPISODIC = 0.45
        SEMANTIC = 0.25
        ARCHIVED = 0.10

    If a tier underuses its budget, the leftover rolls to the next tier.
    """

    name = "weighted_fair"

    TIER_WEIGHTS: dict[MemoryTier, float] = {
        MemoryTier.WORKING: 0.20,
        MemoryTier.EPISODIC: 0.45,
        MemoryTier.SEMANTIC: 0.25,
        MemoryTier.ARCHIVED: 0.10,
    }

    def __init__(self, tier_weights: dict[MemoryTier, float] | None = None):
        if tier_weights:
            self.TIER_WEIGHTS = tier_weights

    def order(
        self,
        candidates: list[MemoryBlock],
        query_embedding: list[float],
    ) -> list[MemoryBlock]:
        # Group by tier, sorted by importance within each
        tier_blocks: dict[MemoryTier, list[MemoryBlock]] = {}
        for tier in MemoryTier.ordered():
            blocks = sorted(
                [b for b in candidates if b.tier == tier],
                key=lambda b: b.importance_score,
                reverse=True,
            )
            tier_blocks[tier] = blocks

        # Build the result by iterating tiers in priority order
        # (WORKING first, then EPISODIC, SEMANTIC, ARCHIVED)
        result: list[MemoryBlock] = []
        for tier in MemoryTier.ordered():
            result.extend(tier_blocks.get(tier, []))

        return result


# Registry for policy lookup by name
POLICY_REGISTRY: dict[str, type[SchedulingPolicy]] = {
    "fifo": FIFOPolicy,
    "priority": PriorityPolicy,
    "round_robin": RoundRobinPolicy,
    "weighted_fair": WeightedFairPolicy,
}


def get_policy(name: str) -> SchedulingPolicy:
    """Instantiate a scheduling policy by name string."""
    cls = POLICY_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(POLICY_REGISTRY.keys())
        raise ValueError(f"Unknown scheduling policy '{name}'. Available: {available}")
    return cls()

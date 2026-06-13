"""
ContextOS — EvictionEngine: manages when and how eviction happens.

Sits between the MemoryManager and the eviction policies.
When a tier exceeds its token budget, it selects victims using
the configured policy and demotes/deletes them.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID

from contextos.eviction.base import EvictionPolicy
from contextos.eviction.policies import get_eviction_policy
from contextos.models.block import MemoryBlock
from contextos.models.tier import MemoryTier

if TYPE_CHECKING:
    from contextos.memory.store import MemoryStore

logger = logging.getLogger(__name__)


class EvictionEngine:
    """
    Manages eviction across memory tiers.

    Usage:
        engine = EvictionEngine(store, policy="lru")
        evicted = engine.evict_tier(MemoryTier.EPISODIC, budget=8000)
    """

    def __init__(self, store: "MemoryStore", policy: EvictionPolicy | str = "hybrid"):
        self.store = store
        if isinstance(policy, str):
            self.policy = get_eviction_policy(policy)
        else:
            self.policy = policy

    def evict_tier(
        self,
        tier: MemoryTier,
        budget: int,
        protected_ids: set[UUID] | None = None,
    ) -> list[MemoryBlock]:
        """
        If the tier exceeds `budget` tokens, evict blocks until under budget.

        Args:
            tier:          Which tier to check and evict from.
            budget:        Max allowed tokens in this tier.
            protected_ids: Block IDs that must NOT be evicted (e.g., in a live context window).

        Returns:
            List of evicted MemoryBlocks.

        Eviction behavior:
            - Pinned blocks are always excluded from eviction candidates.
            - Protected IDs (blocks in an active context window) are excluded.
            - For WORKING tier: no eviction ever happens (returns []).
            - For EPISODIC/SEMANTIC: evicted blocks are demoted to ARCHIVED.
            - For ARCHIVED: evicted blocks are permanently deleted.
        """
        if tier == MemoryTier.WORKING:
            return []  # never evict WORKING

        protected_ids = protected_ids or set()
        current_usage = self.store.token_usage(tier)

        if current_usage <= budget:
            return []

        tokens_to_free = current_usage - budget

        # Get all blocks in this tier, filter out pinned and protected
        all_blocks = self.store.get_by_tier(tier)
        candidates = [
            b for b in all_blocks
            if not b.is_pinned() and b.id not in protected_ids
        ]

        if not candidates:
            logger.warning(
                "Tier %s exceeds budget by %d tokens but no evictable candidates",
                tier.value, tokens_to_free,
            )
            return []

        # Ask the policy to pick victims
        victims = self.policy.select_victims(candidates, tokens_to_free)

        evicted: list[MemoryBlock] = []
        for victim in victims:
            if tier == MemoryTier.ARCHIVED:
                # ARCHIVED → delete permanently
                self.store.delete(victim.id)
                logger.debug("Deleted archived block %s (%d tokens)", victim.id, victim.effective_token_count())
            else:
                # EPISODIC/SEMANTIC → demote to ARCHIVED
                self.store.update_tier(victim.id, MemoryTier.ARCHIVED)
                logger.debug(
                    "Evicted %s: %s → archived (%d tokens, policy=%s)",
                    victim.id, tier.value, victim.effective_token_count(), self.policy.name,
                )
            evicted.append(victim)

        total_freed = sum(b.effective_token_count() for b in evicted)
        logger.info(
            "Evicted %d blocks from %s tier, freed %d tokens (policy=%s)",
            len(evicted), tier.value, total_freed, self.policy.name,
        )
        return evicted

    def evict_all_tiers(
        self,
        tier_budgets: dict[MemoryTier, int],
        protected_ids: set[UUID] | None = None,
    ) -> dict[str, list[MemoryBlock]]:
        """
        Run eviction across all tiers given their budgets.

        Args:
            tier_budgets: {MemoryTier.EPISODIC: 8000, MemoryTier.SEMANTIC: 32000, ...}
            protected_ids: Block IDs protected from eviction.

        Returns:
            {"episodic": [...evicted...], "semantic": [...], ...}
        """
        results: dict[str, list[MemoryBlock]] = {}
        for tier, budget in tier_budgets.items():
            evicted = self.evict_tier(tier, budget, protected_ids)
            if evicted:
                results[tier.value] = evicted
        return results

"""
ContextOS — ContextScheduler: the core scheduling engine.

Takes a token budget + query, gathers candidates from the memory manager,
recomputes importance scores, delegates ordering to a SchedulingPolicy,
and packs blocks greedily into a ContextWindow.

This is the implementation behind `ContextOS.allocate()`.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from contextos.models.block import MemoryBlock
from contextos.models.context import ContextWindow
from contextos.models.tier import MemoryTier
from contextos.scheduler.base import SchedulingPolicy
from contextos.scheduler.policies import get_policy

if TYPE_CHECKING:
    from contextos.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


class ContextScheduler:
    """
    The core scheduling engine.

    Usage:
        scheduler = ContextScheduler(policy="weighted_fair")
        window = scheduler.schedule(memory_manager, budget=16_000, query_embedding=[...])
    """

    def __init__(self, policy: SchedulingPolicy | str = "weighted_fair"):
        if isinstance(policy, str):
            self.policy = get_policy(policy)
        else:
            self.policy = policy

    def schedule(
        self,
        memory_manager: "MemoryManager",
        budget: int,
        query_embedding: list[float],
        query: str = "",
    ) -> ContextWindow:
        """
        Build a context window that fits within the token budget.

        Steps:
            1. Always include all WORKING tier blocks (system prompt, current turn).
            2. Gather candidates from EPISODIC + SEMANTIC tiers.
            3. Recompute importance scores against the current query.
            4. Delegate ordering to the scheduling policy.
            5. Greedily pack blocks until budget is exhausted.
            6. Touch every included block (update access tracking).

        Args:
            memory_manager:  The MemoryManager to pull blocks from.
            budget:          Max tokens allowed in the context window.
            query_embedding: Embedding of the current user query.
            query:           Raw text of the query (stored in ContextWindow).

        Returns:
            A ContextWindow with ordered blocks fitting within the budget.
        """
        # ── Step 1: Always include WORKING blocks ─────────────────────────
        working_blocks = memory_manager.get_tier(MemoryTier.WORKING)
        included: list[MemoryBlock] = []
        tokens_used = 0

        for block in working_blocks:
            cost = block.effective_token_count()
            if tokens_used + cost <= budget:
                included.append(block)
                tokens_used += cost
            else:
                logger.warning(
                    "WORKING block %s (%d tokens) exceeds remaining budget, skipping",
                    block.id, cost,
                )

        remaining_budget = budget - tokens_used

        # ── Step 2: Gather candidates from other tiers ────────────────────
        candidates: list[MemoryBlock] = []
        for tier in [MemoryTier.EPISODIC, MemoryTier.SEMANTIC]:
            candidates.extend(memory_manager.get_tier(tier))

        if not candidates:
            return ContextWindow(
                blocks=included,
                token_budget=budget,
                tokens_used=tokens_used,
                query=query,
            )

        # ── Step 3: Recompute importance scores ───────────────────────────
        memory_manager.recompute_scores(candidates, query_embedding)

        # ── Step 4: Delegate ordering to the policy ───────────────────────
        ordered = self.policy.order(candidates, query_embedding)

        # ── Step 5: Greedy packing ────────────────────────────────────────
        for block in ordered:
            cost = block.effective_token_count()
            if cost == 0:
                continue
            if tokens_used + cost > budget:
                continue  # skip this block, try smaller ones
            included.append(block)
            tokens_used += cost

        # ── Step 6: Touch included blocks ─────────────────────────────────
        for block in included:
            memory_manager.touch(block.id)

        window = ContextWindow(
            blocks=included,
            token_budget=budget,
            tokens_used=tokens_used,
            query=query,
        )

        logger.info(
            "Scheduled context: %s (policy=%s)",
            window.summary(),
            self.policy.name,
        )

        return window

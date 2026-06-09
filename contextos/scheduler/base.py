"""
ContextOS — SchedulingPolicy ABC.

Every scheduling algorithm implements this interface. The scheduler
calls `order()` to get a priority-sorted list of candidates, then
greedily packs them into the token budget.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from contextos.models.block import MemoryBlock


class SchedulingPolicy(ABC):
    """
    Abstract base class for context scheduling algorithms.

    Implementations must define `order()`, which takes scored candidates
    and returns them in the order they should be packed into the context window.

    The ContextScheduler then packs greedily from the front of this list
    until the token budget is exhausted.
    """

    name: str = "base"

    @abstractmethod
    def order(
        self,
        candidates: list[MemoryBlock],
        query_embedding: list[float],
    ) -> list[MemoryBlock]:
        """
        Return candidates in priority order (highest priority first).

        Args:
            candidates:      Scored MemoryBlocks from all eligible tiers.
            query_embedding: Embedding of the current user query.

        Returns:
            The same blocks, reordered by this policy's strategy.
        """
        ...

"""
ContextOS — EvictionPolicy ABC.

Every eviction algorithm implements this interface.
Given a list of blocks and a token count to free, it returns the
victims (blocks to evict) in order.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from contextos.models.block import MemoryBlock


class EvictionPolicy(ABC):
    """Abstract base class for eviction policies."""

    name: str = "base"

    @abstractmethod
    def select_victims(
        self,
        blocks: list[MemoryBlock],
        tokens_to_free: int,
    ) -> list[MemoryBlock]:
        """
        Select blocks to evict to free at least `tokens_to_free` tokens.

        Args:
            blocks:         Candidate blocks (already filtered to exclude pinned).
            tokens_to_free: Minimum tokens to reclaim.

        Returns:
            List of blocks to evict, in eviction order.
            Total effective_token_count of returned blocks >= tokens_to_free.
        """
        ...

"""
ContextOS — core data models.
"""
from enum import Enum


class MemoryTier(str, Enum):
    """
    Five-tier memory hierarchy mirroring CPU memory architecture.

    WORKING   → CPU registers:  current turn, always in context, never evicted.
    EPISODIC  → L2 cache / RAM: recent conversation, included when budget allows.
    SEMANTIC  → SSD:            compressed facts, retrieved on demand via vector search.
    ARCHIVED  → HDD:            cold storage, only in vector DB, paged in on retrieval.
    """

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    ARCHIVED = "archived"

    # Ordered by "hotness" — lower index = hotter
    @staticmethod
    def ordered() -> list["MemoryTier"]:
        return [
            MemoryTier.WORKING,
            MemoryTier.EPISODIC,
            MemoryTier.SEMANTIC,
            MemoryTier.ARCHIVED,
        ]

    def is_hotter_than(self, other: "MemoryTier") -> bool:
        order = MemoryTier.ordered()
        return order.index(self) < order.index(other)

    def is_colder_than(self, other: "MemoryTier") -> bool:
        order = MemoryTier.ordered()
        return order.index(self) > order.index(other)

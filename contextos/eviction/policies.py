"""
ContextOS — built-in eviction policies.

FIFO:     Evict oldest (by created_at).
LRU:      Evict least recently used (by last_accessed).
LFU:      Evict least frequently used (by frequency count).
Priority: Evict lowest importance_score.
Hybrid:   Weighted combination of recency + frequency + importance.
"""
from __future__ import annotations

from contextos.eviction.base import EvictionPolicy
from contextos.models.block import MemoryBlock


def _collect_victims(
    sorted_blocks: list[MemoryBlock], tokens_to_free: int
) -> list[MemoryBlock]:
    """Helper: walk a sorted list and collect blocks until enough tokens freed."""
    victims: list[MemoryBlock] = []
    freed = 0
    for block in sorted_blocks:
        if freed >= tokens_to_free:
            break
        victims.append(block)
        freed += block.effective_token_count()
    return victims


class FIFOEviction(EvictionPolicy):
    """Evict oldest blocks first (by created_at)."""
    name = "fifo"

    def select_victims(self, blocks: list[MemoryBlock], tokens_to_free: int) -> list[MemoryBlock]:
        sorted_blocks = sorted(blocks, key=lambda b: b.created_at)
        return _collect_victims(sorted_blocks, tokens_to_free)


class LRUEviction(EvictionPolicy):
    """Evict least recently used first (by last_accessed)."""
    name = "lru"

    def select_victims(self, blocks: list[MemoryBlock], tokens_to_free: int) -> list[MemoryBlock]:
        sorted_blocks = sorted(blocks, key=lambda b: b.last_accessed)
        return _collect_victims(sorted_blocks, tokens_to_free)


class LFUEviction(EvictionPolicy):
    """Evict least frequently used first (by frequency count)."""
    name = "lfu"

    def select_victims(self, blocks: list[MemoryBlock], tokens_to_free: int) -> list[MemoryBlock]:
        sorted_blocks = sorted(blocks, key=lambda b: b.frequency)
        return _collect_victims(sorted_blocks, tokens_to_free)


class PriorityEviction(EvictionPolicy):
    """Evict lowest importance_score first."""
    name = "priority"

    def select_victims(self, blocks: list[MemoryBlock], tokens_to_free: int) -> list[MemoryBlock]:
        sorted_blocks = sorted(blocks, key=lambda b: b.importance_score)
        return _collect_victims(sorted_blocks, tokens_to_free)


class HybridEviction(EvictionPolicy):
    """
    Weighted combination score:
        evict_score = 0.5 * (1 - recency_normalized)
                    + 0.3 * (1 - frequency_normalized)
                    + 0.2 * (1 - importance_normalized)

    Higher evict_score = more likely to be evicted (less worth keeping).
    """
    name = "hybrid"

    def select_victims(self, blocks: list[MemoryBlock], tokens_to_free: int) -> list[MemoryBlock]:
        if not blocks:
            return []

        max_freq = max(b.frequency for b in blocks) or 1

        def evict_score(b: MemoryBlock) -> float:
            # Invert: high recency = keep, so (1 - recency) = evict
            r = 1.0 - b.recency_score
            f = 1.0 - (b.frequency / max_freq) if max_freq > 0 else 1.0
            i = 1.0 - b.importance_score
            return 0.5 * r + 0.3 * f + 0.2 * i

        sorted_blocks = sorted(blocks, key=evict_score, reverse=True)
        return _collect_victims(sorted_blocks, tokens_to_free)


# Registry for policy lookup by name
EVICTION_REGISTRY: dict[str, type[EvictionPolicy]] = {
    "fifo": FIFOEviction,
    "lru": LRUEviction,
    "lfu": LFUEviction,
    "priority": PriorityEviction,
    "hybrid": HybridEviction,
}


def get_eviction_policy(name: str) -> EvictionPolicy:
    """Instantiate an eviction policy by name string."""
    cls = EVICTION_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(EVICTION_REGISTRY.keys())
        raise ValueError(f"Unknown eviction policy '{name}'. Available: {available}")
    return cls()

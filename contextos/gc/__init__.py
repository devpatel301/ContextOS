"""
ContextOS — MemoryGarbageCollector.

OS-inspired garbage collection for the memory store:
  Pass 1: Expiry      — remove blocks past their expires_at
  Pass 2: Dedup       — remove near-duplicate blocks (cosine similarity > threshold)
  Pass 3: Staleness   — remove very old, very low-score ARCHIVED blocks

Each pass is idempotent: running GC twice produces the same result as running it once.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from contextos.models.tier import MemoryTier
from contextos.utils.embedder import Embedder

if TYPE_CHECKING:
    from contextos.memory.store import MemoryStore

logger = logging.getLogger(__name__)


class GarbageCollector:
    """
    Memory garbage collector.

    Usage:
        gc = GarbageCollector(store, dedup_threshold=0.92)
        report = gc.run()
        print(report)
        # {'expired': 3, 'deduplicated': 2, 'pruned': 1, 'total_freed': 6}
    """

    def __init__(
        self,
        store: "MemoryStore",
        dedup_threshold: float = 0.92,
        prune_min_age_days: int = 30,
        prune_max_importance: float = 0.1,
    ):
        self.store = store
        self.dedup_threshold = dedup_threshold
        self.prune_min_age_days = prune_min_age_days
        self.prune_max_importance = prune_max_importance

    def run(self) -> dict[str, int]:
        """
        Execute all GC passes. Returns counts of blocks removed.

        Safe to call at any time. Idempotent.
        """
        report: dict[str, int] = {
            "expired": 0,
            "deduplicated": 0,
            "pruned": 0,
            "total_freed": 0,
        }

        report["expired"] = self._pass_expiry()
        report["deduplicated"] = self._pass_dedup()
        report["pruned"] = self._pass_staleness()
        report["total_freed"] = report["expired"] + report["deduplicated"] + report["pruned"]

        if report["total_freed"] > 0:
            logger.info("GC completed: %s", report)
        else:
            logger.debug("GC completed: nothing to collect")

        return report

    def _pass_expiry(self) -> int:
        """Pass 1: Remove blocks whose expires_at has passed."""
        expired = self.store.get_expired()
        if not expired:
            return 0

        ids = [b.id for b in expired]
        count = self.store.delete_many(ids)
        logger.debug("GC expiry: removed %d expired blocks", count)
        return count

    def _pass_dedup(self) -> int:
        """
        Pass 2: Remove near-duplicate blocks.

        For each pair (A, B) where cosine_similarity > threshold:
          - Keep the one in the hotter tier (or the newer one if same tier)
          - Delete the other

        Uses O(n²) pairwise comparison — fine for stores up to ~5000 blocks.
        For larger stores, use approximate nearest neighbor in the future.
        """
        all_blocks = self.store.get_all()
        if len(all_blocks) < 2:
            return 0

        # Pre-filter: only blocks with embeddings
        blocks = [b for b in all_blocks if b.embedding]
        deleted_ids = set()
        removed = 0

        for i in range(len(blocks)):
            if blocks[i].id in deleted_ids:
                continue
            for j in range(i + 1, len(blocks)):
                if blocks[j].id in deleted_ids:
                    continue

                sim = Embedder.cosine_similarity(blocks[i].embedding, blocks[j].embedding)
                if sim >= self.dedup_threshold:
                    # Keep the "better" block
                    a, b = blocks[i], blocks[j]
                    # Prefer hotter tier; if same tier, prefer higher importance; if tied, newer
                    if a.tier.is_hotter_than(b.tier):
                        victim = b
                    elif b.tier.is_hotter_than(a.tier):
                        victim = a
                    elif a.importance_score >= b.importance_score:
                        victim = b
                    else:
                        victim = a

                    # Don't delete pinned blocks
                    if victim.is_pinned():
                        continue

                    self.store.delete(victim.id)
                    deleted_ids.add(victim.id)
                    removed += 1
                    logger.debug(
                        "GC dedup: removed %s (sim=%.3f with %s)",
                        victim.id, sim, (a.id if victim == b else b.id),
                    )

        return removed

    def _pass_staleness(self) -> int:
        """
        Pass 3: Remove very old, very low-score ARCHIVED blocks.

        Blocks must be:
          - In ARCHIVED tier
          - Older than prune_min_age_days
          - importance_score < prune_max_importance
          - Not pinned
        """
        all_archived = self.store.get_by_tier(MemoryTier.ARCHIVED)
        cutoff = datetime.utcnow() - timedelta(days=self.prune_min_age_days)

        to_delete = [
            b for b in all_archived
            if not b.is_pinned()
            and b.created_at < cutoff
            and b.importance_score < self.prune_max_importance
        ]

        if not to_delete:
            return 0

        count = self.store.delete_many([b.id for b in to_delete])
        logger.debug("GC staleness: pruned %d old archived blocks", count)
        return count

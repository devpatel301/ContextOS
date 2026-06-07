"""
ContextOS — MemoryManager: tier controller and lifecycle engine.

Responsible for:
- Storing new memories with embeddings + token counts
- Triggering tier promotions and demotions based on access patterns
- Applying time decay to recency scores
- Exposing tier status (token usage, block counts)
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from uuid import UUID

from contextos.config import ContextConfig
from contextos.memory.scoring import recompute_scores_for_blocks
from contextos.memory.store import MemoryStore
from contextos.models.block import MemoryBlock
from contextos.models.tier import MemoryTier
from contextos.utils.embedder import Embedder
from contextos.utils.tokens import count_tokens

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Controls the full memory hierarchy.

    Typical usage:

        mm = MemoryManager(config)
        block = mm.store("User prefers Python over JavaScript", tier=MemoryTier.EPISODIC)
        mm.touch(block.id)               # record access
        mm.run_tier_transitions()        # promote/demote based on rules
        blocks = mm.get_tier(MemoryTier.EPISODIC)
    """

    def __init__(self, config: ContextConfig, store: MemoryStore, embedder: Embedder):
        self.config = config
        self._db = store          # MemoryStore instance — named _db to avoid shadowing store()
        self.embedder = embedder

    # ── Write ─────────────────────────────────────────────────────────────────

    def store(
        self,
        content: str,
        tier: MemoryTier = MemoryTier.EPISODIC,
        source: str = "user",
        tags: list[str] | None = None,
        user_weight: float = 0.5,
        expires_in_hours: float | None = None,
        metadata: dict | None = None,
    ) -> MemoryBlock:
        """
        Create and persist a new MemoryBlock.

        Automatically:
          - Counts tokens (tiktoken)
          - Generates embedding (sentence-transformers or stub)
          - Computes initial importance score (recency=1.0, similarity=0 until queried)

        Args:
            content:          The text to remember.
            tier:             Initial tier. Defaults to EPISODIC.
            source:           Where this memory came from ('user', 'tool', 'llm', etc).
            tags:             Labels for filtering/invalidation.
            user_weight:      0.0–1.0 importance hint. Use 1.0 to pin (never evict).
            expires_in_hours: If set, block expires after this many hours.
            metadata:         Arbitrary extra data.

        Returns:
            The created MemoryBlock (also persisted to SQLite).
        """
        tags = tags or []
        metadata = metadata or {}

        now = datetime.utcnow()
        expires_at = (now + timedelta(hours=expires_in_hours)) if expires_in_hours else None

        embedding = self.embedder.embed(content)
        token_count = count_tokens(content, model=self.config.llm_model)

        block = MemoryBlock(
            content=content,
            embedding=embedding,
            tier=tier,
            source=source,
            tags=tags,
            user_weight=user_weight,
            token_count=token_count,
            created_at=now,
            last_accessed=now,
            expires_at=expires_at,
            metadata=metadata,
            # Initial scores: no query context yet
            importance_score=0.3 * 1.0 + 0.1 * user_weight,  # recency=1 + user hint
            recency_score=1.0,
            frequency=0,
        )

        self._db.save(block)
        logger.debug("Stored %s in %s tier", block.id, tier.value)
        return block

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, block_id: UUID) -> MemoryBlock | None:
        return self._db.get(block_id)

    def get_tier(self, tier: MemoryTier) -> list[MemoryBlock]:
        return self._db.get_by_tier(tier)

    def get_all(self) -> list[MemoryBlock]:
        return self._db.get_all()

    # ── Access tracking ───────────────────────────────────────────────────────

    def touch(self, block_id: UUID) -> MemoryBlock | None:
        """
        Record that a block was used (increment frequency, update last_accessed).
        Returns the updated block, or None if not found.
        """
        block = self._db.get(block_id)
        if block is None:
            return None
        block.touch()
        self._db.update_access(block.id, block.last_accessed, block.frequency)
        return block

    # ── Tier transitions ──────────────────────────────────────────────────────

    def promote(self, block_id: UUID, to_tier: MemoryTier) -> bool:
        """
        Move a block to a hotter tier.
        Returns True if the transition happened, False if block not found or
        target is already as hot or hotter.
        """
        block = self._db.get(block_id)
        if block is None:
            return False
        if not to_tier.is_hotter_than(block.tier):
            return False
        self._db.update_tier(block_id, to_tier)
        logger.debug("Promoted %s: %s → %s", block_id, block.tier.value, to_tier.value)
        return True

    def demote(self, block_id: UUID, to_tier: MemoryTier) -> bool:
        """
        Move a block to a colder tier.
        Pinned blocks are never demoted.
        """
        block = self._db.get(block_id)
        if block is None:
            return False
        if block.is_pinned():
            logger.debug("Skipping demotion of pinned block %s", block_id)
            return False
        if not to_tier.is_colder_than(block.tier):
            return False
        self._db.update_tier(block_id, to_tier)
        logger.debug("Demoted %s: %s → %s", block_id, block.tier.value, to_tier.value)
        return True

    def run_tier_transitions(self) -> dict[str, int]:
        """
        Apply all automatic tier transition rules to the full store.

        Rules (from FEATURES.md):
          1. EPISODIC → SEMANTIC if frequency >= threshold
          2. SEMANTIC → ARCHIVED if idle for N hours
          3. Any tier → ARCHIVED if age > N days (except WORKING and pinned)

        Returns a dict with counts of transitions made:
            {"episodic_to_semantic": 3, "semantic_to_archived": 1, ...}
        """
        cfg = self.config
        counts: dict[str, int] = {
            "episodic_to_semantic": 0,
            "semantic_to_archived": 0,
            "aged_to_archived": 0,
        }

        # Rule 1: EPISODIC → SEMANTIC on high access count
        for block in self._db.get_by_tier(MemoryTier.EPISODIC):
            if block.frequency >= cfg.episodic_to_semantic_access_count:
                # EPISODIC → SEMANTIC is "colder" in tier order, so update directly
                self._db.update_tier(block.id, MemoryTier.SEMANTIC)
                counts["episodic_to_semantic"] += 1

        # Rule 2: SEMANTIC → ARCHIVED on idle
        idle_semantic = self._db.get_idle_since(
            hours=cfg.semantic_to_archived_idle_hours, tier=MemoryTier.SEMANTIC
        )
        for block in idle_semantic:
            if not block.is_pinned():
                self._db.update_tier(block.id, MemoryTier.ARCHIVED)
                counts["semantic_to_archived"] += 1

        # Rule 3: Age-based archiving (skip WORKING and pinned)
        age_cutoff = datetime.utcnow() - timedelta(days=cfg.archived_age_days)
        for block in self._db.get_all():
            if block.tier == MemoryTier.WORKING:
                continue
            if block.is_pinned():
                continue
            if block.created_at < age_cutoff and block.tier != MemoryTier.ARCHIVED:
                self._db.update_tier(block.id, MemoryTier.ARCHIVED)
                counts["aged_to_archived"] += 1

        total = sum(counts.values())
        if total > 0:
            logger.info("Tier transitions: %s", counts)
        return counts

    # ── Score management ──────────────────────────────────────────────────────

    def recompute_scores(
        self, blocks: list[MemoryBlock], query_embedding: list[float]
    ) -> None:
        """
        Recompute importance and recency scores for a list of blocks given
        the current query embedding. Mutates blocks in-place and flushes
        updated scores to the store.
        """
        cfg = self.config
        recompute_scores_for_blocks(
            blocks,
            query_embedding,
            half_life_hours=cfg.recency_half_life_hours,
            w_similarity=cfg.score_weight_similarity,
            w_recency=cfg.score_weight_recency,
            w_frequency=cfg.score_weight_frequency,
            w_user=cfg.score_weight_user,
        )
        self._db.flush_tier_scores(blocks)

    def apply_time_decay(self) -> None:
        """
        Apply recency decay to all blocks without a full query context.
        Call this periodically (e.g., every 15 minutes) to keep scores current.
        """
        all_blocks = self._db.get_all()
        self.recompute_scores(all_blocks, query_embedding=[])
        logger.debug("Applied time decay to %d blocks", len(all_blocks))

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """
        Return a snapshot of current memory usage across all tiers.

        Example output:
            {
                "total_blocks": 142,
                "tiers": {
                    "working":  {"blocks": 3,  "tokens": 1200},
                    "episodic": {"blocks": 28, "tokens": 9400},
                    "semantic": {"blocks": 80, "tokens": 18200},
                    "archived": {"blocks": 31, "tokens": 7100},
                }
            }
        """
        tiers: dict[str, dict] = {}
        total = 0
        for tier in MemoryTier:
            count = self._db.count(tier)
            tokens = self._db.token_usage(tier)
            tiers[tier.value] = {"blocks": count, "tokens": tokens}
            total += count
        return {"total_blocks": total, "tiers": tiers}

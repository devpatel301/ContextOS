"""
ContextOS — top-level facade.

This is the single class users interact with. It wires together:
  - ContextConfig
  - MemoryStore (SQLite)
  - Embedder (sentence-transformers / stub)
  - MemoryManager (tier controller)
  - ContextScheduler (Phase 2)
  - EvictionEngine + GarbageCollector (Phase 3)
  - EvictionEngine + GarbageCollector (Phase 3)
  - SemanticCache     (Phase 4)
  - RetrieverEngine   (Phase 5)

Later phases will add:
  - ContextProfiler   (Phase 6)
"""
from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

from contextos.cache.engine import SemanticCache
from contextos.cache.store import CacheStore
from contextos.config import ContextConfig
from contextos.eviction.engine import EvictionEngine
from contextos.gc import GarbageCollector
from contextos.memory.manager import MemoryManager
from contextos.memory.store import MemoryStore
from contextos.models.block import MemoryBlock
from contextos.models.context import ContextWindow
from contextos.models.tier import MemoryTier
from contextos.rag.engine import RetrieverEngine
from contextos.scheduler.engine import ContextScheduler
from contextos.utils.embedder import Embedder

logger = logging.getLogger(__name__)


class ContextOS:
    """
    The main entry point for ContextOS.

    Quick start::

        from contextos import ContextOS

        cos = ContextOS()
        cos.store("The user's name is Alex")
        cos.store("Alex is interested in Linux kernel development")

        context = cos.allocate(budget=4096, query="What does Alex like?")
        print(context.as_prompt())

    Phase 1 API:
        cos.store(...)           store a new memory
        cos.get(block_id)        retrieve a specific block
        cos.get_tier(tier)       list all blocks in a tier
        cos.touch(block_id)      record access (update recency/frequency)
        cos.promote(...)         move a block to a hotter tier
        cos.demote(...)          move a block to a colder tier
        cos.run_transitions()    fire all automatic tier transition rules
        cos.status()             snapshot of current memory usage
        cos.close()              close DB connection

    Phase 2 API:
        cos.allocate(budget, query)   → ContextWindow (packed, budget-respecting)

    Phase 3 API:
        cos.evict(tier, budget)       → evict blocks from a tier
        cos.gc_run()                  → run garbage collector (expiry, dedup, prune)

    Phase 4 API:
        cos.cache_query(query)        → cached response if hit, else None
        cos.cache_response(q, r)      → store a new response in the cache

    Phase 5 API:
        cos.ingest(text, source)      → chunks, embeds, and stores a document
        cos.retrieve(query)           → returns ContextWindow using smart RAG
        cos.naive_retrieve(query)     → returns list of chunks (baseline comparison)
    """

    def __init__(self, config: ContextConfig | None = None):
        self.config = config or ContextConfig()

        # Validate config
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid ContextConfig: {errors}")

        # Ensure DB directory exists
        db_path = self.config.db_path_resolved()
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Wire components — Phase 1
        self._store = MemoryStore(db_path)
        self._embedder = Embedder(
            model_name=self.config.embedding_model,
            stub=self.config.embedding_stub,
        )
        self.memory = MemoryManager(self.config, self._store, self._embedder)

        # Wire components — Phase 2
        self.scheduler = ContextScheduler(policy=self.config.scheduling_policy)

        # Wire components — Phase 3
        self.eviction = EvictionEngine(self._store, policy=self.config.eviction_policy)
        self.gc = GarbageCollector(
            self._store,
            dedup_threshold=self.config.gc_dedup_threshold,
        )

        # Wire components — Phase 4
        if self.config.enable_semantic_cache:
            self._cache_store = CacheStore(db_path.parent / "semantic_cache.db")
            self.cache = SemanticCache(
                store=self._cache_store,
                embedder=self._embedder,
                similarity_threshold=self.config.cache_similarity_threshold,
                ttl_hours=self.config.cache_ttl_hours,
            )
        else:
            self.cache = None

        # Wire components — Phase 5
        self.retriever = RetrieverEngine(
            memory_manager=self.memory,
            persist_dir=self.config.chroma_dir_resolved(),
        )

        logger.info(
            "ContextOS initialised (db=%s, embedder=%s, stub=%s, scheduler=%s, eviction=%s, cache=%s, vector_db=%s)",
            db_path,
            self.config.embedding_model,
            self.config.embedding_stub,
            self.config.scheduling_policy,
            self.config.eviction_policy,
            self.config.enable_semantic_cache,
            self.config.vector_db,
        )

    # ── Context Version Control (VCS) ─────────────────────────────────────────

    def commit(self, tag: str) -> None:
        """Snapshot the current memory state for this agent (Version Control)."""
        import shutil
        import os
        
        # Close connections before copying
        self._store.close()
        if self.cache:
            self._cache_store.close()
            
        base_db = self.config.db_path_resolved()
        vcs_db = base_db.with_name(f"{base_db.stem}_vcs_{tag}{base_db.suffix}")
        shutil.copy2(str(base_db), str(vcs_db))
        
        if self.retriever and self.retriever.vector_index:
            base_chroma = self.config.chroma_dir_resolved()
            vcs_chroma = base_chroma.with_name(f"{base_chroma.name}_vcs_{tag}")
            if vcs_chroma.exists():
                shutil.rmtree(str(vcs_chroma))
            if base_chroma.exists():
                shutil.copytree(str(base_chroma), str(vcs_chroma))
            
        logger.info(f"ContextOS: Committed memory state '{tag}' for agent '{self.config.agent_id}'")
        
        # Reconnect
        self._store._connect()
        if self.cache:
            self._cache_store._connect()

    def checkout(self, tag: str) -> None:
        """Restore memory state to a previous snapshot."""
        import shutil
        import os
        
        base_db = self.config.db_path_resolved()
        vcs_db = base_db.with_name(f"{base_db.stem}_vcs_{tag}{base_db.suffix}")
        
        if not vcs_db.exists():
            raise ValueError(f"Snapshot '{tag}' does not exist for agent '{self.config.agent_id}'.")
            
        # Close connections
        self._store.close()
        if self.cache:
            self._cache_store.close()
        
        # Restore DB
        shutil.copy2(str(vcs_db), str(base_db))
        
        # Restore Chroma if exists
        if self.retriever and self.retriever.vector_index:
            base_chroma = self.config.chroma_dir_resolved()
            vcs_chroma = base_chroma.with_name(f"{base_chroma.name}_vcs_{tag}")
            if vcs_chroma.exists():
                if base_chroma.exists():
                    shutil.rmtree(str(base_chroma))
                shutil.copytree(str(vcs_chroma), str(base_chroma))
                
        logger.info(f"ContextOS: Checked out memory state '{tag}' for agent '{self.config.agent_id}'")
        
        # Reconnect
        self._store._connect()
        if self.cache:
            self._cache_store._connect()

    # ── Phase 1: Memory Management ────────────────────────────────────────────

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
        Store a new piece of information in memory.

        Args:
            content:          The text to remember. Can be any length.
            tier:             Where to initially store it.
                              - WORKING: current turn (included in every context)
                              - EPISODIC: recent events (default)
                              - SEMANTIC: compressed facts
                              - ARCHIVED: cold storage
            source:           Where this came from. Values: 'user', 'llm', 'tool',
                              'document', 'system'.
            tags:             Labels for grouping/filtering. Examples: ['user_profile'],
                              ['task_1'], ['pinned'].
            user_weight:      0.0–1.0 importance hint. 1.0 = never evict (pinned).
            expires_in_hours: Auto-delete after this many hours.
            metadata:         Arbitrary extra data dict.

        Returns:
            MemoryBlock — the created and persisted block.
        """
        return self.memory.store(
            content=content,
            tier=tier,
            source=source,
            tags=tags,
            user_weight=user_weight,
            expires_in_hours=expires_in_hours,
            metadata=metadata,
        )

    def get(self, block_id) -> MemoryBlock | None:
        """Retrieve a block by its UUID."""
        from uuid import UUID
        if not isinstance(block_id, UUID):
            block_id = UUID(str(block_id))
        return self.memory.get(block_id)

    def get_tier(self, tier: MemoryTier) -> list[MemoryBlock]:
        """All blocks in a given tier, sorted by importance score (highest first)."""
        return self.memory.get_tier(tier)

    def get_all(self) -> list[MemoryBlock]:
        """All blocks across all tiers."""
        return self.memory.get_all()

    def touch(self, block_id) -> MemoryBlock | None:
        """
        Record that a block was accessed.
        Increments frequency and updates last_accessed.
        Returns the updated block.
        """
        from uuid import UUID
        if not isinstance(block_id, UUID):
            block_id = UUID(str(block_id))
        return self.memory.touch(block_id)

    def promote(self, block_id, to_tier: MemoryTier) -> bool:
        """Move a block to a hotter tier. Returns True if transition occurred."""
        from uuid import UUID
        if not isinstance(block_id, UUID):
            block_id = UUID(str(block_id))
        return self.memory.promote(block_id, to_tier)

    def demote(self, block_id, to_tier: MemoryTier) -> bool:
        """Move a block to a colder tier. Pinned blocks are never demoted."""
        from uuid import UUID
        if not isinstance(block_id, UUID):
            block_id = UUID(str(block_id))
        return self.memory.demote(block_id, to_tier)

    def run_transitions(self) -> dict[str, int]:
        """
        Fire all automatic tier transition rules across the store.
        Returns counts of transitions made.
        """
        return self.memory.run_tier_transitions()

    def apply_time_decay(self) -> None:
        """Recompute recency scores for all blocks (call periodically)."""
        self.memory.apply_time_decay()

    def status(self) -> dict:
        """
        Snapshot of current memory usage.

        Returns::

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
        return self.memory.status()

    # ── Phase 2: Context Scheduling ───────────────────────────────────────────

    def allocate(
        self,
        budget: int | None = None,
        query: str = "",
    ) -> ContextWindow:
        """
        Allocate a context window: score, schedule, and pack the best memories
        into a token-budget-respecting ContextWindow.

        This is the main API for building a prompt from memory.

        Args:
            budget:  Max tokens for the context window.
                     Defaults to config.token_budget.
            query:   The current user query (used for scoring).

        Returns:
            ContextWindow — ordered MemoryBlocks that fit within budget,
            with `.as_prompt()` for formatted LLM input.

        Example::

            ctx = cos.allocate(budget=4096, query="Explain TCP")
            prompt = ctx.as_prompt()
            # Send prompt to your LLM
        """
        budget = budget or self.config.token_budget
        query_embedding = self._embedder.embed(query) if query else []

        return self.scheduler.schedule(
            memory_manager=self.memory,
            budget=budget,
            query_embedding=query_embedding,
            query=query,
        )

    # ── Phase 3: Eviction + Garbage Collection ────────────────────────────────

    def evict(
        self,
        tier: MemoryTier = MemoryTier.EPISODIC,
        budget: int | None = None,
    ) -> list[MemoryBlock]:
        """
        Evict blocks from a tier until it's under the token budget.

        Pinned blocks and WORKING tier are always protected.

        Args:
            tier:   Which tier to evict from. Default: EPISODIC.
            budget: Max tokens allowed in this tier.
                    Defaults to the config's tier budget.

        Returns:
            List of evicted MemoryBlocks.
        """
        if budget is None:
            if tier == MemoryTier.EPISODIC:
                budget = self.config.episodic_tier_budget
            elif tier == MemoryTier.SEMANTIC:
                budget = self.config.semantic_tier_budget
            else:
                budget = self.config.episodic_tier_budget  # fallback

        return self.eviction.evict_tier(tier, budget)

    def gc_run(self) -> dict[str, int]:
        """
        Run the garbage collector.

        Three passes:
          1. Expiry:  remove blocks past their expires_at
          2. Dedup:   remove near-duplicate blocks (cosine similarity > threshold)
          3. Prune:   remove very old, very low-score archived blocks

        Returns dict with counts: {'expired': N, 'deduplicated': N, 'pruned': N, 'total_freed': N}
        """
        return self.gc.run()

    # ── Phase 4: Semantic Cache ───────────────────────────────────────────────

    def cache_query(self, query: str) -> str | None:
        """
        Check the semantic cache for a similar query.
        Returns the cached response string if a hit is found, else None.
        """
        if not self.cache:
            return None
        hit = self.cache.search(query)
        if hit:
            return hit.response
        return None

    def cache_response(self, query: str, response: str, response_tokens: int = 0) -> None:
        """
        Save an LLM response to the semantic cache.
        """
        if not self.cache:
            return
        self.cache.add(query, response, response_tokens)

    # ── Phase 5: Smart RAG ────────────────────────────────────────────────────

    def ingest(self, text: str, source: str = "document", metadata: dict | None = None) -> list[MemoryBlock]:
        """
        Chunk a document and ingest it into the semantic memory tier and vector index.
        """
        return self.retriever.ingest(text, source, metadata)

    def retrieve(self, query: str, budget: int | None = None, retrieval_k: int = 20) -> ContextWindow:
        """
        Smart ContextOS RAG: retrieves candidates, scores them using importance rules,
        and uses the ContextScheduler to pack a budget-respecting ContextWindow.
        """
        budget = budget or self.config.token_budget
        return self.retriever.smart_retrieve(
            query=query,
            scheduler=self.scheduler,
            budget=budget,
            retrieval_k=retrieval_k,
        )
        
    def naive_retrieve(self, query: str, top_k: int = 5) -> list[MemoryBlock]:
        """
        Baseline RAG: naive top-k vector search without budget/tier awareness.
        """
        return self.retriever.naive_retrieve(query, top_k)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the database connections. Call when done."""
        self._store.close()
        if self.cache:
            self._cache_store.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self) -> str:
        s = self.status()
        return (
            f"ContextOS(blocks={s['total_blocks']}, "
            f"db={self.config.db_path}, "
            f"stub={self.config.embedding_stub})"
        )

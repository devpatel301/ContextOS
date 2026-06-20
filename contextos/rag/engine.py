"""
ContextOS — RetrieverEngine (Phase 5).

Provides both a naive top-K RAG baseline and a smart context-aware pipeline.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from contextos.models.block import MemoryBlock
from contextos.models.context import ContextWindow
from contextos.models.tier import MemoryTier
from contextos.rag.ingest import DocumentChunker
from contextos.rag.vector import VectorIndex

if TYPE_CHECKING:
    from contextos.memory.manager import MemoryManager
    from contextos.scheduler.engine import ContextScheduler

logger = logging.getLogger(__name__)


class RetrieverEngine:
    """
    RAG pipeline integrated with ContextOS.
    """

    def __init__(
        self,
        memory_manager: "MemoryManager",
        persist_dir: str | Path,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.memory = memory_manager
        self.vector_index = VectorIndex(persist_dir)
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)

    def ingest(self, text: str, source: str = "document", metadata: dict | None = None) -> list[MemoryBlock]:
        """
        Chunk a document, store chunks as SEMANTIC memories, and index them.
        """
        chunks = self.chunker.chunk_text(text)
        blocks: list[MemoryBlock] = []

        logger.info("Ingesting document from %s (%d chunks)", source, len(chunks))
        
        for i, chunk_text in enumerate(chunks):
            chunk_meta = (metadata or {}).copy()
            chunk_meta["chunk_index"] = i
            
            # Store in MemoryManager (this will compute embeddings and save to SQLite)
            block = self.memory.store(
                content=chunk_text,
                tier=MemoryTier.SEMANTIC,
                source=source,
                metadata=chunk_meta,
            )
            blocks.append(block)

        # Upsert all to Chroma DB for fast retrieval
        self.vector_index.upsert_blocks(blocks)
        return blocks

    def sync_all_memories(self) -> None:
        """
        One-off utility to sync all existing embedded memories into the vector index.
        """
        all_blocks = self.memory.get_all()
        embedded = [b for b in all_blocks if b.embedding]
        self.vector_index.upsert_blocks(embedded)
        logger.info("Synced %d existing memories to VectorIndex", len(embedded))

    def naive_retrieve(self, query: str, top_k: int = 5) -> list[MemoryBlock]:
        """
        BASELINE: Standard RAG naive top-K retrieval.
        Uses pure cosine similarity, ignores context budget and memory tiers.
        """
        query_embedding = self.memory.embedder.embed(query)
        if not query_embedding:
            return []

        # Find best chunk IDs
        result_ids = self.vector_index.search(query_embedding, top_k=top_k)
        
        # Fetch actual blocks
        blocks = []
        for rid in result_ids:
            block = self.memory.get(UUID(rid))
            if block:
                blocks.append(block)
                
        return blocks

    def smart_retrieve(
        self,
        query: str,
        scheduler: "ContextScheduler",
        budget: int,
        retrieval_k: int = 20,
    ) -> ContextWindow:
        """
        CONTEXTOS RAG: Retrieves candidates, re-scores them using the ContextOS
        importance scoring (recency, frequency, user weight, + similarity), 
        and then routes them through the scheduler alongside working/episodic memory.
        """
        query_embedding = self.memory.embedder.embed(query)
        if not query_embedding:
            return scheduler.schedule(self.memory, budget, [], query)

        # 1. Broad retrieval (get a large candidate pool of SEMANTIC blocks)
        candidate_ids = self.vector_index.search(
            query_embedding,
            top_k=retrieval_k,
            tier_filter=MemoryTier.SEMANTIC.value
        )
        
        retrieved_candidates = []
        for rid in candidate_ids:
            block = self.memory.get(UUID(rid))
            if block:
                retrieved_candidates.append(block)
                
        # 2. Add standard episodic/working candidates
        working = self.memory.get_tier(MemoryTier.WORKING)
        episodic = self.memory.get_tier(MemoryTier.EPISODIC)
        
        all_candidates = retrieved_candidates + working + episodic
        
        # 3. Deduplicate candidates (in case a block is returned twice)
        seen = set()
        unique_candidates = []
        for b in all_candidates:
            if b.id not in seen:
                seen.add(b.id)
                unique_candidates.append(b)

        # 4. Recompute importance scores against the query
        self.memory.recompute_scores(unique_candidates, query_embedding)
        
        # 5. Let the scheduler pack the context window!
        # (We bypass the standard scheduler.schedule() method because we are providing
        # our own custom candidate pool here).
        
        ordered = scheduler.policy.order(unique_candidates, query_embedding)
        
        # Greedily pack
        included = []
        tokens_used = 0
        
        # Force working blocks first
        working_blocks = [b for b in ordered if b.tier == MemoryTier.WORKING]
        for b in working_blocks:
            cost = b.effective_token_count()
            if tokens_used + cost <= budget:
                included.append(b)
                tokens_used += cost
                
        # Add the rest
        other_blocks = [b for b in ordered if b.tier != MemoryTier.WORKING]
        for b in other_blocks:
            cost = b.effective_token_count()
            if cost == 0:
                continue
            if tokens_used + cost > budget:
                continue
            included.append(b)
            tokens_used += cost
            
        # Touch included semantic blocks so they gain frequency/recency
        for b in included:
            if b.tier == MemoryTier.SEMANTIC:
                self.memory.touch(b.id)

        window = ContextWindow(
            blocks=included,
            token_budget=budget,
            tokens_used=tokens_used,
            query=query,
        )
        
        logger.info(
            "Smart retrieval scheduled context: %s (policy=%s)",
            window.summary(),
            scheduler.policy.name,
        )
        
        return window

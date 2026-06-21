"""
ContextOS — VectorIndex (ChromaDB wrapper).
"""
from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

import chromadb

from contextos.models.block import MemoryBlock

logger = logging.getLogger(__name__)


class VectorIndex:
    """
    Manages a ChromaDB collection for fast semantic search of MemoryBlocks.
    """

    def __init__(self, persist_dir: str | Path, collection_name: str = "contextos_memories"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.debug("Initialised VectorIndex at %s", self.persist_dir)

    def upsert_block(self, block: MemoryBlock) -> None:
        """Add or update a MemoryBlock in the vector index."""
        if not block.embedding:
            return

        self.collection.upsert(
            ids=[str(block.id)],
            embeddings=[block.embedding],
            metadatas=[{
                "tier": block.tier.value,
                "source": block.source,
                "created_at": block.created_at.isoformat()
            }],
            documents=[block.content]
        )

    def upsert_blocks(self, blocks: list[MemoryBlock]) -> None:
        """Batch add or update MemoryBlocks."""
        to_insert = [b for b in blocks if b.embedding]
        if not to_insert:
            return

        self.collection.upsert(
            ids=[str(b.id) for b in to_insert],
            embeddings=[b.embedding for b in to_insert],
            metadatas=[{
                "tier": b.tier.value,
                "source": b.source,
                "created_at": b.created_at.isoformat()
            } for b in to_insert],
            documents=[b.content for b in to_insert]
        )

    def delete_block(self, block_id: UUID | str) -> None:
        """Delete a block from the vector index."""
        try:
            self.collection.delete(ids=[str(block_id)])
        except Exception as e:
            logger.warning("Failed to delete block %s from VectorIndex: %s", block_id, e)

    def search(self, query_embedding: list[float], top_k: int = 10, tier_filter: str | None = None) -> list[str]:
        """
        Search for the most similar blocks.
        
        Returns:
            List of UUID strings for the matching blocks.
        """
        where = {"tier": tier_filter} if tier_filter else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances"]
        )
        
        if not results["ids"] or not results["ids"][0]:
            return []
            
        return results["ids"][0]

"""
ContextOS — SemanticCache engine.

Searches the CacheStore for semantically similar queries.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID

from contextos.models.cache import CacheEntry
from contextos.utils.embedder import Embedder

if TYPE_CHECKING:
    from contextos.cache.store import CacheStore

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Semantic cache for LLM responses.

    Usage:
        cache = SemanticCache(store, embedder, threshold=0.92)
        hit = cache.search("Explain TCP")
        if hit:
            return hit.response
        
        response = call_llm("Explain TCP")
        cache.add("Explain TCP", response)
    """

    def __init__(
        self,
        store: "CacheStore",
        embedder: Embedder,
        similarity_threshold: float = 0.92,
        ttl_hours: int = 24,
    ):
        self.store = store
        self.embedder = embedder
        self.threshold = similarity_threshold
        self.ttl_hours = ttl_hours

    def search(self, query: str) -> CacheEntry | None:
        """
        Search for a semantically similar cached query.
        Updates hit_count on cache hit.

        Args:
            query: The user query string.

        Returns:
            The CacheEntry if a match >= threshold is found and not expired,
            otherwise None.
        """
        query_embedding = self.embedder.embed(query)
        if not query_embedding:
            return None

        # O(n) exact search (fine for small caches, but consider HNSW for production)
        all_entries = self.store.get_all()
        
        best_match: CacheEntry | None = None
        best_score = -1.0

        for entry in all_entries:
            if entry.is_expired():
                continue

            sim = Embedder.cosine_similarity(query_embedding, entry.query_embedding)
            if sim >= self.threshold and sim > best_score:
                best_match = entry
                best_score = sim

        if best_match:
            logger.info("Cache hit for query '%s' (similarity: %.3f)", query, best_score)
            best_match.touch()
            self.store.save(best_match)
            return best_match

        logger.debug("Cache miss for query '%s'", query)
        return None

    def add(
        self,
        query: str,
        response: str,
        response_tokens: int = 0,
        tags: list[str] | None = None,
    ) -> CacheEntry | None:
        """
        Cache a new query and response pair.

        Args:
            query: The user query.
            response: The LLM response.
            response_tokens: Optional token count of the response.
            tags: Optional labels.

        Returns:
            The created CacheEntry, or None if embedding failed.
        """
        from datetime import datetime, timedelta, timezone
        
        query_embedding = self.embedder.embed(query)
        if not query_embedding:
            logger.warning("Could not embed query '%s', skipping cache", query)
            return None

        entry = CacheEntry(
            query=query,
            query_embedding=query_embedding,
            response=response,
            response_tokens=response_tokens,
            tags=tags or [],
        )
        
        # Override default TTL if needed
        entry.expires_at = datetime.now(timezone.utc) + timedelta(hours=self.ttl_hours)
        
        self.store.save(entry)
        logger.debug("Cached response for query '%s' (id=%s)", query, entry.id)
        return entry

    def invalidate_expired(self) -> int:
        """Remove expired entries from the store."""
        count = self.store.delete_expired()
        if count > 0:
            logger.info("SemanticCache invalidated %d expired entries", count)
        return count

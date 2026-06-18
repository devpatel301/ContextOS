"""
ContextOS — CacheStore.

SQLite persistence layer for the Semantic Cache.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from uuid import UUID

from contextos.models.cache import CacheEntry

logger = logging.getLogger(__name__)


class CacheStore:
    """
    SQLite persistence for Semantic Cache entries.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = None
        self._connect()
        self._init_db()

    def _connect(self):
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode
        )
        self._conn.row_factory = sqlite3.Row

    def _init_db(self):
        """Create the cache table if it doesn't exist."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_cache (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                query_embedding TEXT NOT NULL,
                response TEXT NOT NULL,
                response_tokens INTEGER NOT NULL,
                hit_count INTEGER DEFAULT 0,
                tags TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
            """
        )
        # We index on expires_at to quickly prune expired entries
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_expires_at ON semantic_cache(expires_at)"
        )
        self._conn.commit()

    def save(self, entry: CacheEntry) -> None:
        """Insert or update a cache entry."""
        data = entry.to_dict()
        self._conn.execute(
            """
            INSERT INTO semantic_cache (
                id, query, query_embedding, response, response_tokens,
                hit_count, tags, metadata, created_at, expires_at
            ) VALUES (
                :id, :query, :query_embedding, :response, :response_tokens,
                :hit_count, :tags, :metadata, :created_at, :expires_at
            )
            ON CONFLICT(id) DO UPDATE SET
                hit_count = excluded.hit_count,
                expires_at = excluded.expires_at,
                tags = excluded.tags,
                metadata = excluded.metadata
            """,
            data,
        )

    def get(self, entry_id: UUID | str) -> CacheEntry | None:
        """Retrieve a specific cache entry by ID."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM semantic_cache WHERE id = ?",
            (str(entry_id),),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return CacheEntry.from_dict(dict(row))

    def get_all(self) -> list[CacheEntry]:
        """Get all cache entries. Caution: loads all embeddings into memory."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM semantic_cache")
        return [CacheEntry.from_dict(dict(row)) for row in cursor.fetchall()]

    def delete(self, entry_id: UUID | str) -> bool:
        """Delete a cache entry."""
        cursor = self._conn.execute(
            "DELETE FROM semantic_cache WHERE id = ?",
            (str(entry_id),),
        )
        return cursor.rowcount > 0

    def delete_expired(self) -> int:
        """Delete all expired cache entries."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            "DELETE FROM semantic_cache WHERE expires_at < ?",
            (now,),
        )
        return cursor.rowcount

    def clear(self) -> int:
        """Delete all cache entries. Returns number of rows deleted."""
        cursor = self._conn.execute("DELETE FROM semantic_cache")
        return cursor.rowcount

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

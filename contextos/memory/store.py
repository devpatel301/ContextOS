"""
ContextOS — SQLite-backed MemoryStore.

Persists MemoryBlocks to a local SQLite database.
Uses only stdlib sqlite3 — no ORM dependency.

Schema is initialised automatically on first use via init_db().
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import UUID

from contextos.models.block import MemoryBlock
from contextos.models.tier import MemoryTier

# ISO 8601 format for datetime round-trips
_DT_FMT = "%Y-%m-%dT%H:%M:%S.%f"


def _dt_to_str(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.strftime(_DT_FMT)


def _str_to_dt(s: str | None) -> datetime | None:
    if s is None:
        return None
    return datetime.strptime(s, _DT_FMT)


class MemoryStore:
    """
    SQLite persistence layer for MemoryBlocks.

    Thread-safety: Each call acquires a connection from a simple per-instance
    connection. For multi-threaded use, create one MemoryStore per thread
    (or add a connection pool).
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS memory_blocks (
        id                TEXT PRIMARY KEY,
        content           TEXT NOT NULL,
        summary           TEXT,
        embedding         BLOB,
        tier              TEXT NOT NULL,
        importance_score  REAL NOT NULL DEFAULT 0.5,
        recency_score     REAL NOT NULL DEFAULT 1.0,
        frequency         INTEGER NOT NULL DEFAULT 0,
        user_weight       REAL NOT NULL DEFAULT 0.5,
        token_count       INTEGER NOT NULL DEFAULT 0,
        token_count_summary INTEGER NOT NULL DEFAULT 0,
        created_at        TEXT NOT NULL,
        last_accessed     TEXT NOT NULL,
        expires_at        TEXT,
        source            TEXT NOT NULL DEFAULT 'user',
        tags              TEXT NOT NULL DEFAULT '[]',
        metadata          TEXT NOT NULL DEFAULT '{}'
    );

    CREATE INDEX IF NOT EXISTS idx_blocks_tier
        ON memory_blocks(tier);

    CREATE INDEX IF NOT EXISTS idx_blocks_importance
        ON memory_blocks(importance_score DESC);

    CREATE INDEX IF NOT EXISTS idx_blocks_last_accessed
        ON memory_blocks(last_accessed DESC);

    CREATE INDEX IF NOT EXISTS idx_blocks_created_at
        ON memory_blocks(created_at DESC);
    """

    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._connect()
        self._init_schema()

    # ── Connection management ─────────────────────────────────────────────────

    def _connect(self) -> None:
        self._conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        # WAL mode for better concurrent reads
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def _init_schema(self) -> None:
        assert self._conn is not None
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _row_to_block(self, row: sqlite3.Row) -> MemoryBlock:
        return MemoryBlock(
            id=UUID(row["id"]),
            content=row["content"],
            summary=row["summary"],
            embedding=MemoryBlock.embedding_from_bytes(row["embedding"] or b""),
            tier=MemoryTier(row["tier"]),
            importance_score=row["importance_score"],
            recency_score=row["recency_score"],
            frequency=row["frequency"],
            user_weight=row["user_weight"],
            token_count=row["token_count"],
            token_count_summary=row["token_count_summary"],
            created_at=_str_to_dt(row["created_at"]) or datetime.utcnow(),
            last_accessed=_str_to_dt(row["last_accessed"]) or datetime.utcnow(),
            expires_at=_str_to_dt(row["expires_at"]),
            source=row["source"],
            tags=json.loads(row["tags"]),
            metadata=json.loads(row["metadata"]),
        )

    def _block_to_params(self, block: MemoryBlock) -> dict:
        return {
            "id": str(block.id),
            "content": block.content,
            "summary": block.summary,
            "embedding": block.embedding_to_bytes(),
            "tier": block.tier.value,
            "importance_score": block.importance_score,
            "recency_score": block.recency_score,
            "frequency": block.frequency,
            "user_weight": block.user_weight,
            "token_count": block.token_count,
            "token_count_summary": block.token_count_summary,
            "created_at": _dt_to_str(block.created_at),
            "last_accessed": _dt_to_str(block.last_accessed),
            "expires_at": _dt_to_str(block.expires_at),
            "source": block.source,
            "tags": block.tags_to_json(),
            "metadata": block.metadata_to_json(),
        }

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def save(self, block: MemoryBlock) -> MemoryBlock:
        """Insert or update a MemoryBlock. Returns the block (same object)."""
        params = self._block_to_params(block)
        self._conn.execute(
            """
            INSERT INTO memory_blocks
                (id, content, summary, embedding, tier, importance_score,
                 recency_score, frequency, user_weight, token_count,
                 token_count_summary, created_at, last_accessed, expires_at,
                 source, tags, metadata)
            VALUES
                (:id, :content, :summary, :embedding, :tier, :importance_score,
                 :recency_score, :frequency, :user_weight, :token_count,
                 :token_count_summary, :created_at, :last_accessed, :expires_at,
                 :source, :tags, :metadata)
            ON CONFLICT(id) DO UPDATE SET
                content           = excluded.content,
                summary           = excluded.summary,
                embedding         = excluded.embedding,
                tier              = excluded.tier,
                importance_score  = excluded.importance_score,
                recency_score     = excluded.recency_score,
                frequency         = excluded.frequency,
                user_weight       = excluded.user_weight,
                token_count       = excluded.token_count,
                token_count_summary = excluded.token_count_summary,
                last_accessed     = excluded.last_accessed,
                expires_at        = excluded.expires_at,
                source            = excluded.source,
                tags              = excluded.tags,
                metadata          = excluded.metadata
            """,
            params,
        )
        self._conn.commit()
        return block

    def get(self, block_id: UUID) -> MemoryBlock | None:
        """Fetch a single block by ID. Returns None if not found."""
        row = self._conn.execute(
            "SELECT * FROM memory_blocks WHERE id = ?", (str(block_id),)
        ).fetchone()
        return self._row_to_block(row) if row else None

    def delete(self, block_id: UUID) -> bool:
        """Delete a block. Returns True if it existed."""
        cur = self._conn.execute(
            "DELETE FROM memory_blocks WHERE id = ?", (str(block_id),)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete_many(self, block_ids: list[UUID]) -> int:
        """Delete multiple blocks. Returns count deleted."""
        if not block_ids:
            return 0
        placeholders = ",".join("?" * len(block_ids))
        cur = self._conn.execute(
            f"DELETE FROM memory_blocks WHERE id IN ({placeholders})",
            [str(bid) for bid in block_ids],
        )
        self._conn.commit()
        return cur.rowcount

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_by_tier(self, tier: MemoryTier) -> list[MemoryBlock]:
        """All blocks in a given tier, ordered by importance DESC."""
        rows = self._conn.execute(
            """
            SELECT * FROM memory_blocks
            WHERE tier = ?
            ORDER BY importance_score DESC, last_accessed DESC
            """,
            (tier.value,),
        ).fetchall()
        return [self._row_to_block(r) for r in rows]

    def get_all(self) -> list[MemoryBlock]:
        """All blocks across all tiers."""
        rows = self._conn.execute(
            "SELECT * FROM memory_blocks ORDER BY importance_score DESC"
        ).fetchall()
        return [self._row_to_block(r) for r in rows]

    def get_by_tag(self, tag: str) -> list[MemoryBlock]:
        """Blocks containing a specific tag (JSON array substring match)."""
        rows = self._conn.execute(
            "SELECT * FROM memory_blocks WHERE tags LIKE ?",
            (f'%"{tag}"%',),
        ).fetchall()
        return [self._row_to_block(r) for r in rows]

    def count(self, tier: MemoryTier | None = None) -> int:
        """Count blocks, optionally filtered by tier."""
        if tier:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM memory_blocks WHERE tier = ?", (tier.value,)
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM memory_blocks"
            ).fetchone()
        return row[0]

    def token_usage(self, tier: MemoryTier | None = None) -> int:
        """Total token count across blocks, optionally filtered by tier."""
        if tier:
            row = self._conn.execute(
                """
                SELECT COALESCE(SUM(
                    CASE WHEN summary IS NOT NULL AND token_count_summary > 0
                         THEN token_count_summary ELSE token_count END
                ), 0)
                FROM memory_blocks WHERE tier = ?
                """,
                (tier.value,),
            ).fetchone()
        else:
            row = self._conn.execute(
                """
                SELECT COALESCE(SUM(
                    CASE WHEN summary IS NOT NULL AND token_count_summary > 0
                         THEN token_count_summary ELSE token_count END
                ), 0)
                FROM memory_blocks
                """
            ).fetchone()
        return row[0]

    def get_expired(self) -> list[MemoryBlock]:
        """All blocks whose expires_at has passed."""
        now = _dt_to_str(datetime.utcnow())
        rows = self._conn.execute(
            "SELECT * FROM memory_blocks WHERE expires_at IS NOT NULL AND expires_at < ?",
            (now,),
        ).fetchall()
        return [self._row_to_block(r) for r in rows]

    def get_idle_since(self, hours: float, tier: MemoryTier | None = None) -> list[MemoryBlock]:
        """Blocks not accessed for at least `hours` hours."""
        from datetime import timedelta
        cutoff = _dt_to_str(datetime.utcnow() - timedelta(hours=hours))
        if tier:
            rows = self._conn.execute(
                "SELECT * FROM memory_blocks WHERE last_accessed < ? AND tier = ?",
                (cutoff, tier.value),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memory_blocks WHERE last_accessed < ?",
                (cutoff,),
            ).fetchall()
        return [self._row_to_block(r) for r in rows]

    def update_tier(self, block_id: UUID, new_tier: MemoryTier) -> None:
        """Update only the tier of a block (fast path for tier transitions)."""
        self._conn.execute(
            "UPDATE memory_blocks SET tier = ? WHERE id = ?",
            (new_tier.value, str(block_id)),
        )
        self._conn.commit()

    def update_access(self, block_id: UUID, last_accessed: datetime, frequency: int) -> None:
        """Fast path: update access tracking fields only."""
        self._conn.execute(
            "UPDATE memory_blocks SET last_accessed = ?, frequency = ? WHERE id = ?",
            (_dt_to_str(last_accessed), frequency, str(block_id)),
        )
        self._conn.commit()

    def update_scores(self, block_id: UUID, importance: float, recency: float) -> None:
        """Fast path: update scores only (called after recompute_scores)."""
        self._conn.execute(
            "UPDATE memory_blocks SET importance_score = ?, recency_score = ? WHERE id = ?",
            (importance, recency, str(block_id)),
        )
        self._conn.commit()

    def flush_tier_scores(self, blocks: list[MemoryBlock]) -> None:
        """Batch update scores for a list of blocks (efficient after recompute)."""
        params = [
            (b.importance_score, b.recency_score, str(b.id)) for b in blocks
        ]
        self._conn.executemany(
            "UPDATE memory_blocks SET importance_score = ?, recency_score = ? WHERE id = ?",
            params,
        )
        self._conn.commit()

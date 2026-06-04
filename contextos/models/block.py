"""
ContextOS — MemoryBlock, the atomic unit of the memory system.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4

from contextos.models.tier import MemoryTier


@dataclass
class MemoryBlock:
    """
    The atomic unit stored and managed by ContextOS.

    Every piece of information — a user message, a tool result, a retrieved
    document chunk, a compressed summary — is a MemoryBlock.
    """

    # ── Identity ─────────────────────────────────────────────────────────────
    id: UUID = field(default_factory=uuid4)

    # ── Content ───────────────────────────────────────────────────────────────
    content: str = ""
    """Raw text content. Always preserved; never discarded."""

    summary: str | None = None
    """LLM-compressed version. If set, this is served in the context window
    instead of `content` to save tokens."""

    embedding: list[float] = field(default_factory=list)
    """Sentence-transformer embedding of `content`. Used for semantic search."""

    token_count: int = 0
    """Token count of `content`, computed via tiktoken."""

    token_count_summary: int = 0
    """Token count of `summary` (0 if no summary)."""

    # ── Tier ──────────────────────────────────────────────────────────────────
    tier: MemoryTier = MemoryTier.WORKING

    # ── Scoring ───────────────────────────────────────────────────────────────
    importance_score: float = 0.5
    """Composite score (0–1). Recomputed on each allocate() call.
    Formula: 0.4*similarity + 0.3*recency + 0.2*frequency + 0.1*user_weight"""

    recency_score: float = 1.0
    """Time-decay score (0–1). 1.0 = just created, halves every 24h by default."""

    frequency: int = 0
    """Number of times this block has been accessed / included in a context window."""

    user_weight: float = 0.5
    """User-assigned importance hint. Default 0.5 = neutral. 1.0 = pinned."""

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None

    # ── Provenance ────────────────────────────────────────────────────────────
    agent_id: str = "default"
    """The agent namespace this block belongs to."""
    
    source: str = "user"
    """Origin of this memory. Values: 'user', 'llm', 'tool', 'document', 'system'."""

    tags: list[str] = field(default_factory=list)
    """Free-form labels for grouping, filtering, and cache invalidation."""

    metadata: dict = field(default_factory=dict)
    """Arbitrary extra data (page number, doc name, tool name, etc.)."""

    # ── Derived helpers ───────────────────────────────────────────────────────

    def effective_content(self) -> str:
        """Return summary if available, else raw content."""
        return self.summary if self.summary else self.content

    def effective_token_count(self) -> int:
        """Token cost as it would appear in a context window."""
        if self.summary and self.token_count_summary > 0:
            return self.token_count_summary
        return self.token_count

    def is_expired(self) -> bool:
        """True if this block has passed its expiry time."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def is_pinned(self) -> bool:
        """Pinned blocks never get evicted (user_weight == 1.0 or 'pinned' tag)."""
        return self.user_weight >= 1.0 or "pinned" in self.tags

    def touch(self) -> None:
        """Update last_accessed and increment frequency. Call when block is used."""
        self.last_accessed = datetime.utcnow()
        self.frequency += 1

    # ── Serialisation helpers (used by MemoryStore) ───────────────────────────

    def tags_to_json(self) -> str:
        return json.dumps(self.tags)

    def metadata_to_json(self) -> str:
        return json.dumps(self.metadata)

    def embedding_to_bytes(self) -> bytes:
        import struct
        if not self.embedding:
            return b""
        return struct.pack(f"{len(self.embedding)}f", *self.embedding)

    @staticmethod
    def embedding_from_bytes(data: bytes) -> list[float]:
        import struct
        if not data:
            return []
        count = len(data) // 4
        return list(struct.unpack(f"{count}f", data))

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return (
            f"MemoryBlock(id={str(self.id)[:8]}…, tier={self.tier.value}, "
            f"score={self.importance_score:.2f}, tokens={self.token_count}, "
            f"content={preview!r})"
        )

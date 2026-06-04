"""
ContextOS — Cache data models.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4


def _default_expires_at() -> datetime:
    """Default TTL for cache entries is 24 hours."""
    # Using timezone-aware objects to represent datetimes in UTC
    from datetime import timezone
    return datetime.now(timezone.utc) + timedelta(hours=24)


@dataclass
class CacheEntry:
    """
    Represents a cached LLM response for a specific query.
    """
    query: str
    query_embedding: list[float]
    response: str
    id: UUID = field(default_factory=uuid4)
    response_tokens: int = 0
    hit_count: int = 0
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # We use timezone-aware objects
    created_at: datetime = field(init=False)
    expires_at: datetime = field(default_factory=_default_expires_at)

    def __post_init__(self):
        from datetime import timezone
        if not hasattr(self, "created_at"):
            self.created_at = datetime.now(timezone.utc)

    def touch(self) -> None:
        """Record a cache hit."""
        self.hit_count += 1
        
    def is_expired(self) -> bool:
        """Check if the entry has passed its expiration date."""
        from datetime import timezone
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> dict:
        """Convert to dict for persistence."""
        return {
            "id": str(self.id),
            "query": self.query,
            "query_embedding": json.dumps(self.query_embedding),
            "response": self.response,
            "response_tokens": self.response_tokens,
            "hit_count": self.hit_count,
            "tags": json.dumps(self.tags),
            "metadata": json.dumps(self.metadata),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CacheEntry:
        """Reconstruct from persistence dict."""
        entry = cls(
            id=UUID(data["id"]),
            query=data["query"],
            query_embedding=json.loads(data["query_embedding"]),
            response=data["response"],
            response_tokens=data.get("response_tokens", 0),
            hit_count=data.get("hit_count", 0),
            tags=json.loads(data.get("tags", "[]")),
            metadata=json.loads(data.get("metadata", "{}")),
        )
        
        # Override created_at which was set in post_init
        entry.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("expires_at"):
            entry.expires_at = datetime.fromisoformat(data["expires_at"])
            
        return entry

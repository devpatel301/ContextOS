"""
ContextOS — global configuration dataclass.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ContextConfig:
    """
    All tuneable parameters for a ContextOS instance.

    Create with defaults and override only what you need:

        config = ContextConfig(token_budget=32_000, eviction_policy="lru")
    """

    # ── Token budget ──────────────────────────────────────────────────────────
    token_budget: int = 16_000
    """Total token budget for a context window (input tokens sent to LLM)."""

    working_memory_reserve: int = 2_000
    """Tokens always reserved for WORKING tier (system prompt + current turn)."""

    # ── Tier budgets ──────────────────────────────────────────────────────────
    working_tier_budget: int = 4_000
    episodic_tier_budget: int = 8_000
    # Semantic/Archived are stored in DB; their "budget" is retrieval count, not tokens

    # ── LLM backend ───────────────────────────────────────────────────────────
    llm_provider: str = "openai"          # "openai" | "ollama" | "none"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    """sentence-transformers model name. Used for memory storage and retrieval."""

    embedding_stub: bool = False
    """If True, use a deterministic hash-based fake embedding (no GPU needed).
    Useful for Phase 0/1 tests that don't need real semantic similarity."""

    # ── Vector DB ─────────────────────────────────────────────────────────────
    vector_db: str = "chroma"             # "chroma" | "faiss" | "none"
    chroma_persist_dir: str = "./.chroma"
    agent_id: str = "default"

    # Memory limitse ────────────────────────────────────────────────────────────────
    db_path: str = "./contextos.db"

    # ── Eviction + scheduling ─────────────────────────────────────────────────
    eviction_policy: str = "hybrid"       # lru | lfu | fifo | priority | hybrid
    scheduling_policy: str = "weighted_fair"

    # ── Tier transition thresholds ────────────────────────────────────────────
    episodic_to_semantic_access_count: int = 3
    """How many accesses promote a block from EPISODIC → SEMANTIC."""

    semantic_to_archived_idle_hours: int = 24
    """Hours of no access before SEMANTIC → ARCHIVED."""

    archived_age_days: int = 7
    """Days after creation before any tier block can be archived."""

    # ── Importance scoring weights ─────────────────────────────────────────────
    score_weight_similarity: float = 0.40
    score_weight_recency: float = 0.30
    score_weight_frequency: float = 0.20
    score_weight_user: float = 0.10

    recency_half_life_hours: float = 24.0
    """Recency score halves every N hours (exponential decay)."""

    # ── Semantic cache ─────────────────────────────────────────────────────────
    enable_semantic_cache: bool = True
    cache_similarity_threshold: float = 0.85
    cache_ttl_hours: int = 24

    # ── Garbage collector ──────────────────────────────────────────────────────
    enable_gc: bool = True
    gc_interval_seconds: int = 300
    gc_dedup_threshold: float = 0.92

    # ── Observability ──────────────────────────────────────────────────────────
    enable_profiler: bool = True

    # ── Derived helpers ────────────────────────────────────────────────────────

    def db_path_resolved(self) -> Path:
        p = Path(self.db_path).expanduser().resolve()
        if self.agent_id != "default":
            p = p.with_name(f"{p.stem}_{self.agent_id}{p.suffix}")
        return p

    def chroma_dir_resolved(self) -> Path:
        p = Path(self.chroma_persist_dir).expanduser().resolve()
        if self.agent_id != "default":
            p = p.with_name(f"{p.name}_{self.agent_id}")
        return p

    def validate(self) -> list[str]:
        """Return list of validation error strings (empty = valid)."""
        errors: list[str] = []
        weights = (
            self.score_weight_similarity
            + self.score_weight_recency
            + self.score_weight_frequency
            + self.score_weight_user
        )
        if abs(weights - 1.0) > 1e-6:
            errors.append(f"Importance score weights must sum to 1.0, got {weights:.4f}")
        if self.token_budget <= self.working_memory_reserve:
            errors.append("token_budget must be > working_memory_reserve")
        return errors

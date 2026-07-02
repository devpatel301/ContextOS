"""
Shared test fixtures used across all test files.
"""
import pytest
from contextos import ContextConfig, ContextOS
from contextos.memory.store import MemoryStore
from contextos.models.tier import MemoryTier
from contextos.utils.embedder import Embedder


@pytest.fixture
def stub_embedder():
    """A deterministic fake embedder that needs no GPU."""
    return Embedder(stub=True)


@pytest.fixture
def in_memory_store():
    """SQLite in-memory store (reset between tests)."""
    store = MemoryStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def stub_config(tmp_path):
    """ContextConfig configured for fast testing: stub embedder, temp SQLite DB."""
    return ContextConfig(
        db_path=str(tmp_path / "test.db"),
        embedding_stub=True,
        llm_provider="none",
        enable_semantic_cache=False,
        enable_gc=False,
        enable_profiler=False,
        token_budget=8_000,
    )


@pytest.fixture
def cos(stub_config):
    """A fully-wired ContextOS instance with stub embedder and temp DB."""
    with ContextOS(stub_config) as c:
        yield c

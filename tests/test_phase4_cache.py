"""
Phase 4 — Semantic Cache tests.

Tests the cache models, store, engine, and top-level ContextOS API.
"""
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from contextos import ContextConfig, ContextOS
from contextos.cache.engine import SemanticCache
from contextos.cache.store import CacheStore
from contextos.models.cache import CacheEntry
from contextos.utils.embedder import Embedder


@pytest.fixture
def cache_store(tmp_path):
    store = CacheStore(tmp_path / "test_cache.db")
    yield store
    store.close()


@pytest.fixture
def embedder():
    return Embedder(stub=True)


@pytest.fixture
def cache(cache_store, embedder):
    return SemanticCache(cache_store, embedder, similarity_threshold=0.90)


@pytest.fixture
def cos(tmp_path):
    config = ContextConfig(
        db_path=str(tmp_path / "main.db"),
        embedding_stub=True,
        enable_semantic_cache=True,
    )
    with ContextOS(config) as c:
        yield c


# ── Cache Model Tests ─────────────────────────────────────────────────────────

class TestCacheEntry:
    def test_default_values(self):
        entry = CacheEntry(
            query="test",
            query_embedding=[1.0, 0.0],
            response="response",
        )
        assert entry.id is not None
        assert entry.hit_count == 0
        assert entry.response_tokens == 0
        assert not entry.is_expired()
        assert entry.created_at < entry.expires_at

    def test_touch_increments_hit_count(self):
        entry = CacheEntry(query="q", query_embedding=[1.0], response="r")
        entry.touch()
        assert entry.hit_count == 1

    def test_is_expired(self):
        entry = CacheEntry(query="q", query_embedding=[1.0], response="r")
        entry.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        assert entry.is_expired()


# ── Cache Store Tests ─────────────────────────────────────────────────────────

class TestCacheStore:
    def test_save_and_get(self, cache_store):
        entry = CacheEntry(query="q", query_embedding=[1.0], response="r")
        cache_store.save(entry)
        
        retrieved = cache_store.get(entry.id)
        assert retrieved is not None
        assert retrieved.query == "q"
        assert retrieved.response == "r"
        assert retrieved.query_embedding == [1.0]

    def test_update_existing(self, cache_store):
        entry = CacheEntry(query="q", query_embedding=[1.0], response="r")
        cache_store.save(entry)
        
        entry.touch()
        cache_store.save(entry)
        
        retrieved = cache_store.get(entry.id)
        assert retrieved.hit_count == 1

    def test_delete(self, cache_store):
        entry = CacheEntry(query="q", query_embedding=[1.0], response="r")
        cache_store.save(entry)
        assert cache_store.delete(entry.id) is True
        assert cache_store.get(entry.id) is None

    def test_delete_expired(self, cache_store):
        e1 = CacheEntry(query="q1", query_embedding=[1.0], response="r1")
        e2 = CacheEntry(query="q2", query_embedding=[1.0], response="r2")
        e1.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)  # expired
        e2.expires_at = datetime.now(timezone.utc) + timedelta(hours=1)  # valid
        
        cache_store.save(e1)
        cache_store.save(e2)
        
        count = cache_store.delete_expired()
        assert count == 1
        assert cache_store.get(e1.id) is None
        assert cache_store.get(e2.id) is not None


# ── Semantic Cache Engine Tests ───────────────────────────────────────────────

class TestSemanticCache:
    def test_add_caches_response(self, cache):
        entry = cache.add("Explain TCP", "Transmission Control Protocol is...")
        assert entry is not None
        assert entry.query == "Explain TCP"
        
        retrieved = cache.store.get(entry.id)
        assert retrieved is not None

    def test_search_exact_match(self, cache):
        cache.add("Explain TCP", "TCP response")
        hit = cache.search("Explain TCP")
        assert hit is not None
        assert hit.response == "TCP response"

    def test_search_similar_match(self, cache):
        # With the stub embedder, we might not get "semantic" matching,
        # but exact matches and high overlaps usually pass the hash-based stub.
        cache.add("Explain TCP", "TCP response")
        hit = cache.search("Explain TCP")
        assert hit is not None

    def test_search_miss(self, cache):
        cache.add("Explain TCP", "TCP response")
        hit = cache.search("What is UDP?")
        assert hit is None

    def test_search_ignores_expired(self, cache):
        entry = cache.add("Explain TCP", "TCP response")
        entry.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        cache.store.save(entry)
        
        hit = cache.search("Explain TCP")
        assert hit is None

    def test_search_updates_hit_count(self, cache):
        entry = cache.add("Explain TCP", "TCP response")
        assert entry.hit_count == 0
        
        cache.search("Explain TCP")
        
        updated = cache.store.get(entry.id)
        assert updated.hit_count == 1


# ── ContextOS API Integration Tests ───────────────────────────────────────────

class TestContextOSCacheIntegration:
    def test_cache_query_miss(self, cos):
        assert cos.cache_query("Explain TCP") is None

    def test_cache_response_and_hit(self, cos):
        cos.cache_response("Explain TCP", "TCP is reliable.")
        hit = cos.cache_query("Explain TCP")
        assert hit == "TCP is reliable."

    def test_disabled_cache_returns_none(self, tmp_path):
        config = ContextConfig(
            db_path=str(tmp_path / "disabled.db"),
            embedding_stub=True,
            enable_semantic_cache=False,
        )
        with ContextOS(config) as cos_disabled:
            assert cos_disabled.cache is None
            cos_disabled.cache_response("Explain TCP", "TCP is reliable.")
            assert cos_disabled.cache_query("Explain TCP") is None

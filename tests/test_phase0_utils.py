"""
Phase 0 — Token counting and Embedder tests.

No LLM or real sentence-transformers needed.
"""
import pytest

from contextos.utils.embedder import Embedder
from contextos.utils.tokens import count_tokens, count_tokens_batch, fits_in_budget


# ── Token counting ────────────────────────────────────────────────────────────

class TestTokenCounting:
    def test_empty_string_is_zero(self):
        assert count_tokens("") == 0

    def test_short_text_positive(self):
        n = count_tokens("hello world")
        assert n > 0

    def test_longer_text_more_tokens(self):
        short = count_tokens("hi")
        long = count_tokens("This is a much longer sentence with many words in it.")
        assert long > short

    def test_batch_same_as_individual(self):
        texts = ["hello", "world", "this is a test sentence"]
        batch = count_tokens_batch(texts)
        individual = [count_tokens(t) for t in texts]
        assert batch == individual

    def test_fits_in_budget_true(self):
        assert fits_in_budget("hi", budget=100)

    def test_fits_in_budget_false(self):
        long_text = "word " * 10_000
        assert not fits_in_budget(long_text, budget=100)

    def test_batch_empty_list(self):
        assert count_tokens_batch([]) == []


# ── Stub Embedder ─────────────────────────────────────────────────────────────

class TestStubEmbedder:
    def test_embed_returns_384_floats(self):
        emb = Embedder(stub=True)
        v = emb.embed("hello world")
        assert len(v) == 384
        assert all(isinstance(x, float) for x in v)

    def test_embed_deterministic(self):
        emb = Embedder(stub=True)
        v1 = emb.embed("same text")
        v2 = emb.embed("same text")
        assert v1 == v2

    def test_embed_different_texts_differ(self):
        emb = Embedder(stub=True)
        v1 = emb.embed("cat")
        v2 = emb.embed("dog")
        assert v1 != v2

    def test_embed_unit_normalized(self):
        """Stub embeddings should be unit-norm (for cosine similarity to work)."""
        import math
        emb = Embedder(stub=True)
        v = emb.embed("normalize me")
        norm = math.sqrt(sum(x ** 2 for x in v))
        assert abs(norm - 1.0) < 1e-4

    def test_embed_batch_length(self):
        emb = Embedder(stub=True)
        texts = ["alpha", "beta", "gamma", "delta"]
        vecs = emb.embed_batch(texts)
        assert len(vecs) == 4
        for v in vecs:
            assert len(v) == 384

    def test_cosine_similarity_identical(self):
        emb = Embedder(stub=True)
        v = emb.embed("hello")
        sim = Embedder.cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-5

    def test_cosine_similarity_different(self):
        emb = Embedder(stub=True)
        v1 = emb.embed("cat")
        v2 = emb.embed("dog")
        sim = Embedder.cosine_similarity(v1, v2)
        assert -1.0 <= sim <= 1.0

    def test_cosine_similarity_empty_returns_zero(self):
        assert Embedder.cosine_similarity([], [1.0, 0.0]) == 0.0
        assert Embedder.cosine_similarity([1.0, 0.0], []) == 0.0

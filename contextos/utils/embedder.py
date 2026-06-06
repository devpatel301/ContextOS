"""
ContextOS — sentence-transformer embedder with a stub fallback.

The stub (embedding_stub=True) uses a deterministic hash to produce
a reproducible fake 384-dimensional embedding. This is useful for:
  - Phase 0/1 unit tests (no GPU / model download needed)
  - CI environments

For real semantic similarity, use the default (embedding_stub=False)
which loads the sentence-transformers model.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np


class Embedder:
    """Wraps sentence-transformers with a deterministic stub fallback."""

    EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dim

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", stub: bool = False):
        self.model_name = model_name
        self.stub = stub
        self._model = None

        if not stub:
            self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for real embeddings. "
                "Install with: pip install sentence-transformers\n"
                "Or use embedding_stub=True for testing without a model."
            ) from e

    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a list[float] of length 384."""
        if self.stub:
            return self._stub_embed(text)

        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. More efficient than calling embed() in a loop."""
        if self.stub:
            return [self._stub_embed(t) for t in texts]

        vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=64)
        return [v.tolist() for v in vecs]

    @classmethod
    def _stub_embed(cls, text: str) -> list[float]:
        """
        Deterministic fake embedding based on sha256 of the text.
        NOT semantically meaningful — only for structural/unit tests.
        Produces the same vector for the same input every time.
        """
        digest = hashlib.sha256(text.encode()).digest()
        # Repeat digest to fill 384 floats (384 * 4 bytes = 1536, digest = 32 bytes)
        raw = (digest * (cls.EMBEDDING_DIM // 8 + 1))[: cls.EMBEDDING_DIM * 4]
        ints = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        ints = ints[: cls.EMBEDDING_DIM]
        # Normalize to unit sphere
        norm = np.linalg.norm(ints)
        if norm < 1e-9:
            return [0.0] * cls.EMBEDDING_DIM
        return (ints / norm).tolist()

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two embedding vectors."""
        if not a or not b:
            return 0.0
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        dot = float(np.dot(va, vb))
        norm = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if norm < 1e-9:
            return 0.0
        return max(-1.0, min(1.0, dot / norm))

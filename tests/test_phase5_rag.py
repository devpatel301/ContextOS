"""
Phase 5 — Smart RAG Integration tests.
"""
from uuid import uuid4

import pytest

from contextos import ContextConfig, ContextOS
from contextos.models.tier import MemoryTier
from contextos.rag.ingest import DocumentChunker


@pytest.fixture
def cos(tmp_path):
    config = ContextConfig(
        db_path=str(tmp_path / "rag.db"),
        chroma_persist_dir=str(tmp_path / ".chroma"),
        embedding_stub=True,  # using stub embedder for fast tests
        token_budget=1000,
        working_memory_reserve=500,
    )
    with ContextOS(config) as c:
        yield c


class TestDocumentChunker:
    def test_chunk_by_words(self):
        # We test the fallback (words) by manually setting encoding to None
        chunker = DocumentChunker(chunk_size=10, chunk_overlap=2)
        chunker.encoding = None
        
        text = "word " * 20
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 0
        assert len(chunks[0].split()) <= int(10 * 0.75)


class TestRetrieverEngine:
    def test_ingest_document(self, cos):
        text = "This is a sentence. " * 50
        blocks = cos.ingest(text, source="test_doc", metadata={"author": "alice"})
        
        assert len(blocks) > 0
        for b in blocks:
            assert b.tier == MemoryTier.SEMANTIC
            assert b.source == "test_doc"
            assert b.metadata.get("author") == "alice"
            assert "chunk_index" in b.metadata
            assert b.embedding is not None
            
        # Verify it went to SQLite store
        assert cos.status()["tiers"]["semantic"]["blocks"] == len(blocks)

    def test_naive_retrieve(self, cos):
        cos.ingest("Apples are red.", source="doc1")
        cos.ingest("Bananas are yellow.", source="doc2")
        
        # Stub embedder just hashes the query and does some determinisic stuff, 
        # so exact or near exact matches might work, but let's just ensure we get results back
        results = cos.naive_retrieve("Apples", top_k=1)
        assert len(results) == 1
        assert isinstance(results[0].content, str)

    def test_smart_retrieve(self, cos):
        cos.ingest("Apples are red.", source="doc1")
        
        # Add some working memory
        cos.store("I like fruit.", tier=MemoryTier.WORKING)
        
        window = cos.retrieve("What color are apples?", budget=500, retrieval_k=2)
        
        # ContextWindow should contain the working memory AND potentially the semantic chunk
        assert window.token_budget == 500
        assert len(window.blocks) >= 1
        
        has_working = any(b.tier == MemoryTier.WORKING for b in window.blocks)
        assert has_working

    def test_sync_all_memories(self, cos):
        # Store directly bypassing ingest
        cos.store("some fact", tier=MemoryTier.SEMANTIC)
        
        # Sync to Chroma
        cos.retriever.sync_all_memories()
        
        # Naive retrieve should find it
        results = cos.naive_retrieve("some fact")
        assert len(results) >= 1

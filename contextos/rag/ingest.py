"""
ContextOS — Document chunking and ingestion.
"""
from __future__ import annotations

import re

# We will use tiktoken for chunking by tokens if available, else fallback to character length
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


class DocumentChunker:
    """
    Simple sliding-window text chunker.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if HAS_TIKTOKEN:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        else:
            self.encoding = None

    def chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        if not text.strip():
            return []

        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        if self.encoding:
            return self._chunk_by_tokens(text)
        else:
            return self._chunk_by_words(text)

    def _chunk_by_tokens(self, text: str) -> list[str]:
        tokens = self.encoding.encode(text)
        chunks = []
        i = 0
        while i < len(tokens):
            end = min(i + self.chunk_size, len(tokens))
            chunk_tokens = tokens[i:end]
            chunks.append(self.encoding.decode(chunk_tokens))
            if end == len(tokens):
                break
            i += (self.chunk_size - self.chunk_overlap)
        return chunks

    def _chunk_by_words(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        i = 0
        # rough approximation: 1 token ~= 0.75 words
        word_chunk_size = int(self.chunk_size * 0.75)
        word_overlap = int(self.chunk_overlap * 0.75)
        
        while i < len(words):
            end = min(i + word_chunk_size, len(words))
            chunks.append(" ".join(words[i:end]))
            if end == len(words):
                break
            i += (word_chunk_size - word_overlap)
        return chunks

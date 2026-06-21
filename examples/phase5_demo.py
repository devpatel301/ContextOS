"""
examples/phase5_demo.py

Demonstrates Phase 5: Smart RAG vs Naive RAG

Run with:
    conda activate denv
    python examples/phase5_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from contextos import ContextOS, ContextConfig
from contextos.models.tier import MemoryTier

print("=" * 60)
print("ContextOS Phase 5 Demo: Smart RAG")
print("=" * 60)

config = ContextConfig(
    db_path="./demo_p5.db",
    chroma_persist_dir="./.chroma_demo",
    embedding_stub=True,  # Fast string-hashing stub for the demo
    token_budget=1000,
    working_memory_reserve=300,
)

cos = ContextOS(config)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Ingestion
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 1. Document Ingestion ---")
doc_text = """
The Python programming language was created by Guido van Rossum and first released in 1991.
It emphasizes code readability with its notable use of significant whitespace.
Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects.

Garbage collection in Python is primarily done through reference counting.
When an object's reference count drops to zero, it is deallocated.
However, reference counting cannot resolve cyclical references (e.g., when object A references object B, and object B references object A).
To solve this, Python has a cyclic garbage collector that periodically scans for and cleans up reference cycles.

The Global Interpreter Lock (GIL) is a mechanism used in CPython.
It prevents multiple native threads from executing Python bytecodes at once.
This is necessary mainly because CPython's memory management is not thread-safe.
While it simplifies implementation and makes single-threaded programs faster, it makes multi-threading for CPU-bound tasks less effective.
"""

# Ingest document
blocks = cos.ingest(doc_text, source="python_docs.txt", metadata={"topic": "Python"})
print(f"  Ingested document into {len(blocks)} chunks (SEMANTIC tier)")

# Let's also add some context to the WORKING and EPISODIC memory
cos.store("System: You are an expert Python tutor.", tier=MemoryTier.WORKING, tags=["system"])
cos.store("User: I have an interview tomorrow for a backend role.", tier=MemoryTier.EPISODIC)
cos.store("User: They use a lot of async Python and multithreading.", tier=MemoryTier.EPISODIC)
print("  Added conversation history (WORKING + EPISODIC tiers)")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Naive Top-K Retrieval
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("2. Naive RAG (Baseline)")
print("─" * 50)

query = "Explain how the GIL affects threading."
print(f"Query: '{query}'")

naive_chunks = cos.naive_retrieve(query, top_k=2)
print(f"\n  [Naive RAG retrieves {len(naive_chunks)} chunks directly from VectorDB]")
for i, chunk in enumerate(naive_chunks):
    print(f"\n  CHUNK {i+1}:")
    print(f"  {chunk.content}")

print("\n  Notice how Naive RAG has no idea about the system prompt or conversation history.")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Smart ContextOS Retrieval
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("3. Smart RAG (ContextOS)")
print("─" * 50)

# Smart retrieve packs the context window with the retrieval candidates + working + episodic
context_window = cos.retrieve(query, budget=500, retrieval_k=2)

print("\n  [Smart RAG retrieves candidates, scores them, and packs a ContextWindow]")
print(f"  Result: {context_window.summary()}")

print("\n--- Final Context Window Prompt ---")
print(context_window.as_prompt())

print("\n  Notice how Smart RAG intelligently blended the retrieved document chunk")
print("  with the system prompt and recent conversation history, fitting perfectly")
print("  into the token budget!")

# Cleanup
cos.close()
import os
import shutil

if os.path.exists("./demo_p5.db"):
    os.remove("./demo_p5.db")
if os.path.exists("./semantic_cache.db"):
    os.remove("./semantic_cache.db")
if os.path.exists("./.chroma_demo"):
    shutil.rmtree("./.chroma_demo")
print("\n✓ Phase 5 demo complete. (demo DBs cleaned up)")

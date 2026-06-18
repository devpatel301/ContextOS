"""
examples/phase4_demo.py

Demonstrates Phase 4: Semantic Cache.

Run with:
    conda activate denv
    python examples/phase4_demo.py

No API keys or GPU required. Uses stub embedder.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from contextos import ContextOS, ContextConfig

print("=" * 60)
print("ContextOS Phase 4 Demo: Semantic Cache")
print("=" * 60)

config = ContextConfig(
    db_path="./demo_p4.db",
    embedding_stub=True,  # Fast string-hashing stub for the demo
    enable_semantic_cache=True,
    cache_similarity_threshold=0.90,
)

cos = ContextOS(config)

def simulate_llm_call(query: str) -> str:
    """Fake an LLM call that takes a little time."""
    print(f"  [NETWORK] Calling LLM for: '{query}'...")
    time.sleep(1.0)  # simulate latency
    return f"This is a generated response for: {query}"


def ask(query: str):
    print(f"\nUser query: '{query}'")
    
    # 1. Check cache first
    start = time.time()
    cached_response = cos.cache_query(query)
    
    if cached_response:
        # Cache hit!
        elapsed = time.time() - start
        print(f"  [CACHE HIT] Response found in {elapsed:.3f}s")
        print(f"  Response: {cached_response}")
    else:
        # Cache miss! Must call LLM.
        response = simulate_llm_call(query)
        
        # Save to cache for next time
        cos.cache_response(query, response)
        
        elapsed = time.time() - start
        print(f"  [CACHE MISS] Completed in {elapsed:.3f}s")
        print(f"  Response: {response}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo Execution
# ─────────────────────────────────────────────────────────────────────────────

# Query 1: First time asking -> Cache Miss
ask("What is a race condition in OS?")

# Query 2: Exact same query -> Cache Hit
ask("What is a race condition in OS?")

# Query 3: Different query -> Cache Miss
ask("Explain virtual memory")

# Query 4: Same as Query 3 -> Cache Hit
ask("Explain virtual memory")

# Query 5: Completely different query -> Cache Miss
ask("How do page faults work?")

print("\n" + "─" * 50)
print("Cache Metrics")
print("─" * 50)
# Look at the underlying SQLite store
all_entries = cos.cache.store.get_all()
print(f"Total entries in cache: {len(all_entries)}")
for e in all_entries:
    print(f"  - '{e.query[:30]}...' -> hits: {e.hit_count}")

cos.close()
print("\n✓ Phase 4 demo complete.")

# Cleanup
import os
if os.path.exists("./demo_p4.db"):
    os.remove("./demo_p4.db")
if os.path.exists("./semantic_cache.db"):
    os.remove("./semantic_cache.db")
    print("  (demo DBs cleaned up)")

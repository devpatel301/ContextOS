"""
ContextOS — Benchmark Runner
"""
import time
import uuid
import os
import shutil

from contextos import ContextOS, ContextConfig
from contextos.models.tier import MemoryTier

def run_benchmark(workload: str):
    print(f"============================================================")
    print(f"ContextOS Benchmark Runner: {workload.upper()}")
    print(f"============================================================")
    
    if workload == "long_conversation":
        _benchmark_long_conversation()
    elif workload == "qa":
        _benchmark_qa()
    elif workload == "repeated_queries":
        _benchmark_cache()
    else:
        print(f"Unknown workload: {workload}")

def _cleanup(db_name: str, chroma_dir: str):
    if os.path.exists(db_name):
        os.remove(db_name)
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)
    if os.path.exists("semantic_cache.db"):
        os.remove("semantic_cache.db")

def _benchmark_long_conversation():
    print("Simulating 100-turn conversation...")
    config = ContextConfig(
        db_path="./bench_lc.db",
        chroma_persist_dir="./.chroma_bench_lc",
        embedding_stub=False,
        token_budget=1000,
        working_memory_reserve=200
    )
    cos = ContextOS(config)
    
    # Store an early fact
    cos.store("My name is Dev and my favourite color is neo-brutalist orange.", tier=MemoryTier.EPISODIC)
    
    # Spam 100 turns
    start_time = time.time()
    for i in range(100):
        cos.store(f"User: Turn {i} filler text about random things. Agent: That is interesting.", tier=MemoryTier.EPISODIC)
        
    # Attempt to retrieve early fact
    win = cos.retrieve("What is my name and favourite color?", budget=1000)
    latency = time.time() - start_time
    
    # Check if fact was retained
    fact_retained = any("neo-brutalist orange" in b.content for b in win.blocks)
    
    status = cos.status()
    print(f"\n[Results]")
    print(f"Total Blocks Stored: 101")
    print(f"Blocks Evicted to Archive: {status['tiers']['archived']['blocks']}")
    print(f"Fact Retained in Context Window: {'Yes' if fact_retained else 'No'}")
    print(f"Total time for 100 turns + retrieval: {latency:.2f}s")
    print(f"Tokens packed perfectly in budget: {win.tokens_used}/{win.token_budget}")
    
    cos.close()
    _cleanup("./bench_lc.db", "./.chroma_bench_lc")

def _benchmark_qa():
    print("Simulating Document QA (50 docs)...")
    config = ContextConfig(
        db_path="./bench_qa.db",
        chroma_persist_dir="./.chroma_bench_qa",
        embedding_stub=False,
        token_budget=1000,
        working_memory_reserve=200
    )
    cos = ContextOS(config)
    
    start_time = time.time()
    for i in range(50):
        text = f"This is document {i}. It talks about subject_{i}. The capital of country_{i} is city_{i}."
        cos.ingest(text, source=f"doc_{i}")
        
    win = cos.retrieve("What is the capital of country_42?", budget=1000)
    latency = time.time() - start_time
    
    found = any("country_42" in b.content for b in win.blocks)
    
    print(f"\n[Results]")
    print(f"Documents Ingested: 50")
    print(f"Target found in Context Window: {'Yes' if found else 'No'}")
    print(f"Time: {latency:.2f}s")
    
    cos.close()
    _cleanup("./bench_qa.db", "./.chroma_bench_qa")

def _benchmark_cache():
    print("Simulating Semantic Cache Hits...")
    config = ContextConfig(
        db_path="./bench_cache.db",
        chroma_persist_dir="./.chroma_bench_cache",
        embedding_stub=False,
        enable_semantic_cache=True,
        token_budget=2000,
        working_memory_reserve=500
    )
    cos = ContextOS(config)
    
    # Populate
    cos.cache_response("How do I reverse a string in Python?", "Use slicing: s[::-1]")
    
    # Query exact
    t0 = time.time()
    hit1 = cos.cache_query("How do I reverse a string in Python?")
    t1 = time.time()
    
    print(f"\n[Results]")
    print(f"Cache Hit 1 (Exact): {'Yes' if hit1 else 'No'} in {(t1-t0)*1000:.2f}ms")
    
    cos.close()
    _cleanup("./bench_cache.db", "./.chroma_bench_cache")

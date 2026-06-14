"""
examples/phase2_3_demo.py

Demonstrates Phase 2 (Context Scheduler) and Phase 3 (Eviction + GC).

Run with:
    conda activate denv
    python examples/phase2_3_demo.py

No API keys or GPU required. Uses stub embedder.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from contextos import ContextOS, ContextConfig, MemoryTier

print("=" * 60)
print("ContextOS Phase 2 + 3 Demo")
print("  • Phase 2: Context Scheduler + allocate()")
print("  • Phase 3: Eviction Policies + Garbage Collector")
print("=" * 60)

config = ContextConfig(
    db_path="./demo_p23.db",
    embedding_stub=True,
    llm_provider="none",
    token_budget=200,              # Small budget to see scheduling choices
    working_memory_reserve=50,     # Must be < token_budget
    episodic_tier_budget=150,      # Triggers eviction when exceeded
    scheduling_policy="weighted_fair",
    eviction_policy="lru",
)

cos = ContextOS(config)

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Context Scheduling
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("PHASE 2: Context Scheduling")
print("─" * 50)

# Store a system prompt in WORKING tier (always included)
cos.store(
    "You are a helpful OS tutoring assistant. Answer concisely.",
    tier=MemoryTier.WORKING,
    tags=["system"],
)

# Store several conversation memories
memories = [
    "User asked about process scheduling algorithms in operating systems.",
    "User is interested in the difference between preemptive and cooperative scheduling.",
    "User mentioned they are preparing for a systems programming interview.",
    "The round-robin algorithm assigns each process a fixed time quantum.",
    "Context switching has overhead: saving/restoring registers, TLB flush.",
    "User prefers code examples in Python when explaining concepts.",
    "Deadlocks occur when four conditions hold: mutual exclusion, hold-and-wait, no preemption, circular wait.",
    "User asked about memory-mapped I/O versus port-mapped I/O.",
    "Virtual memory uses page tables to map virtual addresses to physical frames.",
    "The LRU page replacement algorithm evicts the least recently used page.",
]

for mem in memories:
    cos.store(mem, tier=MemoryTier.EPISODIC)

print(f"\nStored {len(memories)} memories + 1 system prompt")
print(f"Status: {cos.status()}")

# ── allocate() with a query ──────────────────────────────────────────────────
print("\n--- Allocating context window (budget=200 tokens) ---")
print("    Query: 'Explain LRU page replacement'")

ctx = cos.allocate(budget=200, query="Explain LRU page replacement")

print(f"\n  Result: {ctx.summary()}")
print(f"  Blocks included: {len(ctx.blocks)}")
print(f"  Token utilization: {ctx.utilization():.0%}")
print(f"  Token usage by tier: {ctx.token_usage_by_tier()}")

print("\n  Blocks in context window:")
for b in ctx.blocks:
    label = f"[{b.tier.value:8s}|score={b.importance_score:.2f}]"
    print(f"    {label} {b.content[:65]}...")

# ── Show the formatted prompt ────────────────────────────────────────────────
print("\n--- Formatted prompt (what gets sent to LLM) ---")
prompt = ctx.as_prompt()
print(prompt)

# ── Try a different scheduling policy ─────────────────────────────────────────
print("\n--- Switching to FIFO scheduling policy ---")
from contextos.scheduler.engine import ContextScheduler
cos.scheduler = ContextScheduler(policy="fifo")

ctx_fifo = cos.allocate(budget=200, query="Explain LRU page replacement")
print(f"  FIFO result: {ctx_fifo.summary()}")
print(f"  First block: {ctx_fifo.blocks[0].content[:60]}..." if ctx_fifo.blocks else "  (empty)")

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Eviction
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("PHASE 3: Eviction + Garbage Collection")
print("─" * 50)

# Show current episodic usage
ep_tokens = cos.status()["tiers"]["episodic"]["tokens"]
print(f"\nEpisodic tier: {cos.status()['tiers']['episodic']['blocks']} blocks, {ep_tokens} tokens")
print(f"Episodic budget: {config.episodic_tier_budget} tokens")

if ep_tokens > config.episodic_tier_budget:
    print(f"\n  ⚠  Over budget by {ep_tokens - config.episodic_tier_budget} tokens!")
    print("  Running eviction (LRU policy)...")
    evicted = cos.evict(tier=MemoryTier.EPISODIC)
    print(f"  Evicted {len(evicted)} blocks:")
    for v in evicted:
        print(f"    → {v.content[:60]}...")
    ep_after = cos.status()["tiers"]["episodic"]["tokens"]
    print(f"  Episodic tier now: {ep_after} tokens (was {ep_tokens})")
else:
    print("  ✓ Under budget, no eviction needed.")

# ── Garbage Collector ─────────────────────────────────────────────────────────
print("\n--- Garbage Collector ---")

# Create some duplicate memories to test dedup
cos.store("User prefers Python for scripting.")
cos.store("User prefers Python for scripting.")  # exact duplicate
cos.store("User prefers Python for scripting.")  # another duplicate

print(f"  Before GC: {cos.status()['total_blocks']} total blocks")

report = cos.gc_run()
print(f"  GC report: {report}")
print(f"  After GC:  {cos.status()['total_blocks']} total blocks")

# ── Final status ──────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("Final Memory Status")
print("─" * 50)
status = cos.status()
print(f"  Total blocks: {status['total_blocks']}")
for tier_name, info in status["tiers"].items():
    if info["blocks"] > 0:
        print(f"  {tier_name:10s}: {info['blocks']} blocks, {info['tokens']} tokens")

cos.close()
print("\n✓ Phase 2+3 demo complete.")

# Cleanup
import os
if os.path.exists("./demo_p23.db"):
    os.remove("./demo_p23.db")
    print("  (demo_p23.db cleaned up)")

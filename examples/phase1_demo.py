"""
examples/phase1_demo.py

A runnable walkthrough of the Phase 1 ContextOS API.

Run with:
    python examples/phase1_demo.py

No API keys or GPU required. Uses stub embedder.
"""
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from contextos import ContextOS, ContextConfig, MemoryTier

# ─────────────────────────────────────────────────────────────────────────────
# 1. Initialise ContextOS with stub embedder (no GPU / model download)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("ContextOS Phase 1 Demo")
print("=" * 60)

config = ContextConfig(
    db_path="./demo.db",          # Will create this file in the project root
    embedding_stub=True,          # Fast deterministic embeddings, no GPU needed
    llm_provider="none",          # No LLM calls in Phase 1
)

cos = ContextOS(config)
print(f"\nInitialised: {cos}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Store some memories
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 40)
print("Storing memories...")
print("─" * 40)

# User profile — pinned so it never gets evicted
profile = cos.store(
    "The user's name is Alex. They are a CS student interested in systems programming.",
    tier=MemoryTier.EPISODIC,
    tags=["user_profile"],
    user_weight=1.0,          # pinned
    source="user",
)
print(f"  Stored profile:  {profile}")

# Some conversation context
ctx1 = cos.store(
    "User asked about virtual memory and got a detailed explanation of page tables.",
    tier=MemoryTier.EPISODIC,
    tags=["conversation"],
)
ctx2 = cos.store(
    "User mentioned they prefer Python for scripting but uses C for systems work.",
    tier=MemoryTier.EPISODIC,
    tags=["conversation", "preference"],
)

# A fact to go into semantic memory
fact = cos.store(
    "TCP (Transmission Control Protocol) provides reliable, ordered delivery of data.",
    tier=MemoryTier.SEMANTIC,
    tags=["tcp", "networking"],
)

print(f"  Stored context1: {ctx1}")
print(f"  Stored context2: {ctx2}")
print(f"  Stored fact:     {fact}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Status snapshot
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 40)
print("Memory status:")
print("─" * 40)
status = cos.status()
print(f"  Total blocks: {status['total_blocks']}")
for tier_name, info in status["tiers"].items():
    if info["blocks"] > 0:
        print(f"  {tier_name:10s}: {info['blocks']} blocks, {info['tokens']} tokens")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Touch (record access) — simulates the scheduler using a block
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 40)
print("Simulating access pattern (touch × 3 on ctx1)...")
print("─" * 40)
for _ in range(3):
    cos.touch(ctx1.id)
updated = cos.get(ctx1.id)
print(f"  ctx1.frequency is now: {updated.frequency}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Tier transitions — ctx1 has been accessed 3 times → EPISODIC → SEMANTIC
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 40)
print("Running tier transitions...")
print("─" * 40)
counts = cos.run_transitions()
print(f"  Transitions: {counts}")

updated_ctx1 = cos.get(ctx1.id)
print(f"  ctx1 tier is now: {updated_ctx1.tier.value}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Manual promote / demote
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 40)
print("Manual tier operations...")
print("─" * 40)

# Promote fact from semantic to episodic (page it in)
cos.promote(fact.id, MemoryTier.EPISODIC)
print(f"  Promoted fact to: {cos.get(fact.id).tier.value}")

# Try to demote pinned profile (should fail)
success = cos.demote(profile.id, MemoryTier.SEMANTIC)
print(f"  Demote pinned profile: success={success} (expected False)")
print(f"  Profile tier: {cos.get(profile.id).tier.value} (unchanged)")

# ─────────────────────────────────────────────────────────────────────────────
# 7. List all blocks in episodic tier
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 40)
print("Episodic tier blocks:")
print("─" * 40)
episodic = cos.get_tier(MemoryTier.EPISODIC)
for b in episodic:
    print(f"  [{b.importance_score:.2f}] {b.content[:60]}...")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Store with expiry
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 40)
print("Expiring memory demo...")
print("─" * 40)
temp = cos.store(
    "This is a temporary note that expires in 1 hour.",
    tags=["temp"],
    expires_in_hours=1,
)
print(f"  Stored temporary block. Expires at: {temp.expires_at}")
print(f"  Is expired now? {temp.is_expired()}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Final status
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 40)
print("Final memory status:")
print("─" * 40)
status = cos.status()
print(f"  Total blocks: {status['total_blocks']}")
for tier_name, info in status["tiers"].items():
    print(f"  {tier_name:10s}: {info['blocks']} blocks, {info['tokens']} tokens")

# Cleanup
cos.close()
print("\n✓ Demo complete. Check demo.db to inspect the SQLite database.")

# Clean up demo DB
# import os
# if os.path.exists("./demo.db"):
#     os.remove("./demo.db")
#     print("  (demo.db cleaned up)")

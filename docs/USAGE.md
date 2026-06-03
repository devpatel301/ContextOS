# ContextOS — User Guide

> How a developer actually uses ContextOS. What to import, what to call, what to expect.

---

## The Core Idea

You never touch prompts manually again.

```python
# Before ContextOS (everyone does this):
messages = conversation_history[-20:]  # fingers crossed nothing important got cut
response = llm.chat(messages)

# With ContextOS:
context = cos.allocate(budget=16_000, query=user_message)
response = llm.chat(context.as_prompt())
```

ContextOS decides what goes into that 16,000 token window. It scores, ranks, compresses, caches, and pages memory — you just call `allocate()`.

---

## Installation

```bash
# Clone and set up
git clone https://github.com/yourusername/ContextOS
cd ContextOS

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Phase 0/1 only needs:** `tiktoken numpy pytest python-dotenv`
> Full stack (sentence-transformers, chromadb, fastapi) needed from Phase 2+

---

## Phase 1 Usage: Memory Management

Phase 1 gives you a fully working, persistent memory hierarchy.
No LLM calls, no API keys. Just memory in, memory out.

### Quickstart

```python
from contextos import ContextOS, ContextConfig, MemoryTier

# Minimal setup — uses SQLite at ./contextos.db, stub embedder (no GPU)
cos = ContextOS(ContextConfig(embedding_stub=True))

# Store memories
cos.store("The user's name is Alex")
cos.store("Alex is interested in Linux kernel development")

# Check what's in memory
status = cos.status()
print(status)
# {'total_blocks': 2, 'tiers': {'episodic': {'blocks': 2, 'tokens': 14}, ...}}

cos.close()
```

### Storing Memories

```python
# Basic store — goes to EPISODIC by default
block = cos.store("User asked about virtual memory paging")

# Store with tags for filtering later
block = cos.store(
    "User prefers Python over JavaScript",
    tags=["user_preference", "language"],
)

# Pin a memory so it's NEVER evicted (user_weight=1.0)
profile = cos.store(
    "User is Alex, a CS student at IIT Delhi.",
    tags=["user_profile"],
    user_weight=1.0,      # pinned = never evicted
)

# Store in a specific tier
cos.store(
    "TCP provides reliable, ordered data delivery",
    tier=MemoryTier.SEMANTIC,   # facts go to semantic
    tags=["networking"],
)

# Store with automatic expiry
cos.store(
    "User is currently debugging a segfault",
    tags=["current_task"],
    expires_in_hours=2,   # auto-deleted after 2 hours
)

# Store with custom metadata
cos.store(
    "The `fork()` system call creates a child process",
    source="document",
    metadata={"doc": "linux_kernel_book.pdf", "page": 142},
)
```

### Retrieving Memories

```python
# Retrieve by ID (returns MemoryBlock or None)
block = cos.get(block.id)
print(block.content)
print(block.tier)
print(block.importance_score)

# List all blocks in a tier (sorted by importance, highest first)
episodic_memories = cos.get_tier(MemoryTier.EPISODIC)
for b in episodic_memories:
    print(f"[{b.importance_score:.2f}] {b.content[:80]}")

# Get everything
all_blocks = cos.get_all()
```

### Recording Access (Touch)

When the scheduler includes a block in a context window, tell ContextOS:

```python
# This increments frequency and updates last_accessed
cos.touch(block.id)

# After 3 touches, the block is a candidate for EPISODIC → SEMANTIC promotion
```

### Manual Tier Transitions

```python
# Promote: move to a HOTTER tier (e.g., ARCHIVED → EPISODIC when re-retrieved)
cos.promote(fact_block.id, MemoryTier.EPISODIC)

# Demote: move to a COLDER tier (e.g., EPISODIC → SEMANTIC when compressing)
cos.demote(old_conversation.id, MemoryTier.SEMANTIC)

# Pinned blocks (user_weight=1.0) cannot be demoted — returns False
success = cos.demote(pinned_block.id, MemoryTier.ARCHIVED)  # → False
```

### Automatic Tier Transitions

Call this periodically (e.g., at the end of each turn) to apply the OS-style rules:

```python
counts = cos.run_transitions()
# Example output:
# {'episodic_to_semantic': 2, 'semantic_to_archived': 0, 'aged_to_archived': 0}
```

**The rules it applies:**
1. EPISODIC → SEMANTIC if a block has been accessed ≥ 3 times
2. SEMANTIC → ARCHIVED if idle for > 24 hours
3. Any tier → ARCHIVED if block is older than 7 days (except WORKING and pinned)

### Memory Status Snapshot

```python
status = cos.status()

# Output:
# {
#   "total_blocks": 47,
#   "tiers": {
#     "working":  {"blocks": 2,  "tokens": 1240},
#     "episodic": {"blocks": 18, "tokens": 6200},
#     "semantic": {"blocks": 22, "tokens": 8100},
#     "archived": {"blocks": 5,  "tokens": 1800},
#   }
# }
```

---

## Phase 2 Usage: Context Scheduling

Phase 2 introduces `allocate()`, which replaces manual prompt assembly. It gathers the most important memories and packs them into a budget-respecting `ContextWindow`.

### Allocating a Context Window

```python
# Tell ContextOS to pack a 4000-token prompt for this query
context = cos.allocate(budget=4000, query="Explain LRU page replacement")

# Inspect what it chose
print(context.summary())
# "ContextWindow(3850/4000 tokens, blocks=12, [working=250, episodic=1800, semantic=1800])"

# See exactly which blocks were selected
for b in context.blocks:
    print(f"[{b.tier.value}] {b.content[:50]}...")
```

### Generating the Final Prompt

You don't need to manually string blocks together. Just call `.as_prompt()`:

```python
prompt = context.as_prompt()

# Send directly to your LLM
response = llm.chat(prompt)
```

### Scheduling Policies

You can change how blocks are prioritized via `ContextConfig.scheduling_policy`:
- **fifo**: Oldest first.
- **priority**: Pure greedy based on importance score.
- **round_robin**: Interleaves memories from each tier.
- **weighted_fair** (default): Guarantees proportional token budget to each tier before filling with highest-scoring blocks.

---

## Phase 3 Usage: Eviction & Garbage Collection

Phase 3 keeps your database fast and prevents token usage from exploding by evicting low-value memories and deduplicating.

### Eviction Policies

When a tier (like EPISODIC) gets too large, ContextOS evicts blocks.
Available policies: `fifo`, `lru`, `lfu`, `priority`, `hybrid` (default).

```python
# Evict from EPISODIC until it's under its configured token budget
evicted = cos.evict(tier=MemoryTier.EPISODIC)

# You can also manually specify a target budget
cos.evict(tier=MemoryTier.SEMANTIC, budget=10_000)
```
*Note: `WORKING` tier and blocks with `user_weight=1.0` (pinned) are never evicted.*

### Garbage Collection

Run GC periodically to keep the memory store clean:

```python
report = cos.gc_run()
print(report)
# {'expired': 2, 'deduplicated': 1, 'pruned': 0, 'total_freed': 3}
```

**GC passes:**
1. **Expiry**: Removes blocks past their `expires_at`.
2. **Deduplication**: Removes blocks with `cosine_similarity > 0.92`.
3. **Pruning**: Deletes very old, very low-score blocks from the `ARCHIVED` tier.

---

## Phase 4 Usage: Semantic Cache

Phase 4 allows you to skip expensive LLM calls if the user asks a question that is semantically similar to one you've answered before.

```python
query = "What is a race condition in OS?"

# 1. Check the cache
cached_response = cos.cache_query(query)

if cached_response:
    # 2a. Cache HIT — return immediately
    print(cached_response)
else:
    # 2b. Cache MISS — call your LLM
    response = llm.chat(query)
    
    # 3. Store the response for next time
    cos.cache_response(query, response)
```

The threshold for what counts as a "hit" is configured via `ContextConfig.cache_similarity_threshold` (default is 0.92 cosine similarity). 

---

## Phase 5 Usage: Smart RAG

Traditional RAG just retrieves the top 5 chunks of a document and blindly passes them to the LLM, destroying conversation history and blowing up the token budget. 

ContextOS Smart RAG retrieves candidates from your documents via a built-in VectorDB (Chroma) and then passes them to the **Context Scheduler**, which elegantly mixes them with your `WORKING` memory (system prompts) and `EPISODIC` memory (conversation history).

### 1. Ingestion
```python
# Pass raw text — ContextOS chunks it, embeds it, and stores it in the SEMANTIC tier
blocks = cos.ingest(
    "Long document text here...", 
    source="my_document.pdf",
    metadata={"author": "Alex"}
)
```

### 2. Smart Retrieval
```python
# Generates a perfectly balanced prompt containing the system prompt, 
# recent conversation history, AND the most relevant document chunks, 
# all guaranteed to fit in your budget!
context_window = cos.retrieve(query="What does the document say about X?", budget=4000)

prompt = context_window.as_prompt()
```

*(Note: If you want to see the difference yourself, you can call `cos.naive_retrieve(query)` to see what a standard RAG pipeline would have done).*

---

## Phase 6 Usage: Observability Dashboard

ContextOS does a lot of work under the hood. Phase 6 gives you an "htop for LLMs" — a live visual profiler built with FastAPI and inspired by neo-brutalism.

To run the dashboard:
```bash
uvicorn contextos.api.main:app --reload
```
Then navigate to `http://127.0.0.1:8000` in your browser.

The dashboard displays:
- **Memory Tiers**: Block counts and token usage across `WORKING`, `EPISODIC`, `SEMANTIC`, and `ARCHIVED`.
- **Token Budget**: A live progress bar showing how much of your token budget is currently consumed.
- **Context Scheduler Console**: A terminal where you can simulate a query and instantly see exactly how the `ContextScheduler` packs the context window, scoring and prioritizing chunks based on budget constraints.
- **Semantic Cache**: Real-time hit/miss metrics and simulated LLM network latency comparisons.

---

## Phase 7 Usage: Agent Runtime

ContextOS isn't just a RAG pipeline; it's designed to be the memory substrate for autonomous agents. In Phase 7, we provide an `AgentRuntime` that executes loops, calls tools, and automatically stores its thoughts and actions in the `EPISODIC` memory tier.

```python
from contextos import ContextOS, ContextConfig
from contextos.agent import AgentRuntime, StubLLM, get_default_tools

config = ContextConfig(token_budget=4000)
cos = ContextOS(config)

# Initialize the runtime with the ContextOS instance
agent = AgentRuntime(
    cos=cos, 
    llm=StubLLM(), 
    tools=get_default_tools(), 
    max_steps=10
)

# Run a task
final_answer = agent.run("Find out the weather in London and calculate 25 * 4.")

# Under the hood, ContextOS seamlessly managed the context window, 
# evicted old thoughts, and kept the token budget safe!
```

---

## Multi-Agent Support & Context Version Control

ContextOS natively supports multi-agent setups. By passing an `agent_id` into the configuration, ContextOS will automatically namespace and sandbox the SQLite database and ChromaDB collections for that specific agent.

ContextOS also provides `git`-like version control for your agent's memory state!

```python
from contextos import ContextOS, ContextConfig

# Sandboxing agent memory
config = ContextConfig(agent_id="research_agent_01")
cos = ContextOS(config)

cos.store("Important research fact")

# Snapshot the memory state before doing risky tasks
cos.commit(tag="pre_experiment")

# Agent makes mistakes and fills memory with noise
cos.store("Bad reasoning step")
cos.store("Hallucinated fact")

# Revert to the snapshot!
cos.checkout(tag="pre_experiment")
# Memory is completely restored to the clean state!
```

---

## Understanding MemoryBlock

Every piece of information in ContextOS is a `MemoryBlock`:

```python
block = cos.store("User asked about TCP")

block.id              # UUID — unique identifier
block.content         # "User asked about TCP"
block.summary         # None (Phase 4 will add LLM compression here)
block.tier            # MemoryTier.EPISODIC
block.importance_score # 0.35 — composite score, higher = more likely to be included
block.recency_score   # 0.98 — decays over time (1.0 = just created)
block.frequency       # 0 — how many times touched
block.user_weight     # 0.5 — 1.0 = pinned
block.token_count     # 5 — tiktoken count of content
block.created_at      # datetime
block.last_accessed   # datetime
block.expires_at      # None (or datetime if expires_in_hours was set)
block.source          # "user"
block.tags            # []
block.metadata        # {}
block.embedding       # list[float] of length 384 (sentence-transformer vector)

# Helpers
block.effective_content()      # returns summary if set, else content
block.effective_token_count()  # returns summary token count if compressed
block.is_expired()             # True if past expires_at
block.is_pinned()              # True if user_weight=1.0 or 'pinned' in tags
block.touch()                  # increment frequency, update last_accessed (in-memory only)
```

---

## Configuration Reference

```python
from contextos import ContextConfig

config = ContextConfig(
    # Storage
    db_path="./my_agent.db",       # SQLite file location

    # Embeddings (choose one)
    embedding_stub=True,           # Fast fake embedder — no GPU, for testing
    embedding_stub=False,          # Real sentence-transformers (default)
    embedding_model="all-MiniLM-L6-v2",  # Which model to use

    # Token budget (Phase 2+)
    token_budget=16_000,
    working_tier_budget=4_000,
    episodic_tier_budget=8_000,

    # Tier transition thresholds
    episodic_to_semantic_access_count=3,   # touches before EPISODIC → SEMANTIC
    semantic_to_archived_idle_hours=24,    # hours idle before SEMANTIC → ARCHIVED
    archived_age_days=7,                   # days old before → ARCHIVED

    # Importance scoring weights (must sum to 1.0)
    score_weight_similarity=0.40,
    score_weight_recency=0.30,
    score_weight_frequency=0.20,
    score_weight_user=0.10,
    recency_half_life_hours=24.0,

    # LLM (Phase 2+)
    llm_provider="openai",         # "openai" | "ollama" | "none"
    llm_model="gpt-4o-mini",
    # Set OPENAI_API_KEY in .env file

    # Features (Phase 2+)
    enable_semantic_cache=True,
    cache_similarity_threshold=0.92,
    enable_gc=True,
    enable_profiler=True,
)
```

---

## Memory Tier Reference

| Tier | OS Analogy | Token Budget | Always In Context? | Evictable? |
|---|---|---|---|---|
| `WORKING` | CPU registers | ~2,000–4,000 | ✅ Yes | Never |
| `EPISODIC` | RAM | ~4,000–8,000 | If budget allows | Yes (LRU/LFU/etc.) |
| `SEMANTIC` | SSD | Unlimited in DB | Retrieved on demand | Yes |
| `ARCHIVED` | HDD | Unlimited in DB | Paged in on retrieval | Yes |

**When to use each tier:**
- `WORKING` → System prompt, current user message, tool schemas
- `EPISODIC` → Recent conversation turns (last 5–10 turns)
- `SEMANTIC` → Facts, preferences, compressed summaries
- `ARCHIVED` → Cold storage, old documents, rarely-accessed facts

---

## Context Manager (Recommended Pattern)

```python
with ContextOS(config) as cos:
    cos.store("user message here")
    # ...do work...
# DB connection auto-closed on exit
```

---

## Running the Demos

```bash
source .venv/bin/activate

# Phase 1: Storage and Tiers
python examples/phase1_demo.py

# Phase 2 & 3: Scheduling, Eviction, GC
python examples/phase2_3_demo.py

# Phase 4: Semantic Cache
python examples/phase4_demo.py

# Phase 5: Smart RAG vs Naive RAG
python examples/phase5_demo.py

# Phase 6: Visual Dashboard
uvicorn contextos.api.main:app --reload

# Phase 7: Agent Runtime Loop
python examples/phase7_demo.py

# Phase 8: Benchmarks
python -m contextos.benchmark run --workload long_conversation
python -m contextos.benchmark run --workload qa
python -m contextos.benchmark run --workload repeated_queries
```

Expected output for Phase 2/3:
```
============================================================
ContextOS Phase 2 + 3 Demo
  • Phase 2: Context Scheduler + allocate()
  • Phase 3: Eviction Policies + Garbage Collector
============================================================

Initialised: ContextOS(blocks=0, db=./demo.db, stub=True)

────────────────────────────────────────
Storing memories...
────────────────────────────────────────
  Stored profile:  MemoryBlock(id=3fa8b2e0…, tier=episodic, score=0.40, ...)
  ...

Memory status:
  Total blocks: 4
  episodic  : 3 blocks, 29 tokens
  semantic  : 1 blocks, 9 tokens
...
```

## Conclusion

ContextOS is complete! It features comprehensive memory management for autonomous agents, complete with smart RAG, multi-tier scheduling, garbage collection, semantic caching, visual observability, sandboxed multi-agent support, and `git`-style context versioning.

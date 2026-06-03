# ContextOS — Development Roadmap

> A phase-by-phase build plan. Each phase is independently useful, shippable, and demonstrable.

---

## Overview

```
Phase 0  →  Foundation & Design
Phase 1  →  Core Memory Manager + Hierarchy
Phase 2  →  Context Scheduler + Token Budget
Phase 3  →  Eviction Policies + Garbage Collector
Phase 4  →  Semantic Cache
Phase 5  →  Smart RAG Integration
Phase 6  →  Observability Dashboard
Phase 7  →  Agent Runtime + Multi-step Workflows
Phase 8  →  Benchmark Harness + Evaluation
Phase 9  →  Advanced Extensions
```

Each phase has a **goal**, **deliverables**, **acceptance criteria**, and **resume value**.

---

## Phase 0: Foundation & Design (3–5 days)

**Goal:** Lock the architecture before writing a single line of production code.

### Deliverables
- [ ] Finalize documentation suite (README, ARCHITECTURE, FEATURES, EVALUATION, EXTENSIONS)
- [ ] Design core data models: `Memory`, `MemoryBlock`, `MemoryTier`, `ContextWindow`
- [ ] Define public API surface: `allocate()`, `store()`, `evict()`, `ingest()`
- [ ] Set up project scaffolding: directory layout, venv, linting, pre-commit hooks
- [ ] Initialize SQLite schema for persistent memory storage
- [ ] Write stub tests for every component (test-first stubs)

### Key Data Model

```python
@dataclass
class MemoryBlock:
    id: UUID
    content: str                      # raw text
    summary: str | None               # compressed version
    embedding: list[float]            # sentence-transformer vector
    tier: MemoryTier                  # WORKING | EPISODIC | SEMANTIC | ARCHIVED
    importance_score: float           # computed composite
    recency_score: float              # time-decayed
    frequency: int                    # access count
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime | None
    tags: list[str]
    metadata: dict
    token_count: int                  # tiktoken-computed
```

### Acceptance Criteria
- `pytest` passes (all stubs trivially)
- Schema documented and reviewed
- Every component has an abstract interface before implementation

---

## Phase 1: Core Memory Manager + Hierarchy (1–2 weeks)

**Goal:** Working in-memory + persistent memory system with tier management.

### Deliverables
- [ ] `MemoryTier` enum: WORKING, EPISODIC, SEMANTIC, ARCHIVED
- [ ] `MemoryStore` — SQLite-backed CRUD layer
- [ ] `MemoryManager` — tier promotion, demotion, retrieval
- [ ] Composite importance scoring
- [ ] Token counting via `tiktoken` on every block
- [ ] Unit tests for all tier transitions

### Importance Score Formula

```
score = 0.40 * similarity
      + 0.30 * recency        # exponential time decay
      + 0.20 * log(frequency)
      + 0.10 * user_weight
```

### Tier Transition Rules

| Trigger | Transition |
|---|---|
| Accessed > 3 times | EPISODIC → SEMANTIC |
| Not accessed in 24h | SEMANTIC → ARCHIVED |
| Block age > 7 days | Any → ARCHIVED |
| Accessed this turn | ARCHIVED → EPISODIC (paged in) |
| Token budget pressure | EPISODIC → compress → SEMANTIC |

### Acceptance Criteria
- 1000-block store with no degradation
- Tier transitions fire correctly on simulated time-lapse
- Token budget of any tier is computable on demand

---

## Phase 2: Context Scheduler + Token Budget (1–2 weeks)

**Goal:** A scheduler that receives a token budget and returns an optimally-packed context window.

### Deliverables
- [x] `ContextScheduler` with pluggable algorithm interface
- [x] Initial algorithms: FIFO, Priority, Round Robin, Weighted Fair
- [x] `ContextWindow` dataclass (ordered blocks + token count)
- [x] `ContextOS.allocate(budget, query)` — the main public API
- [x] Integration test: end-to-end `allocate()` produces a valid context window

### Plugin Interface

```python
class SchedulingPolicy(ABC):
    @abstractmethod
    def schedule(
        self,
        candidates: list[MemoryBlock],
        budget: int,
        query_embedding: list[float],
    ) -> list[MemoryBlock]: ...
```

### Context Window Output Format

```
[WORKING MEMORY — current turn]
...

[RECENT CONTEXT — last 5 turns]
...

[RETRIEVED MEMORIES — semantic search]
...

[SYSTEM INSTRUCTIONS]
...

Total: 14,230 / 16,000 tokens
```

### Acceptance Criteria
- `allocate(budget=4096, query="explain TCP")` returns valid `ContextWindow`
- Scheduler never exceeds token budget
- `allocate()` latency < 200ms on 1000-block store

---

## Phase 3: Eviction Policies + Garbage Collector (1 week)

**Goal:** When memory is full, evict intelligently. GC removes garbage proactively.

### Eviction Policies
- **FIFO** — evict oldest
- **LRU** — evict least recently used
- **LFU** — evict least frequently used
- **Priority** — evict lowest importance score
- **Hybrid** — weighted combination (recency primary, frequency tiebreak)

### Plugin Interface

```python
class EvictionPolicy(ABC):
    @abstractmethod
    def select_victims(
        self,
        blocks: list[MemoryBlock],
        tokens_to_free: int,
    ) -> list[MemoryBlock]: ...
```

### Garbage Collector Features
- **Deduplication**: remove blocks where `cosine_similarity > 0.92`
- **Staleness expiry**: remove blocks past `expires_at`
- **Semantic merging**: merge near-duplicates into single canonical block

### Acceptance Criteria
- All 5 policies pass unit tests
- GC deduplicates a store seeded with 20% near-duplicates
- Eviction never removes a block in a live context window
- GC is idempotent

---

## Phase 4: Semantic Cache (1 week)

**Goal:** Cache LLM responses and serve them on semantically-similar future queries.

### Deliverables
- [x] `CacheEntry` dataclass
- [x] `CacheStore` for SQLite persistence
- [x] `SemanticCache` engine (similarity threshold matching)
- [x] `cos.cache_query()` and `cos.cache_response()` API
- [x] Integration with test suite and demo script

### How It Works
1. Embed incoming query
2. Search cache for embedding with similarity > threshold (default 0.92)
3. **Cache hit**: return stored response (no LLM call)
4. **Cache miss**: call LLM, store `(query_embedding, response)`, return

### Cache Entry Schema

```python
@dataclass
class CacheEntry:
    id: UUID
    query: str
    query_embedding: list[float]
    response: str
    response_tokens: int
    created_at: datetime
    expires_at: datetime
    hit_count: int
    tags: list[str]
```

### Acceptance Criteria
- `"Explain TCP"` → LLM call (miss, stored)
- `"Can you explain TCP again?"` → cache hit, no LLM call
- `"Explain UDP"` → correctly rejected (miss)
- Hit rate > 60% on synthetic repeated-query benchmark
- Cost savings are measurable

---

## Phase 5: Smart RAG Integration (1–2 weeks)

**Goal:** Replace naive top-k retrieval with a pipeline that goes through the Context Scheduler.

### Deliverables
- [x] `DocumentChunker` using `tiktoken`
- [x] `VectorIndex` wrapper around ChromaDB
- [x] `RetrieverEngine` coordinating ingestion and retrieval
- [x] `cos.ingest(text)` and `cos.retrieve(query)` APIs
- [x] `cos.naive_retrieve(query)` for baseline comparison
- [x] Tests and demo script showing smart vs naive RAG

### Retrieval Pipeline

```
User Query
  → Query Expansion (optional LLM rewrite)
  → BM25 + Dense Vector Search (ChromaDB)
  → Candidate Pool (top-50)
  → Importance Reranking
  → Context Scheduler (token budget fit)
  → Memory Compression (if needed)
  → Context Window → LLM
```

### Adaptive Strategy Selection

| Query Type | Strategy |
|---|---|
| Factual | Dense vector similarity |
| Keyword-heavy | BM25 |
| Conversational | Recency-weighted |

### Acceptance Criteria
- Ingest 100-page document, retrieval finds correct chunks
- Reranked retrieval outperforms naive top-k by > 10% on held-out QA set
- Adaptive strategy selection works for at least 3 query types

---

## Phase 6: Observability Dashboard — "htop for LLMs" (2 weeks)

**Goal:** Real-time visual profiler showing exactly what ContextOS is doing with memory.

### Deliverables
- [x] FastAPI backend serving metrics from `ContextOS.status()`
- [x] Neo-brutalist styled frontend inspired by devpatel301.github.io
- [x] Live visualization of the token budget and Context Scheduler
- [x] Visual semantic cache hit/miss latency comparisons

### Panels

**Memory Panel**
- Token usage per tier (stacked bar, live)
- Block count per tier
- Eviction rate over time
- GC activity timeline

**Performance Panel**
- LLM call latency (P50, P95, P99)
- Cache hit rate over time
- Token usage vs. cost over time

**Retrieval Panel**
- Retrieval precision@k
- Reranking score distribution

**Decision Log**
- Live stream: "promoted block X", "evicted block Y (LRU)", "cache hit for Z"
- Filterable by decision type, tier, time

### Dashboard Wireframe

```
┌─────────────────────────────────────────────────────────┐
│  ContextOS Profiler       [live]    uptime: 2h 14m      │
├──────────────┬──────────────┬──────────────┬────────────┤
│ Token Usage  │  Cache Hits  │  LLM Calls   │  Cost $    │
│  14,230      │   67%        │   34         │  $0.18     │
├──────────────┴──────────────┴──────────────┴────────────┤
│ Memory Tiers                                            │
│ [Working ████░] [Episodic ██████░] [Semantic ███░░]     │
├─────────────────────────────────────────────────────────┤
│ Decision Log                                            │
│ 14:32:01  CACHE HIT   query="explain TCP"               │
│ 14:31:58  EVICTED     block=a3f2 (LRU)                  │
│ 14:31:55  PROMOTED    block=b91c EPISODIC→SEMANTIC       │
│ 14:31:52  COMPRESSED  block=c44a (2140→380 tokens)       │
└─────────────────────────────────────────────────────────┘
```

### Tech
- **Backend**: FastAPI + Prometheus metrics endpoints
- **Frontend**: React + Recharts
- **Updates**: 2-second polling or WebSocket

### Acceptance Criteria
- Dashboard loads in < 2 seconds
- Metrics update in real time during a live agent session
- Decision log is human-readable and filterable

---

## Phase 7: Agent Runtime + Multi-Step Workflows (1–2 weeks)

**Goal:** A minimal agent execution loop using ContextOS as its memory substrate.

### Deliverables
- [x] Simple `Tool` and `BaseLLM` abstractions
- [x] A `StubLLM` to simulate LLM responses safely
- [x] `AgentRuntime` that executes loops, calls tools, and stores thoughts/actions automatically via ContextOS
- [x] Pytest and Demo script proving context management during tasks

### Agent Loop

```
while not done:
    1. context = allocate(budget, current_task)   # ContextOS
    2. prompt = format(context_window)
    3. thought, action, input = call_llm(prompt)
    4. result = execute_tool(action, input)
    5. store(result) in memory                    # ContextOS
    6. update_scores()
    7. run_gc_if_triggered()
    8. checkpoint()
```

### Example Agents
- Research agent (web search + document QA)
- Code assistant (file read/write + code execution)
- Data analysis agent (CSV + charting)

### Acceptance Criteria
- Agent completes a 10-step research task without context overflow
- Checkpointing allows resuming a failed agent mid-task
- Memory usage measurably better vs. naive baseline agent

---

## Phase 8: Benchmark Harness + Evaluation (1–2 weeks)

**Goal:** Quantitatively compare memory management strategies.

### Deliverables
- [x] Command-line runner (`python -m contextos.benchmark run`)
- [x] Workload: Long Conversation (fact retention)
- [x] Workload: Document QA (retrieval precision)
- [x] Workload: Repeated Queries (cache hit rate)

---

## Phase 9: Multi-Agent Support & Context Version Control

**Goal:** Allow multiple agents to use ContextOS securely and introduce `git`-like snapshotting for memory states.

### Deliverables
- [x] **Agent Isolation**: Added `agent_id` to `ContextConfig`. ContextOS automatically namespaces SQLite databases and ChromaDB collections based on the `agent_id`, providing complete sandboxing.
- [x] **Context VCS**: Added `cos.commit(tag="v1")` and `cos.checkout(tag="v1")` to instantly snapshot and restore memory states, enabling branching logic for agents.

### Policy Comparison Table

| Policy | Tokens | Cost | Latency | Retention | Precision |
|---|---|---|---|---|---|
| Naive baseline | baseline | baseline | baseline | baseline | baseline |
| FIFO | ... | ... | ... | ... | ... |
| LRU | ... | ... | ... | ... | ... |
| Priority | ... | ... | ... | ... | ... |
| Hybrid + compression | ... | ... | ... | ... | ... |
| Full ContextOS | ... | ... | ... | ... | ... |

### Acceptance Criteria
- All benchmarks run end-to-end automatically
- Results are reproducible (fixed seeds, deterministic workloads)
- At least one ContextOS config beats naive baseline on all metrics

---

## Phase 9: Advanced Extensions (ongoing)

Pick-and-choose after core is solid.

| Extension | Description | Complexity |
|---|---|---|
| Multi-agent sharing | Shared memory store across agents, with locking | Medium |
| Context version control | Git-like memory snapshots, rollback, branch | High |
| Adaptive context window | Auto-detect model's context limit, adjust scheduling | Medium |
| Learning eviction policy | Train small classifier to replace hand-written heuristics | High |
| Context diff API | Human-readable diff between two memory states | Low |

---

## Recommended "Fast Demo" Build Order

If you want a working demo fast for recruiting:

1. **Phase 0** — architecture lock (do this right, saves weeks later)
2. **Phase 1** — memory hierarchy (foundation)
3. **Phase 4** — semantic cache (most impressive to demo live)
4. **Phase 2** — scheduler (makes `allocate()` the main API)
5. **Phase 6** — dashboard (turns everything into a visual story)
6. **Phase 8** — benchmarks (gives you numbers to talk about)

**Realistic timeline:** 4–6 weeks of focused evenings/weekends = working demo.

---

## Milestone Summary

| Milestone | Phase | Outcome |
|---|---|---|
| M0 | 0 | Architecture locked, scaffolding ready |
| M1 | 1 | Memory hierarchy end-to-end |
| M2 | 2 | `allocate()` API with configurable scheduling |
| M3 | 3 | Eviction + GC, pluggable policies |
| M4 | 4 | Semantic cache reduces LLM calls measurably |
| M5 | 5 | Smart RAG beats naive top-k |
| M6 | 6 | Dashboard live with real-time memory decisions |
| M7 | 7 | Agent completes multi-step tasks via ContextOS |
| M8 | 8 | Benchmark results, policy comparison documented |
| M9 | 9 | Advanced extensions (select subset) |

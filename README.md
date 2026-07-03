# ContextOS

> **An Operating System for LLM Context**
> Managing agent memory the way an OS manages RAM — with scheduling, paging, compression, caching, and eviction.

---

## The Problem

Every LLM agent today works like this:

```
User → Prompt → LLM → Response
```

But after a real conversation:

```
Conversation history
+ 20 documents
+ tool outputs
+ past reasoning traces
+ retrieved memories
+ generated code
+ web search results
= 100,000+ token prompt
```

This causes:

- **Context overflow** — the model simply can't fit everything
- **Cost explosion** — token pricing scales linearly
- **Latency degradation** — larger prompts are slower
- **Selective amnesia** — critical facts get dropped silently

The industry's answer is RAG. But most RAG is just:

```
embed → vector DB → top-k
```

That's primitive. That's like replacing RAM with a lookup table and calling it a memory manager.

---

## The Insight

Context management is an **operating systems problem**.

An OS never loads everything into RAM. It pages, schedules, compresses, caches, and evicts memory intelligently — using policies tuned to workload patterns.

ContextOS applies the same thinking to LLM agents.

```
Developer calls:

    context = contextos.allocate(budget=16_000, query=user_query)

ContextOS decides:
  - What gets loaded from working memory
  - What gets retrieved from long-term storage
  - What gets compressed vs. dropped
  - What gets served from semantic cache
  - How the token budget is allocated across competing memories
```

Instead of developers writing `messages[-10:]` and hoping, ContextOS gives agents a principled, observable memory substrate.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────┐
│                   User / Agent                  │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│              Context Scheduler                  │
│   (token budget allocation, priority queuing)   │
└────────┬─────────────────────┬──────────────────┘
         │                     │
         ▼                     ▼
┌────────────────┐   ┌──────────────────────────┐
│ Semantic Cache │   │     Memory Manager        │
│ (cache hits    │   │  (hierarchy controller)   │
│  skip LLM)     │   └──────┬───────────┬────────┘
└────────────────┘          │           │
                            ▼           ▼
                   ┌──────────────┐  ┌──────────────────┐
                   │ Short-Term   │  │   Long-Term       │
                   │ Memory       │  │   Memory          │
                   │ (working +   │  │   (episodic +     │
                   │  episodic)   │  │    semantic +     │
                   └──────┬───────┘  │    archived)      │
                          │          └────────┬──────────┘
                          └────────┬──────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │     Vector DB         │
                        │  (ChromaDB / FAISS)   │
                        └──────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │  Observability Layer  │
                        │  (profiler, metrics,  │
                        │   dashboard)          │
                        └──────────────────────┘
```

---

## Core Components

| Component                   | OS Analogy            | Purpose                                            |
| --------------------------- | --------------------- | -------------------------------------------------- |
| **Memory Manager**    | RAM controller        | Manages the full memory hierarchy                  |
| **Context Scheduler** | CPU scheduler         | Allocates token budget across competing memories   |
| **Context Pager**     | Virtual memory / swap | Pages old memories out, loads relevant ones in     |
| **Memory Compressor** | zRAM / zswap          | Summarizes verbose memories to save tokens         |
| **Semantic Cache**    | CPU instruction cache | Returns cached LLM responses on similar queries    |
| **Eviction Engine**   | Page replacement      | Decides what to drop (LRU / LFU / FIFO / priority) |
| **Garbage Collector** | JVM GC                | Removes duplicates, stale, and expired memories    |
| **Context Profiler**  | `htop` / `vmstat` | Visualizes memory usage, cost, latency, decisions  |
| **Agent Runtime**     | Process scheduler     | Executes multi-step workflows with checkpointing   |
| **Benchmark Harness** | perf / valgrind       | Compares memory policies quantitatively            |

---

## Memory Hierarchy

Directly mirrors CPU memory hierarchy:

```
┌───────────────────────────────┐   ← Registers
│  System Prompt / Instructions │     Always present
└───────────────────────────────┘
┌───────────────────────────────┐   ← L1 Cache
│  Working Memory               │     Current turn context
└───────────────────────────────┘
┌───────────────────────────────┐   ← L2 Cache
│  Conversation Memory          │     Recent N turns
└───────────────────────────────┘
┌───────────────────────────────┐   ← RAM
│  Episodic Memory              │     Key past events, compressed
└───────────────────────────────┘
┌───────────────────────────────┐   ← SSD
│  Semantic Memory              │     Facts, summaries, knowledge
└───────────────────────────────┘
┌───────────────────────────────┐   ← HDD
│  Archived Memory              │     Cold storage, vector DB
└───────────────────────────────┘
```

---

## Use Cases

### 1. Long-Running Research Agent

A research agent that reads 50 papers over a 2-hour session. Without ContextOS, it forgets paper #3 by the time it reads paper #40. With ContextOS, the most relevant facts are always in context — compressed, ranked, and retrieved on demand.

### 2. Customer Support Bot

A support bot handling a complex multi-session troubleshooting case. ContextOS maintains a compressed episodic memory of previous sessions, so the agent never asks the same question twice, and important customer context persists across days.

### 3. Coding Assistant with Codebase Context

An agent exploring a large codebase. ContextOS pages code files in and out of context based on relevance to the current task, rather than naively truncating or stuffing everything in.

### 4. Multi-Agent Workflow

A planner → coder → reviewer pipeline where each agent shares a common memory. ContextOS manages read/write access, deduplication, and version history across agents.

### 5. Cost-Sensitive Production Deployment

Every repeated or semantically-similar query hits the semantic cache instead of the LLM. Token usage and API cost drop dramatically for repetitive workloads.

---

## What Makes This Different

| Typical RAG           | ContextOS                                   |
| --------------------- | ------------------------------------------- |
| Top-k retrieval       | Scheduled retrieval with priority scoring   |
| Static context window | Dynamic token budget allocation             |
| No eviction           | LRU / LFU / FIFO / hybrid eviction policies |
| No compression        | Summarization-based memory compression      |
| No caching            | Semantic similarity cache                   |
| No observability      | Full profiler with metrics and dashboard    |
| No evaluation         | Built-in benchmark harness                  |
| Monolithic            | Pluggable, policy-swappable architecture    |

---

## Documentation

| Document                                 | Description                                      |
| ---------------------------------------- | ------------------------------------------------ |
| [ROADMAP.md](./docs/ROADMAP.md)           | Phase-by-phase build plan                        |
| [ARCHITECTURE.md](./docs/ARCHITECTURE.md) | Detailed system design and data flows            |
| [FEATURES.md](./docs/FEATURES.md)         | Feature specifications with implementation notes |
| [EVALUATION.md](./docs/EVALUATION.md)     | Metrics, benchmarks, and evaluation methodology  |
| [EXTENSIONS.md](./docs/EXTENSIONS.md)     | Advanced features and future directions          |

---

## Tech Stack

| Layer         | Technology                            |
| ------------- | ------------------------------------- |
| Backend       | Python 3.11+, FastAPI                 |
| LLM           | OpenAI API + Ollama (local)           |
| Embeddings    | sentence-transformers                 |
| Vector DB     | ChromaDB (dev), FAISS (production)    |
| Storage       | SQLite (dev), PostgreSQL (production) |
| Observability | Prometheus + Grafana                  |
| Frontend      | React + Recharts                      |
| Testing       | pytest, hypothesis                    |

---

## Status

> 🚧 **Pre-development** — Planning phase. See [ROADMAP.md](./docs/ROADMAP.md).

---

*ContextOS is a systems research project exploring whether OS-inspired memory management primitives produce meaningfully better long-context agent behavior than naive RAG.*

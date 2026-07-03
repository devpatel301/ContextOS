# ContextOS: Operating System for LLM Context

ContextOS is a framework that manages LLM agent memory using operating system primitives. Instead of treating context as an infinite string or relying exclusively on top-k vector search, ContextOS introduces scheduling, paging, compression, caching, and eviction to the context window.

## The Problem

Standard LLM agents operate by appending conversation history, tool outputs, reasoning traces, and retrieved documents into a single prompt. For long-running workflows, this approach quickly exceeds token limits, increases latency, drives up API costs, and leads to selective amnesia where critical information is dropped.

Traditional Retrieval-Augmented Generation (RAG) attempts to solve this by storing data in a vector database and retrieving the top results. However, naive retrieval lacks lifecycle management. It does not compress older memories, deduplicate redundant thoughts, or schedule information dynamically based on priority and recency.

## The Solution

Context management is fundamentally an operating systems problem. An OS manages RAM through paging, scheduling, and eviction based on workload demands. ContextOS brings these same principles to LLM agents.

When an agent queries the system, ContextOS evaluates the available token budget and dynamically selects what gets loaded from working memory, retrieved from long-term storage, compressed, or dropped entirely. It also intercepts redundant queries using a semantic cache to save API calls.

## Core Architecture

ContextOS is built around components that mirror traditional OS memory management:

* **Memory Manager (RAM Controller):** Oversees the entire memory hierarchy, migrating data between short-term and long-term storage.
* **Context Scheduler (CPU Scheduler):** Allocates the available token budget across competing memory blocks based on priority scores.
* **Context Pager (Virtual Memory):** Pages old memories out to disk and loads relevant ones back into the active context window.
* **Memory Compressor (zRAM):** Summarizes verbose or older memories to maximize token efficiency.
* **Semantic Cache (Instruction Cache):** Intercepts queries and returns cached responses for semantically identical requests.
* **Eviction Engine (Page Replacement):** Determines which blocks to drop when capacity is reached, supporting LRU, LFU, FIFO, and hybrid policies.
* **Garbage Collector (GC):** Periodically runs in the background to deduplicate redundant memories and purge expired data.

## Memory Hierarchy

The system enforces a strict hierarchy to organize information:

1. **System Instructions (Registers):** Core rules that are never evicted.
2. **Working Memory (L1 Cache):** The current turn's active context.
3. **Conversation Memory (L2 Cache):** The most recent interaction history.
4. **Episodic Memory (RAM):** Compressed logs of past actions and events.
5. **Semantic Memory (SSD):** Verified facts and knowledge bases.
6. **Archived Memory (HDD):** Cold storage maintained in a vector database for infrequent retrieval.

## Use Cases

**Long-Running Agents:** For agents conducting extensive research, ContextOS ensures that early findings remain accessible and are not blindly pushed out of the context window.

**Customer Support:** Support bots handling multi-day troubleshooting cases can maintain compressed episodic memory of previous sessions without duplicating context.

**Codebase Navigation:** Coding assistants can page specific files in and out of the context window based on current relevance rather than indiscriminately truncating files.

**Multi-Agent Workflows:** ContextOS provides sandboxed memory management, allowing multiple agents to collaborate while maintaining independent, version-controlled context states.

## Technical Implementation

ContextOS is written in Python and provides a FastAPI-based observability backend for profiling memory usage. It integrates seamlessly with standard LLM APIs. The memory substrate is backed by SQLite for relational metadata and ChromaDB for vector storage, with native support for local embedding models.

---
*ContextOS is a systems research project exploring the application of OS-level memory primitives to autonomous agents.*

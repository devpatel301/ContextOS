# ContextOS

ContextOS handles LLM context windows the same way an operating system handles RAM. Instead of blindly stuffing everything into a single prompt or relying on basic top-k vector search, it uses scheduling, paging, compression, and caching to manage an agent's memory.

## Why build this?

Most LLM agents today just append conversation history, retrieved documents, and tool outputs into one massive prompt. When you hit the token limit, things break. Even if you don't hit the limit, huge prompts are slow, expensive, and cause the model to forget facts stuck in the middle.

Typical Retrieval-Augmented Generation (RAG) tries to fix this by storing data in a vector DB and returning the most relevant chunks. But it's static. It doesn't compress old memories, it doesn't deduplicate, and it doesn't prioritize information based on how recently it was used.

I built ContextOS to treat the context window like limited physical RAM. It decides what to page in, what to compress, and what to evict based on the workload.

## How it works

When an agent needs context, you pass it a token budget. ContextOS will then:
- Load recent turns from working memory.
- Retrieve relevant facts from long-term storage (ChromaDB).
- Compress older memories into summaries to save space.
- Intercept duplicate queries using a semantic cache.
- Allocate the remaining token budget based on a scoring algorithm.

## Core concepts

- **Memory Manager:** Moves data between short-term (SQLite) and long-term (ChromaDB) storage.
- **Context Scheduler:** Packs the context window optimally without overflowing the token limit.
- **Memory Compressor:** Shrinks older memories down to their core facts.
- **Semantic Cache:** Returns cached LLM outputs if a semantically identical query was already processed.
- **Eviction Engine:** Drops blocks using LRU (least recently used) or hybrid policies when memory fills up.
- **Garbage Collector:** Runs in the background to clean up expired or duplicate memories.

## How to run it

**1. Install dependencies**
```bash
# Optional: create a virtual environment first
pip install -r requirements.txt
```

**2. Run the observability dashboard**
ContextOS comes with a built-in FastAPI dashboard to visualize memory usage, cache hit rates, and token budgets in real-time.
```bash
uvicorn contextos.api.main:app --reload
```
Then open `http://localhost:8000` in your browser.

**3. Run the examples**
I've included several demo scripts in the `examples/` directory that show off the different features. 

For the complete agent runtime (which shows tool usage and memory persistence):
```bash
python examples/phase7_demo.py
```

For testing multi-agent isolation and context version control (like `git checkout` for agent memory):
```bash
python examples/phase9_demo.py
```

**4. Run the benchmarks**
I built a CLI harness to quantitatively test memory retention against naive RAG approaches.
```bash
python -m contextos.benchmark run --workload long_conversation
python -m contextos.benchmark run --workload qa
python -m contextos.benchmark run --workload repeated_queries
```

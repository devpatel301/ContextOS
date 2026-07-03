"""
ContextOS — API & Dashboard (Phase 6).
"""
import asyncio
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from contextos import ContextConfig, ContextOS
from contextos.models.tier import MemoryTier

# Initialize global ContextOS instance for the dashboard to monitor
config = ContextConfig(
    db_path="./demo_dashboard.db",
    chroma_persist_dir="./.chroma_dashboard",
    embedding_stub=False,  # Fast string hashing for demo
    enable_semantic_cache=True,
    token_budget=2000,
    working_memory_reserve=500,
)
cos = ContextOS(config)

# Setup FastAPI
app = FastAPI(title="ContextOS Dashboard")

# Mount static files & templates
base_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")
templates = Jinja2Templates(directory=str(base_dir / "templates"))


class QueryRequest(BaseModel):
    query: str
    simulate_latency: bool = True

@app.on_event("startup")
async def startup_event():
    # Seed some data for the dashboard to look interesting
    cos.store("System: You are a helpful assistant.", tier=MemoryTier.WORKING)
    for i in range(5):
        cos.store(f"Episodic memory block {i}", tier=MemoryTier.EPISODIC)
    cos.store("Fact: Python is a programming language.", tier=MemoryTier.SEMANTIC)

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/status")
async def get_status():
    status = cos.status()
    # Add a mock cache status
    cache_status = {"entries": 0, "hits": 0, "misses": 0}
    if cos.cache:
        all_cache = cos.cache.store.get_all()
        cache_status["entries"] = len(all_cache)
        cache_status["hits"] = sum(e.hit_count for e in all_cache)
        cache_status["misses"] = len(all_cache) # roughly
        
    return {
        "memory": status,
        "cache": cache_status,
        "config": {
            "budget": config.token_budget,
            "policy": config.scheduling_policy
        }
    }

@app.post("/api/query")
async def run_query(req: QueryRequest):
    # 1. Check Cache
    cache_hit = cos.cache_query(req.query)
    
    # 2. Retrieve Context Window
    window = cos.retrieve(req.query, budget=config.token_budget)
    
    if req.simulate_latency and not cache_hit:
        await asyncio.sleep(1.0)
        
    if not cache_hit:
        cos.cache_response(req.query, f"Generated response for: {req.query}")
        
    return {
        "cache_hit": bool(cache_hit),
        "tokens_used": window.tokens_used,
        "budget": window.token_budget,
        "blocks": [
            {
                "tier": b.tier.value,
                "score": round(b.importance_score, 2),
                "content": b.content[:50] + "..." if len(b.content) > 50 else b.content
            } for b in window.blocks
        ]
    }

@app.post("/api/reset")
async def reset_memory():
    # 1. Clear SQLite store
    with cos._store._conn:
        cos._store._conn.execute("DELETE FROM memory_blocks")
        
    # 2. Clear Chroma collection
    try:
        cos.retriever.vector_index.client.delete_collection("contextos_memories")
        cos.retriever.vector_index.collection = cos.retriever.vector_index.client.get_or_create_collection(
            name="contextos_memories",
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to reset ChromaDB collection: {e}")
        
    # 3. Clear Cache store
    if cos.cache:
        with cos.cache.store._conn:
            cos.cache.store._conn.execute("DELETE FROM semantic_cache")
            
    # 4. Re-seed default initial data
    cos.store("System: You are a helpful assistant.", tier=MemoryTier.WORKING)
    for i in range(5):
        cos.store(f"Episodic memory block {i}", tier=MemoryTier.EPISODIC)
    cos.store("Fact: Python is a programming language.", tier=MemoryTier.SEMANTIC)
    
    return {"status": "success", "message": "Memory and cache cleared and re-seeded."}

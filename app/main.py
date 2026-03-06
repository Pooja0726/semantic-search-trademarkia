"""
Part 4: FastAPI Service
========================
Exposes the semantic cache as a live API with proper state management.

Endpoints:
  POST /query          — embed, cache-check, retrieve, store
  GET  /cache/stats    — return hit/miss/entry counts
  DELETE /cache        — flush cache and reset stats

State management:
  The SemanticCache instance is created at application startup via FastAPI's
  lifespan context manager.  It is shared across all requests via dependency
  injection (get_cache_dep).  This is the correct pattern for FastAPI — avoids
  the deprecated @app.on_event and ensures clean startup/shutdown.

Startup command (from project root, inside venv):
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

# We import the cache module from the same package
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cache import SemanticCache, DEFAULT_THRESHOLD

# ── Global cache instance ─────────────────────────────────────────────────────
# Stored here so lifespan can set it; dependency function reads it.
_cache: Optional[SemanticCache] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load the sentence-transformer model, connect to ChromaDB,
             load FCM centroids.  This runs ONCE when uvicorn starts.
    Shutdown: nothing to clean up (ChromaDB is persistent, model is in-memory).
    """
    global _cache
    print("[Startup] Initialising semantic cache...")
    _cache = SemanticCache(threshold=DEFAULT_THRESHOLD)
    _cache.load_resources()
    print("[Startup] Ready to serve requests.")
    yield
    # Shutdown
    print("[Shutdown] Cache flushed.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Semantic Search Service",
    description=(
        "Semantic search over the 20 Newsgroups corpus with fuzzy clustering "
        "and a cluster-aware semantic cache."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Dependency ────────────────────────────────────────────────────────────────

def get_cache_dep() -> SemanticCache:
    if _cache is None:
        raise HTTPException(status_code=503, detail="Cache not initialised yet.")
    return _cache


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000,
                       example="What are the latest developments in space exploration?")


class QueryResponse(BaseModel):
    query:            str
    cache_hit:        bool
    matched_query:    Optional[str]   = None
    similarity_score: Optional[float] = None
    result:           str
    dominant_cluster: int


class CacheStats(BaseModel):
    total_entries: int
    hit_count:     int
    miss_count:    int
    hit_rate:      float


class CacheFlushResponse(BaseModel):
    message: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Semantic query with cache",
    description=(
        "Embeds the query, checks the semantic cache for a similar prior query "
        "(cosine similarity ≥ threshold), and returns the cached result on hit.  "
        "On miss, retrieves from ChromaDB, stores in cache, and returns."
    ),
)
def query_endpoint(
    body:  QueryRequest,
    cache: SemanticCache = Depends(get_cache_dep),
) -> QueryResponse:
    """
    Main query endpoint.

    - Checks semantic cache first (cluster-aware, O(N/K) lookup).
    - On HIT:  returns cached result with matched_query and similarity_score.
    - On MISS: queries ChromaDB vector store, caches result, returns it.
    """
    result = cache.query(body.query)
    return QueryResponse(**result)


@app.get(
    "/cache/stats",
    response_model=CacheStats,
    summary="Cache statistics",
    description="Returns current cache state: entry count, hit/miss counts, hit rate.",
)
def cache_stats_endpoint(
    cache: SemanticCache = Depends(get_cache_dep),
) -> CacheStats:
    return CacheStats(**cache.stats())


@app.delete(
    "/cache",
    response_model=CacheFlushResponse,
    summary="Flush cache",
    description="Clears all cache entries and resets hit/miss counters.",
)
def cache_flush_endpoint(
    cache: SemanticCache = Depends(get_cache_dep),
) -> CacheFlushResponse:
    cache.flush()
    return CacheFlushResponse(message="Cache flushed and stats reset.")


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}


# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "Semantic Search Service",
        "docs":    "/docs",
        "redoc":   "/redoc",
    }
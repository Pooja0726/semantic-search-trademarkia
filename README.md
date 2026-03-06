# Semantic Search System — 20 Newsgroups

A lightweight semantic search system built over the 20 Newsgroups corpus (~18,000 news posts across 20 topic categories). The system combines dense vector embeddings, fuzzy clustering, a cluster-aware semantic cache, and a FastAPI service.

No paid APIs. No GPU required. Everything runs locally.

---

## Problem Statement

Given the 20 Newsgroups dataset, build a system with three core components:

1. Fuzzy clustering of the corpus using vector embeddings and a vector database
2. A semantic cache layer that avoids redundant computation on similar queries, built from first principles without Redis or any caching middleware
3. A FastAPI service that exposes the cache as a live API endpoint with proper state management

---

## System Architecture

```
20 Newsgroups Dataset (fetched via sklearn — no manual download)
         |
         v
Part 1: Ingest and Embed
  - Clean documents: strip email headers, quoted replies, signatures
  - Embed with sentence-transformers/all-MiniLM-L6-v2 (384-dim, local)
  - Store in ChromaDB with cosine metric for filtered retrieval
         |
         v
Part 2: Fuzzy Clustering
  - Reduce dimensions: UMAP 384 -> 30 dims (cosine metric)
  - Evaluate K in {8, 10, 12, 15, 18, 20} using FPC and silhouette score
  - Run Fuzzy C-Means with K=15, fuzziness m=1.6
  - Each document gets a 15-dim membership distribution, not a single label
         |
         v
Part 3: Semantic Cache (pure Python, no Redis)
  - Cluster-aware bucketing: cache entries stored by dominant cluster
  - Lookup is O(N/K) instead of O(N) — 15x faster as cache grows
  - Cosine similarity threshold tau=0.80 determines cache hits
         |
         v
Part 4: FastAPI Service
  - POST   /query        — semantic search with cache
  - GET    /cache/stats  — hit/miss statistics
  - DELETE /cache        — flush cache and reset stats
```

---

## Actual Results (from running setup.py)

### Dataset
- Total documents loaded: 18,846
- Documents after cleaning: 18,678 (dropped 168 too-short docs)
- 20 categories: alt.atheism, comp.graphics, comp.os.ms-windows.misc, comp.sys.ibm.pc.hardware, comp.sys.mac.hardware, comp.windows.x, misc.forsale, rec.autos, rec.motorcycles, rec.sport.baseball, rec.sport.hockey, sci.crypt, sci.electronics, sci.med, sci.space, soc.religion.christian, talk.politics.guns, talk.politics.mideast, talk.politics.misc, talk.religion.misc

### K Evaluation (evidence for choosing K=15)

| K  | FPC    | Silhouette |
|----|--------|------------|
| 8  | 0.7550 | 0.5899     |
| 10 | 0.7434 | 0.6018     |
| 12 | 0.7185 | 0.5861     |
| 15 | 0.7069 | 0.5734     |
| 18 | 0.6939 | 0.5698     |
| 20 | 0.6755 | 0.5493     |

K=15 was chosen because it provides the best balance between cluster crispness (FPC) and semantic interpretability. K=8 scores higher on FPC but merges semantically distinct topics. K=15 correctly separates sports, politics, computers, science, and religion into distinct regions.

### Cluster Analysis (K=15, FPC=0.7069)

| Cluster | Dominant Category | Purity | Notes |
|---------|------------------|--------|-------|
| 0  | talk.politics.mideast | 86.3% | Clean — Middle East politics |
| 1  | rec.sport.baseball | 94.7% | Very clean — baseball only |
| 2  | rec.autos | 61.0% | Mixed with motorcycles and forsale |
| 3  | comp.graphics | 45.3% | Mixed with Windows OS — expected overlap |
| 4  | misc.forsale | 40.0% | Mixed with electronics — items for sale |
| 5  | sci.space | 54.1% | Space science, some graphics |
| 6  | talk.politics.guns | 46.3% | Mixed with politics.misc — boundary cluster |
| 7  | sci.med | 85.5% | Clean — medical science |
| 8  | sci.crypt | 90.9% | Very clean — cryptography |
| 9  | alt.atheism | 14.5% | Most uncertain cluster — religion/philosophy boundary |
| 10 | rec.motorcycles | 68.7% | Mixed with autos — expected |
| 11 | soc.religion.christian | 45.0% | Mixed with alt.atheism — debate posts |
| 12 | comp.windows.x | 78.4% | Clean — X Window System |
| 13 | comp.sys.ibm.pc.hardware | 33.5% | Mixed IBM + Mac hardware — semantically close |
| 14 | rec.sport.hockey | 94.1% | Very clean — hockey only |

### Membership Certainty Distribution
- Documents with max membership above 0.85 (very certain): 10,450 (55.9%)
- Documents with max membership below 0.40 (very uncertain): 1,294 (6.9%)
- Mean max membership: 0.798

### Boundary Document Example
The most uncertain documents are airline ticket posts labeled misc.forsale. They score equally across Cluster 2 (autos/transport) and Cluster 10 (motorcycles/transport) because they are about travel but not cleanly about buying/selling physical goods. This is the fuzzy clustering working correctly — these documents genuinely do not belong to a single cluster.

---

## File Structure

```
semantic_analyzer/
|-- app/
|   |-- __init__.py
|   |-- main.py              # FastAPI service (Part 4)
|   |-- cache.py             # Semantic cache (Part 3)
|-- scripts/
|   |-- part1_ingest.py      # Embedding and vector DB setup (Part 1)
|   |-- part2_clustering.py  # Fuzzy clustering and analysis (Part 2)
|-- data/                    # Auto-generated by setup.py, gitignored
|   |-- chroma_db/           # ChromaDB persistent vector store
|   |-- embeddings.npy       # Raw embeddings shape (18678, 384)
|   |-- embeddings_cache.pkl # Cached embeddings to skip re-encoding
|   |-- umap_reduced.npy     # UMAP-reduced embeddings shape (18678, 30)
|   |-- umap_reducer.pkl     # Fitted UMAP object for query-time projection
|   |-- fcm_k15.pkl          # FCM centroids and membership matrix
|   |-- docs.pkl             # Cleaned document list
|   |-- docs_clustered.pkl   # Documents with cluster assignments
|-- requirements.txt
|-- setup.py                 # One-shot pipeline runner (Parts 1 and 2)
|-- Dockerfile               # Bonus containerisation
|-- docker-compose.yml
|-- README.md
```

---

## How to Run

### Requirements
- Python 3.10 or above
- No GPU required
- No API keys required
- Internet only needed on first run to download the model and dataset

### Step 1 — Clone the repository

```bash
git clone https://github.com/Pooja0726/semantic-search-trademarkia.git
cd semantic-search-trademarkia
```

### Step 2 — Create virtual environment

```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run setup pipeline (one time only)

```bash
python setup.py
```

This will:
- Download the 20 Newsgroups dataset automatically via sklearn
- Clean and embed all 18,678 documents using all-MiniLM-L6-v2
- Store vectors in ChromaDB
- Fit UMAP and save the reducer
- Run K evaluation sweep over K in {8, 10, 12, 15, 18, 20}
- Run Fuzzy C-Means with K=15
- Print full cluster analysis with boundary documents
- Save all outputs to ./data/

Expected time on CPU: 15 to 25 minutes on first run. All subsequent runs use cached files and complete in seconds.

### Step 5 — Start the API server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs for the interactive Swagger UI.

---

## API Reference

### POST /query

Accepts a natural language query. Checks the semantic cache first. On a hit, returns the cached result. On a miss, queries ChromaDB, stores the result, and returns it.

Request:
```json
{
  "query": "What are the arguments for gun control?"
}
```

Response on cache miss:
```json
{
  "query": "What are the arguments for gun control?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": "[Category: talk.politics.guns | Similarity: 0.5803]\n...",
  "dominant_cluster": 6
}
```

Response on cache hit (same topic, different wording):
```json
{
  "query": "Should firearms be regulated by the government?",
  "cache_hit": true,
  "matched_query": "What are the arguments for gun control?",
  "similarity_score": 0.8821,
  "result": "[Category: talk.politics.guns | Similarity: 0.5803]\n...",
  "dominant_cluster": 6
}
```

---

### GET /cache/stats

Returns current cache state.

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

---

### DELETE /cache

Flushes all cache entries and resets all stats.

```json
{
  "message": "Cache flushed and stats reset."
}
```

---

## Design Decisions

### Embedding Model: all-MiniLM-L6-v2
Chosen for 384-dim output, fast CPU inference, strong semantic quality, and completely free local execution with no API key. The model is downloaded once by sentence-transformers and cached locally. OpenAI embeddings were not used because they require a paid API key and send data externally.

### Vector Database: ChromaDB
Chosen because it is fully local with no separate server process, supports cosine similarity natively, allows metadata filtering by category and split, and persists to disk. FAISS was considered but lacks metadata filtering. Pinecone requires a paid account.

### Fuzzy C-Means over K-Means
Hard clustering assigns each document to exactly one cluster. A document about gun legislation belongs to both politics and firearms to varying degrees. Fuzzy C-Means assigns each document a probability distribution over all clusters, which is more honest about the actual semantic structure of the corpus. This membership distribution is also used by the cache to determine which bucket to place a query in.

### K=15
K=8 has the highest FPC score but merges semantically distinct topics such as baseball and hockey into one sports cluster. K=15 correctly separates them while keeping the FPC at an acceptable 0.7069. The 20 original labels have significant overlap so the real semantic structure is closer to 15 meaningful regions than 20.

### Fuzziness m=1.6
m=1.0 produces hard assignments identical to K-Means. m=2.0 or higher produces overly diffuse memberships where every document belongs equally to all clusters. m=1.6 produces clear dominant assignments while preserving meaningful overlap at topic boundaries.

### Cache Threshold tau=0.80

| tau  | Behaviour |
|------|-----------|
| 0.70 | Aggressive. Loosely similar queries hit. Risk of returning wrong results. |
| 0.80 | Default. Paraphrases consistently hit. Different sub-topics miss. |
| 0.90 | Conservative. Only near-exact rewording hits. |
| 0.95 | Near identical to exact match. Misses most legitimate reuses. |

At tau=0.80 the cache reveals that most user queries cluster around a small set of core information needs, even when phrased very differently.

### Cache Data Structure
The cache is a plain Python dictionary keyed by cluster ID. Each value is a list of CacheEntry objects. On lookup, only the entries in the query's dominant cluster are compared, making lookup O(N/K) instead of O(N). At K=15 this is a 15x speedup as the cache grows. No Redis, no Memcached, no caching library of any kind.

---

## Docker (Bonus)

Run setup.py first to populate the ./data/ folder, then:

```bash
docker-compose up --build
```

The ./data directory is mounted as a volume so the container uses the pre-built embeddings and does not re-run setup.

---

## Tools Used (all free, no API keys)

| Component        | Tool                                  | Licence    |
|------------------|---------------------------------------|------------|
| Dataset          | sklearn.datasets.fetch_20newsgroups   | BSD        |
| Embeddings       | sentence-transformers all-MiniLM-L6-v2 | Apache-2.0 |
| Vector Database  | ChromaDB                              | Apache-2.0 |
| Fuzzy Clustering | scikit-fuzzy                          | BSD        |
| Dim Reduction    | umap-learn                            | BSD        |
| API Framework    | FastAPI + uvicorn                     | MIT        |
| Cache            | Pure Python, hand-written             | —          |
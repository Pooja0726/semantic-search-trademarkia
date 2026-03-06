"""
setup.py — One-shot pipeline runner
=====================================
Run this ONCE before starting the API server.
It executes Parts 1, 2 in sequence, producing all the data files
needed by the FastAPI service.

Usage (from project root, inside venv):
    python setup.py

Then start the server:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

NOTE: First run takes ~10-20 minutes depending on hardware:
  - ~5-10 min   to embed ~18,000 docs with MiniLM (CPU)
  - ~2-5  min   for UMAP reduction
  - ~3-8  min   for FCM K-sweep + final clustering
  Subsequent runs use cached .npy/.pkl files and are near-instant.
"""

import os
import sys
import pickle
import numpy as np

# Ensure local scripts are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.part1_ingest    import run_part1
from scripts.part2_clustering import run_part2, BEST_K

import umap as umap_lib  # save the fitted reducer for Part 3 inference


def save_umap_reducer(embeddings: np.ndarray):
    """
    Fit and save the UMAP reducer so Part 3 can project new query vectors.
    This is separate from the cached reduced matrix — we need the fitted
    reducer object to transform unseen embeddings at query time.
    """
    reducer_path = "./data/umap_reducer.pkl"
    if os.path.exists(reducer_path):
        print("[Setup] UMAP reducer already saved.")
        return

    print("[Setup] Fitting UMAP reducer for query-time projection...")
    reducer = umap_lib.UMAP(
        n_components=30,
        n_neighbors=15,
        metric="cosine",
        random_state=42,
        low_memory=False,
    )
    # Fit on full corpus embeddings
    reducer.fit(embeddings)

    with open(reducer_path, "wb") as f:
        pickle.dump(reducer, f)
    print(f"[Setup] UMAP reducer saved → {reducer_path}")


def main():
    print("=" * 60)
    print("  Semantic Search System — Full Pipeline Setup")
    print("=" * 60)

    os.makedirs("./data", exist_ok=True)

    # Part 1: ingest, clean, embed, store in ChromaDB
    docs, embeddings, category_names = run_part1()

    # Save UMAP reducer for runtime inference
    save_umap_reducer(embeddings)

    # Part 2: fuzzy clustering, analysis
    docs_clustered, u, summaries = run_part2()

    print("\n" + "=" * 60)
    print("  Setup complete!  All data files are in ./data/")
    print("  Start the API server with:")
    print("    uvicorn app.main:app --host 0.0.0.0 --port 8000")
    print("=" * 60)


if __name__ == "__main__":
    main()
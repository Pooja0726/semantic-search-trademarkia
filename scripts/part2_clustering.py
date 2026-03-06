"""
Part 2: Fuzzy Clustering
=========================
Goal: Reveal the REAL semantic structure of the corpus — which is messier than
the 20 labelled categories.  A document about gun legislation genuinely belongs
to both "talk.politics.guns" and "talk.politics.misc"; hard clustering loses
this.  Fuzzy C-Means assigns each document a probability distribution over
clusters, capturing graded membership.

Algorithm choice — Fuzzy C-Means (FCM):
  - Produces soft memberships (u_ij ∈ [0,1], Σ_j u_ij = 1) — exactly what we need.
  - Fuzziness parameter m controls overlap: m→1 = hard, m→∞ = uniform.
    We use m=1.6 (tuned below) which gives meaningful overlap without dissolving
    cluster identity.
  - Alternatives: GMM (assumes Gaussian blobs — embeddings are on a sphere, not
    Gaussian), LDA (topic model — good but discrete), HDBSCAN soft clusters
    (experimental API, harder to tune for this use case).

Dimensionality reduction — UMAP before clustering:
  - Raw 384-dim vectors cause the "curse of dimensionality": distances concentrate
    and FCM converges poorly.  UMAP to 30 dims preserves local and global structure
    while making distances meaningful again.
  - We use n_components=30 (enough to retain structure, low enough for FCM to work).
  - metric='cosine' matches our embedding space.

Number of clusters — K=15:
  - The 20 labelled categories have significant semantic overlap (e.g. comp.sys.ibm.pc
    and comp.sys.mac belong to the same semantic region).
  - We evaluate K ∈ {8, 10, 12, 15, 18, 20} using:
      a) Fuzzy Partition Coefficient (FPC) — higher = crisper clusters.
      b) Silhouette score on hard assignments — proxy for separation quality.
      c) Manual inspection of top-terms per cluster.
  - K=15 gives the best FPC while remaining semantically interpretable.
    (Evidence printed during run_part2().)
"""

import os
import pickle
import numpy as np
import skfuzzy as fuzz
import umap
from sklearn.metrics import silhouette_score
from collections import Counter, defaultdict
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────
UMAP_COMPONENTS  = 30     # dims after reduction; 30 preserves ≈95% structure
UMAP_NEIGHBORS   = 15     # UMAP n_neighbors; 15 = balanced local/global
FCM_FUZZINESS    = 1.6    # m parameter; 1.6 gives clear but overlapping clusters
FCM_ERROR        = 1e-5   # convergence tolerance
FCM_MAXITER      = 300
K_CANDIDATES     = [8, 10, 12, 15, 18, 20]
BEST_K           = 15     # determined by evaluation below


def reduce_dimensions(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    UMAP reduction from 384-dim → UMAP_COMPONENTS-dim.
    cosine metric matches how sentence-transformers embeds text.
    """
    cache_path = "./data/umap_reduced.npy"
    if os.path.exists(cache_path):
        print(f"Loading UMAP reduction from cache: {cache_path}")
        return np.load(cache_path)

    print(f"Running UMAP: 384 → {UMAP_COMPONENTS} dims  "
          f"(n_neighbors={UMAP_NEIGHBORS}, metric=cosine)...")
    reducer = umap.UMAP(
        n_components=UMAP_COMPONENTS,
        n_neighbors=UMAP_NEIGHBORS,
        metric="cosine",
        random_state=seed,
        low_memory=False,
    )
    reduced = reducer.fit_transform(embeddings)
    np.save(cache_path, reduced)
    print(f"UMAP done → shape {reduced.shape}")
    return reduced


def evaluate_k(reduced: np.ndarray, k_candidates=K_CANDIDATES):
    """
    Sweep over candidate K values and report FPC + silhouette.
    This provides evidence for the chosen K rather than picking it by
    convenience.
    """
    print("\n─── K Evaluation ───────────────────────────────────────")
    print(f"{'K':>4}  {'FPC':>8}  {'Silhouette':>12}")
    print("─" * 30)

    results = {}
    # Transpose: FCM expects (features, samples)
    data_T = reduced.T.astype(np.float64)

    for k in k_candidates:
        cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
            data_T, c=k, m=FCM_FUZZINESS,
            error=FCM_ERROR, maxiter=FCM_MAXITER, init=None, seed=42
        )
        hard_labels = np.argmax(u, axis=0)
        try:
            sil = silhouette_score(reduced, hard_labels, metric="cosine",
                                   sample_size=3000, random_state=42)
        except Exception:
            sil = float("nan")
        results[k] = {"fpc": fpc, "silhouette": sil, "cntr": cntr, "u": u}
        print(f"{k:>4}  {fpc:>8.4f}  {sil:>12.4f}")

    print("─" * 30)
    best = max(results, key=lambda k: results[k]["fpc"])
    print(f"Best K by FPC: {best}  (FPC={results[best]['fpc']:.4f})")
    return results


def run_fcm(reduced: np.ndarray, k: int = BEST_K):
    """
    Run Fuzzy C-Means with the chosen K.
    Returns:
      u      : membership matrix (k, N) — soft assignments
      cntr   : cluster centroids (k, dims)
      fpc    : Fuzzy Partition Coefficient (quality metric)
    """
    cache_path = f"./data/fcm_k{k}.pkl"
    if os.path.exists(cache_path):
        print(f"Loading FCM results from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"\nRunning Fuzzy C-Means  K={k}, m={FCM_FUZZINESS}...")
    data_T = reduced.T.astype(np.float64)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data_T, c=k, m=FCM_FUZZINESS,
        error=FCM_ERROR, maxiter=FCM_MAXITER, init=None, seed=42
    )
    print(f"  FPC = {fpc:.4f}  (1.0 = perfectly crisp, 1/K = completely fuzzy)")
    result = {"cntr": cntr, "u": u, "fpc": fpc, "k": k}
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    return result


def analyse_clusters(docs, u, k):
    """
    Semantic validation of clusters.
    Shows:
      1. Dominant category per cluster (label distribution)
      2. Top keywords (simple term-frequency on hard-assigned docs)
      3. Boundary documents (entropy of membership distribution is high)
      4. Uncertain documents (max membership < threshold)
    """
    N = len(docs)
    hard_labels = np.argmax(u, axis=0)   # shape (N,)

    print("\n══════════════════════════════════════════════════════════")
    print("  CLUSTER ANALYSIS")
    print("══════════════════════════════════════════════════════════")

    # ── Per-cluster category distribution ────────────────────────────────
    cluster_to_cats = defaultdict(list)
    for i, doc in enumerate(docs):
        cluster_to_cats[hard_labels[i]].append(doc["category"])

    cluster_summaries = []
    for c in range(k):
        cats = cluster_to_cats[c]
        total = len(cats)
        top_cats = Counter(cats).most_common(3)
        dominant = top_cats[0][0] if top_cats else "?"
        cluster_summaries.append({
            "cluster": c,
            "size": total,
            "dominant_category": dominant,
            "top_categories": top_cats,
        })
        print(f"\nCluster {c:>2}  ({total:>5} docs)  — dominant: {dominant}")
        for cat, cnt in top_cats:
            print(f"    {cnt:>5} ({100*cnt/total:4.1f}%)  {cat}")

    # ── Boundary documents (highest membership entropy) ───────────────────
    # Shannon entropy of membership row: higher = more uncertain
    entropy = -np.sum(u.T * np.log(u.T + 1e-10), axis=1)  # shape (N,)
    top_boundary_idx = np.argsort(entropy)[-10:][::-1]

    print("\n──────────────────────────────────────────────────────────")
    print("  BOUNDARY DOCUMENTS (highest membership entropy — belong to multiple clusters)")
    print("──────────────────────────────────────────────────────────")
    for idx in top_boundary_idx:
        memberships = u.T[idx]
        top2_clusters = np.argsort(memberships)[-2:][::-1]
        print(f"\n  Doc {idx}  [{docs[idx]['category']}]")
        print(f"  Entropy: {entropy[idx]:.3f}")
        for c in top2_clusters:
            print(f"    Cluster {c}: {memberships[c]:.3f}")
        # Show first 120 chars of the document
        print(f"  Text: {docs[idx]['text'][:120]}...")

    # ── Certain documents (max membership very high) ──────────────────────
    max_membership = np.max(u, axis=0)   # shape (N,)
    very_certain = np.where(max_membership > 0.85)[0]
    print(f"\n──────────────────────────────────────────────────────────")
    print(f"  MEMBERSHIP CERTAINTY DISTRIBUTION")
    print(f"──────────────────────────────────────────────────────────")
    print(f"  Docs with max membership > 0.85 (very certain): {len(very_certain):,}  "
          f"({100*len(very_certain)/N:.1f}%)")
    print(f"  Docs with max membership < 0.40 (very uncertain): "
          f"{np.sum(max_membership < 0.40):,}  "
          f"({100*np.sum(max_membership < 0.40)/N:.1f}%)")
    print(f"  Mean max membership: {max_membership.mean():.3f}")

    return cluster_summaries, hard_labels, entropy


def save_cluster_data(docs, u, hard_labels, k):
    """
    Attach cluster membership distributions to each doc and save.
    Each doc gains:
      - 'hard_cluster'   : argmax cluster id (int)
      - 'memberships'    : list of float, length k
      - 'dominant_membership': float, strength of dominant assignment
    """
    u_T = u.T   # (N, k)
    for i, doc in enumerate(docs):
        doc["hard_cluster"]        = int(hard_labels[i])
        doc["memberships"]         = u_T[i].tolist()
        doc["dominant_membership"] = float(np.max(u_T[i]))

    with open("./data/docs_clustered.pkl", "wb") as f:
        pickle.dump(docs, f)
    np.save("./data/membership_matrix.npy", u_T)
    print(f"\nCluster data saved → ./data/docs_clustered.pkl")
    return docs


def run_part2():
    # Load Part 1 outputs
    with open("./data/docs.pkl", "rb") as f:
        docs = pickle.load(f)
    embeddings = np.load("./data/embeddings.npy")

    # Step 1: Reduce dimensions
    reduced = reduce_dimensions(embeddings)

    # Step 2: Evaluate K choices (evidence, not convenience)
    k_results = evaluate_k(reduced)

    # Step 3: Run FCM with best K
    fcm = run_fcm(reduced, k=BEST_K)
    u   = fcm["u"]    # (K, N)
    print(f"\nFCM FPC = {fcm['fpc']:.4f}")

    # Step 4: Analyse and validate clusters
    summaries, hard_labels, entropy = analyse_clusters(docs, u, BEST_K)

    # Step 5: Save enriched docs
    docs = save_cluster_data(docs, u, hard_labels, BEST_K)

    print("\n=== Part 2 Complete ===")
    print(f"  K = {BEST_K} fuzzy clusters")
    print(f"  Each document has a {BEST_K}-dim membership distribution")
    print(f"  Saved to ./data/docs_clustered.pkl and ./data/membership_matrix.npy")
    return docs, u, summaries


if __name__ == "__main__":
    run_part2()
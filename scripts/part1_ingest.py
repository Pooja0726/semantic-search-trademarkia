import re
import os
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

CHROMA_PATH     = "./data/chroma_db"
EMBED_CACHE     = "./data/embeddings_cache.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers model
MIN_DOC_LENGTH  = 50   # characters after cleaning; shorter docs are semantically empty
BATCH_SIZE      = 256  # sentence-transformers encodes in batches; 256 fits comfortably in RAM


def clean_document(text: str) -> str:
    lines = text.split("\n")
    header_end = 0
    for i, line in enumerate(lines):
        if line.strip() == "" and i > 0:
            # Check if the block above contains header-like lines
            block = "\n".join(lines[:i])
            if re.search(r"^(From|Subject|Organization|Lines|Message-ID|NNTP|"
                         r"Path|Newsgroups|Distribution|Date|Xref|References"
                         r"|Reply-To|Sender|X-)[\s\S]*:", block, re.MULTILINE):
                header_end = i + 1
            break

    lines = lines[header_end:]

    # 2. Remove quoted reply lines
    lines = [l for l in lines if not l.strip().startswith(">")]

    # 3. Remove signature blocks (everything after "-- " on its own line)
    sig_start = None
    for i, line in enumerate(lines):
        if line.strip() in ("--", "-- "):
            sig_start = i
            break
    if sig_start is not None:
        lines = lines[:sig_start]

    # 4. Remove lines of only punctuation/dashes (decorative separators)
    lines = [l for l in lines if not re.match(r"^[-=_*#]{4,}$", l.strip())]

    # 5. Collapse and return
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_clean_dataset():
    """
    Fetch the full 20 Newsgroups dataset and apply cleaning.
    Returns a list of dicts with keys: id, text, category, split.
    """
    print("Fetching 20 Newsgroups dataset (train + test) via sklearn...")
    raw_train = fetch_20newsgroups(subset="train", remove=(), 
                                    download_if_missing=True)
    raw_test  = fetch_20newsgroups(subset="test",  remove=(), 
                                    download_if_missing=True)

    category_names = raw_train.target_names  # 20 category strings

    docs = []
    skipped = 0

    for split, raw in [("train", raw_train), ("test", raw_test)]:
        for idx, (text, label) in enumerate(zip(raw.data, raw.target)):
            cleaned = clean_document(text)
            if len(cleaned) < MIN_DOC_LENGTH:
                # Doc is too short to carry semantic meaning after cleaning
                skipped += 1
                continue
            docs.append({
                "id":       f"{split}_{idx}",
                "text":     cleaned,
                "category": category_names[label],
                "label":    int(label),
                "split":    split,
            })

    print(f"  Total docs loaded : {len(docs) + skipped:,}")
    print(f"  Docs after filter : {len(docs):,}  (dropped {skipped} too-short docs)")
    return docs, category_names


def embed_documents(docs, model_name=EMBEDDING_MODEL):
    """
    Encode all document texts into dense vectors.
    Uses a local sentence-transformers model — no API key or internet access
    required after the first download.

    Returns np.ndarray of shape (N, 384).
    """
    if os.path.exists(EMBED_CACHE):
        print(f"Loading embeddings from cache: {EMBED_CACHE}")
        with open(EMBED_CACHE, "rb") as f:
            return pickle.load(f)

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [d["text"] for d in docs]
    print(f"Encoding {len(texts):,} documents in batches of {BATCH_SIZE}...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalise → cosine sim = dot product
        convert_to_numpy=True,
    )

    os.makedirs("./data", exist_ok=True)
    with open(EMBED_CACHE, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings cached to {EMBED_CACHE}")
    return embeddings


def build_vector_store(docs, embeddings):
    """
    Persist embeddings + metadata into ChromaDB for filtered retrieval.

    Collection design:
      - One collection: 'newsgroups'
      - Each document stored with metadata: category, label, split
      - cosine distance metric (compatible with L2-normalised vectors)
      - IDs are stable across runs (split_idx format)
    """
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Drop and recreate for idempotent runs
    try:
        client.delete_collection("newsgroups")
    except Exception:
        pass

    collection = client.create_collection(
        name="newsgroups",
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Inserting {len(docs):,} documents into ChromaDB...")

    # ChromaDB add() works best with batches ≤ 5000
    batch_size = 2000
    for start in tqdm(range(0, len(docs), batch_size), desc="Chroma insert"):
        end   = min(start + batch_size, len(docs))
        batch = docs[start:end]
        collection.add(
            ids        = [d["id"]       for d in batch],
            embeddings = embeddings[start:end].tolist(),
            documents  = [d["text"]     for d in batch],
            metadatas  = [{
                "category": d["category"],
                "label":    d["label"],
                "split":    d["split"],
            } for d in batch],
        )

    count = collection.count()
    print(f"ChromaDB collection 'newsgroups' ready — {count:,} vectors indexed.")
    return client, collection


def run_part1():
    os.makedirs("./data", exist_ok=True)
    docs, category_names = load_and_clean_dataset()
    embeddings = embed_documents(docs)

    # Persist docs list for use in Parts 2 & 3
    with open("./data/docs.pkl", "wb") as f:
        pickle.dump(docs, f)
    np.save("./data/embeddings.npy", embeddings)

    client, collection = build_vector_store(docs, embeddings)

    print("\n=== Part 1 Complete ===")
    print(f"  Documents  : {len(docs):,}")
    print(f"  Embedding  : {EMBEDDING_MODEL}  →  {embeddings.shape[1]}-dim")
    print(f"  Vector DB  : ChromaDB (cosine)  →  {CHROMA_PATH}")
    print(f"  Categories : {category_names}")
    return docs, embeddings, category_names


if __name__ == "__main__":
    run_part1()
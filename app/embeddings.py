import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

# Using a lightweight but high-quality model
# all-MiniLM-L6-v2: 80MB, fast, good quality for semantic search
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Singleton pattern — load model once, reuse
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Convert list of text strings to embedding vectors.

    Returns:
        numpy array of shape (num_texts, embedding_dim)
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True  # normalized = cosine similarity works better
    )
    return np.array(embeddings, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS flat index for similarity search.
    IndexFlatIP = Inner Product (works as cosine sim when embeddings are normalized)
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # IP = cosine similarity (since normalized)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors of dimension {dim}")
    return index


def search_index(
    query: str,
    index: faiss.Index,
    chunks: List[str],
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    Search FAISS index for most relevant chunks to a query.

    Returns:
        List of (chunk_text, similarity_score) tuples
    """
    model = get_model()
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )
    query_vec = np.array(query_embedding, dtype=np.float32)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:  # -1 means no result found
            results.append((chunks[idx], float(score)))

    return results


def save_index(index: faiss.Index, chunks: List[str], save_dir: str):
    """Save FAISS index and chunks to disk for reuse."""
    os.makedirs(save_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(save_dir, "index.faiss"))
    with open(os.path.join(save_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    print(f"Index saved to {save_dir}")


def load_index(save_dir: str) -> Tuple[faiss.Index, List[str]]:
    """Load saved FAISS index and chunks from disk."""
    index = faiss.read_index(os.path.join(save_dir, "index.faiss"))
    with open(os.path.join(save_dir, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

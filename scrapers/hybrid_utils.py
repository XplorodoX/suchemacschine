"""
Shared utilities for hybrid (dense + sparse BM25) indexing and search.
Uses fastembed's Qdrant/bm25 model for proper BM25 sparse encoding.

Dense embeddings are centralised here too, so the model, its dimension and the
required task prefixes live in exactly one place. Switch the model via the
EMBEDDING_MODEL env var; every prepare/index script picks it up automatically.
"""

import os

from qdrant_client.models import SparseVector
from fastembed import SparseTextEmbedding

# --- Dense embedding model (one source of truth) ---
# multilingual-e5-base (768-dim) clearly beats the old MiniLM (384-dim) for
# German retrieval. e5 models require task prefixes: "passage: " for indexed
# documents and "query: " for search queries.
DENSE_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
_USE_E5_PREFIX = "e5" in DENSE_MODEL_NAME.lower()
_dense_model = None


def get_dense_model():
    """Lazily load and cache the dense SentenceTransformer model."""
    global _dense_model
    if _dense_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading dense embedding model ({DENSE_MODEL_NAME})...")
        _dense_model = SentenceTransformer(DENSE_MODEL_NAME)
        print(f"✓ Dense model loaded ({_dense_model.get_sentence_embedding_dimension()} dims)")
    return _dense_model


def dense_vector_size() -> int:
    """Embedding dimension — use this when creating Qdrant collections."""
    return get_dense_model().get_sentence_embedding_dimension()


def encode_passage(text: str) -> list:
    """Encode a single document/passage for indexing (applies e5 passage prefix)."""
    payload = f"passage: {text}" if _USE_E5_PREFIX else text
    return get_dense_model().encode(payload).tolist()


def encode_passages(texts: list[str]) -> list:
    """Batch-encode documents/passages for indexing (applies e5 passage prefix)."""
    payloads = [f"passage: {t}" for t in texts] if _USE_E5_PREFIX else list(texts)
    return [v.tolist() for v in get_dense_model().encode(payloads, show_progress_bar=True)]


def encode_query(text: str) -> list:
    """Encode a search query (applies e5 query prefix)."""
    payload = f"query: {text}" if _USE_E5_PREFIX else text
    return get_dense_model().encode(payload).tolist()


# Singleton: loaded once, reused everywhere
_sparse_model = None

def get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        print("Loading BM25 sparse model (Qdrant/bm25)...")
        _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        print("✓ BM25 sparse model loaded")
    return _sparse_model


def sparse_encode(text: str) -> SparseVector:
    """Encode text into a BM25 sparse vector using fastembed."""
    m = get_sparse_model()
    result = list(m.embed([text]))[0]
    return SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )


def sparse_encode_batch(texts: list[str]) -> list[SparseVector]:
    """Encode a batch of texts into BM25 sparse vectors (more efficient)."""
    m = get_sparse_model()
    results = list(m.embed(texts))
    return [
        SparseVector(
            indices=r.indices.tolist(),
            values=r.values.tolist(),
        )
        for r in results
    ]

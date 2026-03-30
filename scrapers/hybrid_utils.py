"""
Shared utilities for hybrid (dense + sparse BM25) indexing and search.
Uses fastembed's Qdrant/bm25 model for proper BM25 sparse encoding.
"""

from qdrant_client.models import SparseVector
from fastembed import SparseTextEmbedding

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

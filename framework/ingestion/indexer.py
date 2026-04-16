"""
QdrantIndexer — writes embedded chunks into a Qdrant vector collection.

Supports:
  - Legacy unnamed vectors (single dense vector per point)
  - Named vectors with optional sparse vectors (for hybrid RRF search)
  - Named vectors with ColBERT multi-vectors (Late Interaction / MaxSim)
  - All combinations of the above

The mode is chosen automatically based on whether chunks contain sparse
and/or colbert_vectors data.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

EXCLUDE_FROM_PAYLOAD = {"embedding", "sparse_indices", "sparse_values", "colbert_vectors"}

# Default ColBERT token embedding dimension.
# colbert-ir/colbertv2.0 and jinaai/jina-colbert-v2 both output 128-dim tokens.
DEFAULT_COLBERT_DIM = 128


class QdrantIndexer:
    """
    Indexes a list of Chunk dicts into a Qdrant collection.

    Args:
        collection:    Qdrant collection name (created/recreated on index())
        vector_size:   dense vector dimension (must match embedding model)
        colbert_dim:   ColBERT token vector dimension (default: 128)
        host:          Qdrant host (default: localhost)
        port:          Qdrant port (default: 6333)
        batch_size:    points per upsert batch (default: 500)
    """

    def __init__(
        self,
        collection: str,
        vector_size: int = 768,
        colbert_dim: int = DEFAULT_COLBERT_DIM,
        host: str = "localhost",
        port: int = 6333,
        batch_size: int = 500,
    ):
        self.collection = collection
        self.vector_size = vector_size
        self.colbert_dim = colbert_dim
        self.batch_size = batch_size

        try:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(host=host, port=port)
            log.info("✓ Qdrant connected at %s:%d", host, port)
        except Exception as e:
            raise RuntimeError(f"Could not connect to Qdrant at {host}:{port}: {e}")

    # ------------------------------------------------------------------
    # Feature detection
    # ------------------------------------------------------------------

    def _has_sparse(self, chunks: list[dict]) -> bool:
        return any("sparse_indices" in c for c in chunks[:5])

    def _has_colbert(self, chunks: list[dict]) -> bool:
        return any("colbert_vectors" in c for c in chunks[:5])

    # ------------------------------------------------------------------
    # Collection setup
    # ------------------------------------------------------------------

    def _recreate_collection(self, use_sparse: bool, use_colbert: bool) -> None:
        from qdrant_client.http.models import (
            Distance,
            VectorParams,
            SparseVectorParams,
            MultiVectorConfig,
            MultiVectorComparator,
        )

        log.info(
            "Creating collection '%s' (dim=%d, sparse=%s, colbert=%s, colbert_dim=%d)",
            self.collection, self.vector_size, use_sparse, use_colbert, self.colbert_dim,
        )

        dense_params = VectorParams(size=self.vector_size, distance=Distance.COSINE)

        if use_colbert:
            # ColBERT MultiVector: one 128-dim vector per token, scored via MaxSim
            colbert_params = VectorParams(
                size=self.colbert_dim,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM,
                ),
            )
            vectors_cfg = {"dense": dense_params, "colbert": colbert_params}
        else:
            vectors_cfg = None  # will decide below

        if use_colbert and use_sparse:
            self._client.recreate_collection(
                collection_name=self.collection,
                vectors_config=vectors_cfg,
                sparse_vectors_config={"sparse": SparseVectorParams()},
            )
        elif use_colbert:
            self._client.recreate_collection(
                collection_name=self.collection,
                vectors_config=vectors_cfg,
            )
        elif use_sparse:
            self._client.recreate_collection(
                collection_name=self.collection,
                vectors_config={"dense": dense_params},
                sparse_vectors_config={"sparse": SparseVectorParams()},
            )
        else:
            # Classic dense-only (unnamed vector, backwards compatible)
            self._client.recreate_collection(
                collection_name=self.collection,
                vectors_config=dense_params,
            )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, chunks: list[dict]) -> int:
        """
        (Re)create the collection and index all chunks.
        Returns number of indexed points.
        """
        from qdrant_client.http.models import PointStruct, SparseVector

        if not chunks:
            log.warning("No chunks to index into '%s'", self.collection)
            return 0

        use_sparse = self._has_sparse(chunks)
        use_colbert = self._has_colbert(chunks)
        self._recreate_collection(use_sparse, use_colbert)

        points = []
        total = 0

        for i, chunk in enumerate(chunks):
            embedding = chunk["embedding"]
            payload = {k: v for k, v in chunk.items() if k not in EXCLUDE_FROM_PAYLOAD}

            # Build vector dict based on which features are active
            if use_colbert or use_sparse:
                vector: dict = {"dense": embedding}

                if use_sparse and "sparse_indices" in chunk:
                    vector["sparse"] = SparseVector(
                        indices=chunk["sparse_indices"],
                        values=chunk["sparse_values"],
                    )

                if use_colbert and "colbert_vectors" in chunk:
                    # Qdrant MultiVector: pass list[list[float]] directly
                    vector["colbert"] = chunk["colbert_vectors"]
            else:
                # Legacy: single unnamed dense vector
                vector = embedding

            points.append(PointStruct(id=i, vector=vector, payload=payload))

            if len(points) >= self.batch_size:
                self._client.upsert(collection_name=self.collection, points=points)
                total += len(points)
                log.debug("  Upserted %d/%d points", total, len(chunks))
                points = []

        if points:
            self._client.upsert(collection_name=self.collection, points=points)
            total += len(points)

        log.info("Indexed %d points into '%s'", total, self.collection)
        return total

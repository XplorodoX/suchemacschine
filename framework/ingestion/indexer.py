"""
QdrantIndexer — writes embedded chunks into a Qdrant vector collection.

Supports both:
  - Legacy unnamed vectors (single dense vector per point)
  - Named vectors with optional sparse vectors (for hybrid RRF search)

The mode is chosen automatically based on whether chunks contain sparse data.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

EXCLUDE_FROM_PAYLOAD = {"embedding", "sparse_indices", "sparse_values"}


class QdrantIndexer:
    """
    Indexes a list of Chunk dicts into a Qdrant collection.

    Args:
        collection:    Qdrant collection name (created/recreated on index())
        vector_size:   dense vector dimension (must match embedding model)
        host:          Qdrant host (default: localhost)
        port:          Qdrant port (default: 6333)
        batch_size:    points per upsert batch (default: 500)
    """

    def __init__(
        self,
        collection: str,
        vector_size: int = 768,
        host: str = "localhost",
        port: int = 6333,
        batch_size: int = 500,
    ):
        self.collection = collection
        self.vector_size = vector_size
        self.batch_size = batch_size

        try:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(host=host, port=port)
            log.info("✓ Qdrant connected at %s:%d", host, port)
        except Exception as e:
            raise RuntimeError(f"Could not connect to Qdrant at {host}:{port}: {e}")

    def _has_sparse(self, chunks: list[dict]) -> bool:
        return any("sparse_indices" in c for c in chunks[:5])

    def _recreate_collection(self, use_sparse: bool):
        from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams

        log.info(
            "Creating collection '%s' (dim=%d, sparse=%s)",
            self.collection, self.vector_size, use_sparse,
        )

        if use_sparse:
            self._client.recreate_collection(
                collection_name=self.collection,
                vectors_config={"dense": VectorParams(size=self.vector_size, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams()},
            )
        else:
            self._client.recreate_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

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
        self._recreate_collection(use_sparse)

        points = []
        total = 0

        for i, chunk in enumerate(chunks):
            embedding = chunk["embedding"]
            payload = {k: v for k, v in chunk.items() if k not in EXCLUDE_FROM_PAYLOAD}

            if use_sparse and "sparse_indices" in chunk:
                vector = {
                    "dense": embedding,
                    "sparse": SparseVector(
                        indices=chunk["sparse_indices"],
                        values=chunk["sparse_values"],
                    ),
                }
            elif use_sparse:
                vector = {"dense": embedding}
            else:
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

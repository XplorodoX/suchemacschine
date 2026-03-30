#!/usr/bin/env python3
"""
Index HS Aalen processed data into Qdrant (hs_aalen_search collection)
Creates hybrid collection with dense + BM25 sparse vectors
"""
import json
import os
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, SparseVectorParams

# Add scrapers dir to path for hybrid_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scrapers'))
from hybrid_utils import sparse_encode

INPUT_FILE = os.getenv("INPUT_FILE", "/app/data/processed_data.jsonl")
# Also check local paths
if not os.path.exists(INPUT_FILE):
    INPUT_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_data.jsonl')

COLLECTION_NAME = "hs_aalen_search"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))


def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Delete old collection if exists
    try:
        client.get_collection(COLLECTION_NAME)
        print(f"Deleting old collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    # Create hybrid collection
    print(f"Creating hybrid collection '{COLLECTION_NAME}' (dense + BM25 sparse)...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=384, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(),
        },
    )

    print(f"Loading data from {INPUT_FILE}...")
    points = []
    count = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            embedding = record["embedding"]
            payload = {k: v for k, v in record.items() if k != "embedding"}

            # Build text for BM25 sparse encoding
            text = payload.get("text", "") or payload.get("content", "")
            title = payload.get("title", "")
            full_text = f"{title} {text[:2000]}"

            points.append(PointStruct(
                id=i,
                vector={
                    "dense": embedding,
                    "sparse": sparse_encode(full_text),
                },
                payload=payload,
            ))

            # Batch upload every 500 points
            if len(points) >= 500:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                count += len(points)
                print(f"  [{count}] Indexed...")
                points = []

    # Upload remaining points
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        count += len(points)

    info = client.get_collection(COLLECTION_NAME)
    print(f"✅ Indexing complete! Collection '{COLLECTION_NAME}': {info.points_count} points (hybrid: dense + BM25 sparse)")


if __name__ == "__main__":
    main()

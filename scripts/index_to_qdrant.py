import json
import os

import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
    SparseVectorParams,
    SparseVector,
)

# Configuration
INPUT_FILE = "/Users/merluee/Desktop/suchemacschine/processed_data.jsonl"
COLLECTION_NAME = "hs_aalen_search"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Vector dimension — must match the embedding model in prepare_data.py
# paraphrase-multilingual-MiniLM-L12-v2 → 384
# intfloat/multilingual-e5-base          → 768
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", 768))

# Set USE_SPARSE_VECTORS=true if prepare_data.py was run with sparse vectors
USE_SPARSE_VECTORS = os.getenv("USE_SPARSE_VECTORS", "false").lower() == "true"


def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    print(f"Creating collection '{COLLECTION_NAME}' (dim={VECTOR_SIZE}, sparse={USE_SPARSE_VECTORS})...")

    if USE_SPARSE_VECTORS:
        # Named vectors: "dense" for bi-encoder, "sparse" for SPLADE/BM42
        # Enables Qdrant's native hybrid RRF fusion search
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"dense": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )
    else:
        # Legacy unnamed vector (backwards compatible with existing app.py search)
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    print(f"Loading data from {INPUT_FILE}...")
    points = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in tqdm.tqdm(enumerate(f)):
            record = json.loads(line)
            embedding = record["embedding"]
            sparse_indices = record.get("sparse_indices")
            sparse_values = record.get("sparse_values")

            exclude_keys = {"embedding", "sparse_indices", "sparse_values"}
            payload = {k: v for k, v in record.items() if k not in exclude_keys}

            if USE_SPARSE_VECTORS and sparse_indices is not None:
                vector = {
                    "dense": embedding,
                    "sparse": SparseVector(indices=sparse_indices, values=sparse_values),
                }
            elif USE_SPARSE_VECTORS:
                # Sparse data missing — skip sparse for this point
                vector = {"dense": embedding}
            else:
                vector = embedding

            points.append(PointStruct(id=i, vector=vector, payload=payload))

            if len(points) >= 1000:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                points = []

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"Indexing complete! Collection '{COLLECTION_NAME}' is ready.")


if __name__ == "__main__":
    main()

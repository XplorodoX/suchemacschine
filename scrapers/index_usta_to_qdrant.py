#!/usr/bin/env python3
"""Index USTA Aalen data into Qdrant hybrid collection."""

import json
import logging
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, SparseVectorParams, VectorParams

from hybrid_utils import dense_vector_size, sparse_encode

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

COLLECTION_NAME = "usta_content"
VECTOR_SIZE = dense_vector_size()

try:
    client.get_collection(COLLECTION_NAME)
    logger.info(f"⚠️  Collection '{COLLECTION_NAME}' existiert — wird gelöscht...")
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass

logger.info(f"📦 Creating hybrid collection '{COLLECTION_NAME}'...")
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={"dense": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)},
    sparse_vectors_config={"sparse": SparseVectorParams()},
)
logger.info("✓ Hybrid collection erstellt (dense + BM25 sparse)")

logger.info("📂 Loading usta_indexed_data.jsonl...")
points = []

try:
    with open("usta_indexed_data.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            record = json.loads(line)
            title = record.get("title", "")
            content = record.get("content", "")[:2000]
            full_text = f"{title} {content}"
            points.append(PointStruct(
                id=i,
                vector={
                    "dense": record["embedding"],
                    "sparse": sparse_encode(full_text),
                },
                payload={
                    "url": record["url"],
                    "title": title,
                    "content": content,
                    "source": record["source"],
                    "type": record.get("type", "usta"),
                },
            ))
except FileNotFoundError:
    logger.error("❌ usta_indexed_data.jsonl not found — run prepare_usta_data.py first")
    exit(1)

logger.info(f"✓ {len(points)} Einträge geladen")

BATCH_SIZE = 100
for i in range(0, len(points), BATCH_SIZE):
    batch = points[i:i + BATCH_SIZE]
    client.upsert(collection_name=COLLECTION_NAME, points=batch)
    logger.info(f"   [{min(i+BATCH_SIZE, len(points))}/{len(points)}] Indexed...")

info = client.get_collection(COLLECTION_NAME)
logger.info(f"✅ USTA collection ready — {info.points_count} Punkte in '{COLLECTION_NAME}'")

#!/usr/bin/env python3
"""
Index HS Aalen Extended Data into Qdrant
Creates hybrid collection with dense + BM25 sparse vectors
"""

import json
import logging
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, SparseVectorParams
from sentence_transformers import SentenceTransformer
from hybrid_utils import sparse_encode

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Qdrant Client
QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

COLLECTION_NAME = "hs_aalen_website"
VECTOR_SIZE = 384

# Check if collection exists
try:
    existing = client.get_collection(COLLECTION_NAME)
    logger.info(f"⚠️  Collection '{COLLECTION_NAME}' already exists. Deleting...")
    client.delete_collection(COLLECTION_NAME)
except:
    pass

# Create hybrid collection (dense + sparse BM25)
logger.info(f"📦 Creating hybrid collection '{COLLECTION_NAME}'...")
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(),
    },
)
logger.info("✓ Hybrid collection created (dense + BM25 sparse)")

# Load and index data
logger.info("📂 Loading hs_aalen_indexed_data.jsonl...")
points = []

with open('hs_aalen_indexed_data.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        record = json.loads(line)
        
        # Build text for sparse encoding
        title = record.get('title', '')
        content = record.get('content', '')[:2000]
        full_text = f"{title} {content}"
        
        point = PointStruct(
            id=i,
            vector={
                "dense": record['embedding'],
                "sparse": sparse_encode(full_text),
            },
            payload={
                'url': record['url'],
                'title': title,
                'content': content,
                'source': record['source'],
                'type': record.get('type', 'webpage')
            }
        )
        points.append(point)

logger.info(f"✓ Loaded {len(points)} records with BM25 sparse vectors")

# Upsert points
logger.info("🔄 Indexing records in batches...")
batch_size = 500
for i in range(0, len(points), batch_size):
    batch = points[i : i + batch_size]
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch,
    )
    logger.info(f"   [{min(i + batch_size, len(points))}/{len(points)}] Indexed...")
logger.info(f"✓ Indexed {len(points)} records")

# Verify
info = client.get_collection(COLLECTION_NAME)
logger.info("✓ Collection created successfully:")
logger.info(f"   Collection: {COLLECTION_NAME}")
logger.info(f"   Points: {info.points_count}")

logger.info("\n✅ HS Aalen Website hybrid collection ready for search!")

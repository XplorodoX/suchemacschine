#!/usr/bin/env python3
"""
Index ASTA Data into Qdrant
Erstellt separate Collection nur für ASTA-Inhalte
"""

import json
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Qdrant Client
QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

COLLECTION_NAME = "asta_content"
VECTOR_SIZE = 384

# Check if collection exists
try:
    existing = client.get_collection(COLLECTION_NAME)
    logger.info(f"⚠️  Collection '{COLLECTION_NAME}' already exists. Deleting...")
    client.delete_collection(COLLECTION_NAME)
except:
    pass

# Create collection
logger.info(f"📦 Creating collection '{COLLECTION_NAME}'...")
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)
logger.info("✓ Collection created")

# Load and index data
logger.info("📂 Loading asta_indexed_data.jsonl...")
points = []

try:
    with open('asta_indexed_data.jsonl', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            record = json.loads(line)
            
            # Create point
            point = PointStruct(
                id=i,
                vector=record['embedding'],
                payload={
                    'url': record['url'],
                    'title': record['title'],
                    'content': record['content'][:500],  # Limit payload
                    'source': record['source'],
                    'type': record.get('type', 'asta')
                }
            )
            points.append(point)
except FileNotFoundError:
    logger.error("❌ asta_indexed_data.jsonl not found!")
    logger.error("   Please run prepare_asta_data.py first")
    exit(1)

logger.info(f"✓ Loaded {len(points)} records")

if len(points) == 0:
    logger.error("❌ No data to index!")
    exit(1)

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
logger.info("✓ ASTA Collection created successfully:")
logger.info(f"   Collection: {COLLECTION_NAME}")
logger.info(f"   Points: {info.points_count}")
logger.info(f"   Vectors size: {info.config.params.vectors.size}")

logger.info("\n✅ ASTA collection ready for search!")

# Verify all collections
logger.info("\n📊 Qdrant Collections Status:")
try:
    colls = client.get_collections()
    for coll in colls.collections:
        info = client.get_collection(coll.name)
        logger.info(f"   ✓ {coll.name}: {info.points_count} points")
except Exception as e:
    logger.warning(f"  Could not get collections: {e}")

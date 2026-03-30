#!/usr/bin/env python3
"""
Index Starplan Semester Data to Qdrant
Creates separate hybrid collections (dense + BM25 sparse) for each semester
"""

import json
import logging
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, SparseVectorParams
from hybrid_utils import sparse_encode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os

# Qdrant client
QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Semester mapping to collection names
SEMESTER_COLLECTIONS = {
    'SoSe26': 'starplan_SoSe26',
    'WS25': 'starplan_WS25',
    'SoSe25': 'starplan_SoSe25',
    'WS24': 'starplan_WS24',
}


def create_collection(collection_name: str):
    """Create hybrid Qdrant collection (dense + BM25 sparse)"""
    try:
        collections = client.get_collections()
        existing = [c.name for c in collections.collections]
        
        if collection_name in existing:
            logger.info(f"  Collection {collection_name} exists, deleting for re-creation...")
            client.delete_collection(collection_name)
        
        logger.info(f"  Creating hybrid collection {collection_name}...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=384, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )
        logger.info(f"  ✓ Hybrid collection {collection_name} created (dense + BM25 sparse)")
    except Exception as e:
        logger.error(f"  ✗ Error creating collection {collection_name}: {e}")
        raise


def index_semester_data(semester_code: str):
    """Index Starplan data for a semester with hybrid vectors"""
    
    logger.info(f"\n📚 Indexing {semester_code}...")
    
    collection_name = SEMESTER_COLLECTIONS.get(semester_code, f"starplan_{semester_code}")
    input_file = f"starplan_{semester_code}_indexed_data.jsonl"
    
    if not Path(input_file).exists():
        logger.warning(f"  ⚠️  File {input_file} not found, skipping...")
        return
    
    create_collection(collection_name)
    
    points = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            record = json.loads(line)
            
            metadata = record.get('metadata', {})
            content = record.get('content', '')
            title = metadata.get('title', '')
            
            # Build full text for BM25 sparse encoding
            full_text = f"{title} {metadata.get('lecturer', '')} {metadata.get('day', '')} {metadata.get('time', '')} {metadata.get('room', '')} {content}"
            
            point = PointStruct(
                id=i,
                vector={
                    "dense": record['embedding'],
                    "sparse": sparse_encode(full_text),
                },
                payload={
                    'title': title,
                    'lecturer': metadata.get('lecturer', ''),
                    'day': metadata.get('day', ''),
                    'time': metadata.get('time', ''),
                    'room': metadata.get('room', ''),
                    'semester': semester_code,
                    'type': 'timetable',
                    'content': content
                }
            )
            points.append(point)
    
    if not points:
        logger.warning(f"  ⚠️  No points to index for {semester_code}")
        return
    
    logger.info(f"  Uploading {len(points)} points with BM25 sparse vectors...")
    
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch
        )
        if (i + batch_size) % 500 == 0 or i == 0:
            logger.info(f"    [{min(i+batch_size, len(points))}/{len(points)}] Uploaded...")
    
    logger.info(f"  ✅ {semester_code}: {len(points)} points indexed to {collection_name}")


def verify_all_collections():
    """Verify all semester collections"""
    logger.info("\n🔍 Verifying all semester collections...")
    
    try:
        collections = client.get_collections()
        
        total_points = 0
        for semester_code, collection_name in SEMESTER_COLLECTIONS.items():
            for collection in collections.collections:
                if collection.name == collection_name:
                    points_count = collection.points_count
                    total_points += points_count
                    logger.info(f"  ✓ {semester_code:8} ({collection_name:20}): {points_count:6} points")
        
        logger.info(f"\n  📊 Total semester points: {total_points}")
        logger.info("  🎯 Semester collections ready!")
        
    except Exception as e:
        logger.error(f"  ✗ Error verifying collections: {e}")


def main():
    logger.info("📚 Starting Semester Data Indexing (Hybrid: Dense + BM25 Sparse)...")
    
    for semester_code in SEMESTER_COLLECTIONS.keys():
        try:
            index_semester_data(semester_code)
        except Exception as e:
            logger.error(f"  ✗ Error processing {semester_code}: {e}")
    
    verify_all_collections()
    logger.info("\n✅ Semester indexing complete!")


if __name__ == "__main__":
    main()

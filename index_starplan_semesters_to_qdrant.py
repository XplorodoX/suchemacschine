#!/usr/bin/env python3
"""
Index Starplan Semester Data to Qdrant
Erstellt separate Collections für jedes Semester
"""

import json
import logging
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Qdrant client
client = QdrantClient("localhost", port=6333)

# Semester mapping to collection names
SEMESTER_COLLECTIONS = {
    'SoSe26': 'starplan_SoSe26',
    'WS25': 'starplan_WS25',
    'SoSe25': 'starplan_SoSe25',
    'WS24': 'starplan_WS24',
}


def create_collection(collection_name: str):
    """Create Qdrant collection mit correct config"""
    try:
        # Check if collection exists
        collections = client.get_collections()
        existing = [c.name for c in collections.collections]
        
        if collection_name not in existing:
            logger.info(f"  Creating collection {collection_name}...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            logger.info(f"  ✓ Collection {collection_name} created")
        else:
            logger.info(f"  Collection {collection_name} already exists, updating...")
    except Exception as e:
        logger.error(f"  ✗ Error creating collection {collection_name}: {e}")
        raise


def index_semester_data(semester_code: str):
    """Index Starplan data für ein Semester"""
    
    logger.info(f"\n📚 Indexing {semester_code}...")
    
    # Get collection name
    collection_name = SEMESTER_COLLECTIONS.get(semester_code, f"starplan_{semester_code}")
    
    # Load indexed data
    input_file = f"starplan_{semester_code}_indexed_data.jsonl"
    
    if not Path(input_file).exists():
        logger.warning(f"  ⚠️  File {input_file} not found, skipping...")
        return
    
    # Create collection
    create_collection(collection_name)
    
    # Load and index records
    points = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            record = json.loads(line)
            
            point = PointStruct(
                id=i,
                vector=record['embedding'],
                payload={
                    'title': record['metadata'].get('title', ''),
                    'lecturer': record['metadata'].get('lecturer', ''),
                    'day': record['metadata'].get('day', ''),
                    'time': record['metadata'].get('time', ''),
                    'room': record['metadata'].get('room', ''),
                    'semester': semester_code,
                    'type': 'timetable',
                    'content': record['content']
                }
            )
            points.append(point)
    
    if not points:
        logger.warning(f"  ⚠️  No points to index for {semester_code}")
        return
    
    # Upload points
    logger.info(f"  Uploading {len(points)} points...")
    
    # Upload in batches
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
        logger.info(f"  🎯 Semester collections ready!")
        
    except Exception as e:
        logger.error(f"  ✗ Error verifying collections: {e}")


def main():
    logger.info("📚 Starting Semester Data Indexing...")
    
    for semester_code in SEMESTER_COLLECTIONS.keys():
        try:
            index_semester_data(semester_code)
        except Exception as e:
            logger.error(f"  ✗ Error processing {semester_code}: {e}")
    
    # Verify
    verify_all_collections()
    logger.info("\n✅ Semester indexing complete!")


if __name__ == "__main__":
    main()

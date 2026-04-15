#!/usr/bin/env python3
"""
Index Starplan timetable data into Qdrant
Creates hybrid collection with dense + BM25 sparse vectors
"""

import json
import logging
import os
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, SparseVectorParams

# Add current dir to path for hybrid_utils
sys.path.insert(0, os.path.dirname(__file__))
from hybrid_utils import sparse_encode

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))


class StarplanQdrantIndexer:
    def __init__(self):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection_name = 'starplan_timetable'
        self.vector_size = 384

    def create_collection(self):
        """Create hybrid collection for timetable data (dense + BM25 sparse)"""
        try:
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' exists, deleting for re-creation...")
            self.client.delete_collection(self.collection_name)
        except:
            pass

        logger.info(f"Creating hybrid collection '{self.collection_name}'...")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(size=self.vector_size, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )
        logger.info("✓ Hybrid collection created (dense + BM25 sparse)")

    def load_records(self, filename='starplan_indexed_data.jsonl'):
        """Load records from JSONL file"""
        logger.info(f"Loading records from {filename}...")

        records = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    records.append(record)

        logger.info(f"Loaded {len(records)} records")
        return records

    def index_records(self, records):
        """Index records into Qdrant with hybrid vectors"""
        logger.info(f"Indexing {len(records)} records with BM25 sparse vectors...")

        points = []
        for i, record in enumerate(records):
            # Build text for BM25 sparse encoding
            full_text = f"{record.get('program', '')} {record.get('day', '')} {record.get('time', '')} {record.get('lecture_info', '')} {record.get('full_text', '')}"

            point = PointStruct(
                id=i,
                vector={
                    "dense": record['embedding'],
                    "sparse": sparse_encode(full_text),
                },
                payload={
                    'source': record.get('source', ''),
                    'program': record.get('program', ''),
                    'program_id': record.get('program_id', ''),
                    'day': record.get('day', ''),
                    'time': record.get('time', ''),
                    'lecture_info': record.get('lecture_info', ''),
                    'type': record.get('type', 'timetable'),
                    'url': record.get('url', ''),
                    'full_text': record.get('full_text', ''),
                }
            )
            points.append(point)

        # Upsert in batches
        batch_size = 500
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            logger.info(f"   [{min(i + batch_size, len(points))}/{len(points)}] Indexed...")

        logger.info(f"✓ Indexed {len(records)} records")

    def verify_index(self):
        """Verify indexing was successful"""
        info = self.client.get_collection(self.collection_name)
        logger.info(f"\n✓ Collection: {self.collection_name}")
        logger.info(f"  Points: {info.points_count}")
        return info.points_count


def main():
    logger.info("=" * 70)
    logger.info("Starplan Timetable Qdrant Indexing (Hybrid: Dense + BM25 Sparse)")
    logger.info("=" * 70 + "\n")

    try:
        indexer = StarplanQdrantIndexer()
        indexer.create_collection()
        records = indexer.load_records()
        indexer.index_records(records)
        point_count = indexer.verify_index()

        logger.info(f"\n✅ Successfully indexed {point_count} timetable records with BM25 sparse!")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

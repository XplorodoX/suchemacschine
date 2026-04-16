#!/usr/bin/env python3
"""
Index Starplan timetable data into Qdrant
"""

import json
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class StarplanQdrantIndexer:
    def __init__(self, qdrant_host='localhost', qdrant_port=6333):
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = 'starplan_timetable'
        self.vector_size = 384  # all-MiniLM-L6-v2 produces 384-dim vectors
    
    def create_collection_if_needed(self):
        """Create collection for timetable data"""
        try:
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists")
        except:
            logger.info(f"Creating collection '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )
    
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
        """Index records into Qdrant"""
        logger.info(f"Indexing {len(records)} records into Qdrant...")
        
        points = []
        for i, record in enumerate(records):
            # Create point with embedding and metadata
            point = PointStruct(
                id=i,
                vector=record['embedding'],
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
        
        # Upsert points into Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.info(f"✓ Indexed {len(records)} records")
    
    def verify_index(self):
        """Verify indexing was successful"""
        info = self.client.get_collection(self.collection_name)
        logger.info(f"\nCollection info:")
        logger.info(f"  Name: {self.collection_name}")
        logger.info(f"  Points: {info.points_count}")
        logger.info(f"  Vectors size: {info.config.params.vectors.size}")
        
        return info.points_count
    
    def test_search(self, query_embedding):
        """Test search with a query embedding"""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=5
        )
        
        logger.info(f"\nSearch results:")
        for i, result in enumerate(results, 1):
            payload = result.payload
            logger.info(f"  {i}. {payload['program']} - {payload['time']} ({payload['lecture_info'][:50]})")
            logger.info(f"     Score: {result.score:.3f}")

def main():
    logger.info("=" * 70)
    logger.info("Starplan Timetable Qdrant Indexing")
    logger.info("=" * 70 + "\n")
    
    try:
        # Create indexer
        indexer = StarplanQdrantIndexer()
        
        # Create collection
        indexer.create_collection_if_needed()
        
        # Load records
        records = indexer.load_records()
        
        # Index records
        indexer.index_records(records)
        
        # Verify
        point_count = indexer.verify_index()
        
        logger.info(f"\n✓ Successfully indexed {point_count} timetable records!")
        logger.info("\nYou can now search by asking:")
        logger.info("  - 'AI Stundenplan'")
        logger.info("  - 'Montag 9 Uhr Informatik'")
        logger.info("  - 'Vorlesung mit Raum und Zeit'")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

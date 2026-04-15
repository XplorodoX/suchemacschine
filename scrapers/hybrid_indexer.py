#!/usr/bin/env python3
"""
Unified Hybrid Indexer: indexes all data sources into Qdrant
with dense (sentence-transformers) + BM25 sparse (fastembed) vectors.
"""
import json
import logging
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, SparseVectorParams
from sentence_transformers import SentenceTransformer
from hybrid_utils import sparse_encode

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

COLLECTION_NAME = "hs_aalen_hybrid"


def setup_collection():
    try:
        client.get_collection(COLLECTION_NAME)
        logger.info(f"Collection '{COLLECTION_NAME}' exists, deleting for re-creation...")
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    logger.info(f"Creating hybrid collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=384, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams()
        }
    )
    logger.info("✓ Hybrid collection created (dense + BM25 sparse)")


def index_file(filename, source_name, p_type="webpage"):
    if not os.path.exists(filename):
        logger.warning(f"File {filename} not found, skipping.")
        return
    
    logger.info(f"📂 Loading {filename}...")
    points = []
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                record = json.loads(line)
                text = record.get('content') or record.get('text') or record.get('full_text') or record.get('description') or ""
                title = record.get('title') or record.get('modul') or record.get('lecture_info') or "Unbenannt"
                
                # Full text for BM25 sparse encoding
                full_text = f"{title} {text} {record.get('program', '')} {record.get('raum', '')} {source_name}"
                
                # Create unique UUID
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source_name}_{i}"))
                
                # Create hybrid point
                point = PointStruct(
                    id=point_id,
                    vector={
                        "dense": record.get('embedding') or [0]*384,
                        "sparse": sparse_encode(full_text),
                    },
                    payload={
                        'url': record.get('url'),
                        'title': title,
                        'content': text[:2000],
                        'source': source_name,
                        'type': record.get('type') or p_type,
                        'modul': record.get('modul'),
                        'start_iso': record.get('start_time'),
                        'raum': record.get('raum'),
                        'instructor': record.get('instructor'),
                        'studiengang': record.get('program')
                    }
                )
                points.append(point)
            except Exception as e:
                logger.error(f"Error parsing line {i} in {filename}: {e}")

    if not points: return

    # Generate dense embeddings for points that don't have them
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    logger.info(f"✨ Checking embeddings for {len(points)} points...")
    zero_vec = [0]*384
    for p in points:
        if p.vector["dense"] == zero_vec:
            p.vector["dense"] = model.encode(p.payload['title'] + " " + p.payload['content'][:500]).tolist()

    logger.info(f"🔄 Upserting {len(points)} points from {source_name}...")
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i : i + batch_size]
        )
    logger.info(f"✅ Finished indexing {source_name}")

if __name__ == "__main__":
    setup_collection()
    index_file('hs_aalen_indexed_data.jsonl', 'hs_aalen', 'webpage')
    index_file('asta_indexed_data.jsonl', 'asta', 'asta')
    index_file('starplan_ical_data.jsonl', 'starplan', 'timetable')
    logger.info("🎬 Hybrid indexing complete!")

#!/usr/bin/env python3
"""
Qdrant Indexer Service
Indexiert embeddings in Qdrant
"""
import os
import json
import glob
from datetime import datetime
from typing import Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

INPUT_DIR = os.getenv("INPUT_DIR", "/data/embeddings")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTIONS = os.getenv("COLLECTIONS", "hs_aalen_search,hs_aalen_website,timetable_data").split(",")

def index_to_qdrant():
    """Main indexing function."""
    print(f"🗂️  Starting Qdrant indexer at {datetime.now()}")
    print(f"   Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"   Collections: {COLLECTIONS}")
    
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        print("   ✅ Connected to Qdrant")
    except Exception as e:
        print(f"   ❌ Failed to connect to Qdrant: {e}")
        return
    
    # Find input files
    input_files = glob.glob(os.path.join(INPUT_DIR, "*.jsonl"))
    
    if not input_files:
        print(f"⚠️  No input files found in {INPUT_DIR}")
        return
    
    input_file = input_files[0]
    print(f"   Input file: {os.path.basename(input_file)}")
    
    # Read and index data
    points = []
    point_id = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                embedding = record.get('embedding', [])
                
                if not embedding or len(embedding) == 0:
                    continue
                
                point_id += 1
                
                # Create point
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "url": record.get('url', ''),
                        "title": record.get('title', ''),
                        "text": record.get('text', ''),
                        "source": record.get('source', 'unknown'),
                        "chunk_id": record.get('chunk_id', 0),
                    }
                )
                points.append(point)
                
                if line_num % 100 == 0:
                    print(f"   Loaded {line_num} records...")
            
            except Exception as e:
                print(f"   ⚠️  Error on line {line_num}: {e}")
    
    print(f"\n   Total points to index: {len(points)}")
    
    if not points:
        print("⚠️  No valid points to index")
        return
    
    # Get vector dimension
    vector_size = len(points[0].vector)
    print(f"   Vector dimension: {vector_size}")
    
    # Create/update collections
    for collection_name in COLLECTIONS:
        collection_name = collection_name.strip()
        
        try:
            # Check if collection exists
            try:
                client.get_collection(collection_name)
                print(f"   Collection '{collection_name}' exists, clearing...")
                client.delete_collection(collection_name)
            except:
                pass  # Collection doesn't exist
            
            # Create collection
            print(f"   Creating collection '{collection_name}'...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            
            # Upload points
            print(f"   Uploading {len(points)} points...")
            client.upsert(
                collection_name=collection_name,
                points=points,
            )
            
            print(f"   ✅ Collection '{collection_name}' indexed successfully")
        
        except Exception as e:
            print(f"   ❌ Error with collection '{collection_name}': {e}")
    
    print(f"\n✅ Indexing complete!")

if __name__ == "__main__":
    index_to_qdrant()

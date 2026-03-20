import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import tqdm

# Configuration
INPUT_FILE = "/Users/merluee/Desktop/suchemacschine/processed_data.jsonl"
COLLECTION_NAME = "hs_aalen_search"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

def main():
    # Initialize Qdrant client
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Create collection (recreate if exists for a fresh start)
    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    
    print(f"Loading data from {INPUT_FILE}...")
    points = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in tqdm.tqdm(enumerate(f)):
            record = json.loads(line)
            url = record["url"]
            text = record["text"]
            embedding = record["embedding"]
            
            points.append(PointStruct(
                id=i,
                vector=embedding,
                payload={"url": url, "text": text}
            ))
            
            # Batch upload every 1000 points
            if len(points) >= 1000:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                points = []

    # Upload remaining points
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        
    print(f"Indexing complete! Collection '{COLLECTION_NAME}' is ready.")

if __name__ == "__main__":
    main()

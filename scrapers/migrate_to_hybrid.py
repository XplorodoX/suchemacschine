import json
import os
import re
import unicodedata
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

COLLECTION_NAME = "hs_aalen_hybrid"
SOURCE_FILE = "hs_aalen_indexed_data.jsonl"

GERMAN_STOPWORDS = {"der", "die", "das", "ein", "eine", "einer", "einem", "einen", "und", "oder", "aber", "mit", "ohne", "auf", "in", "im", "am", "an", "zu", "zum", "zur", "von", "für", "ist", "sind", "war", "wie", "was", "wer", "wo", "wann", "warum", "welche", "welcher", "welches", "ich", "du", "er", "sie", "es", "wir", "ihr", "nicht", "kein", "keine", "mehr", "auch", "den", "dem", "des", "bei", "über", "unter", "nach"}

def normalize_text(text):
    text = (text or "").lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    normalized = normalize_text(text)
    tokens = re.findall(r"[a-zA-Z0-9äöüÄÖÜß]{2,}", normalized)
    return [t for t in tokens if t not in GERMAN_STOPWORDS]

def sparse_encode(text):
    tokens = tokenize(text)
    if not tokens:
        return {"indices": [], "values": []}
    counts = {}
    for t in tokens:
        # Use a simple rolling hash for word indices
        idx = hash(t) % 1000000
        counts[idx] = counts.get(idx, 0) + 1.0
    return {"indices": list(counts.keys()), "values": list(counts.values())}

if not os.path.exists(SOURCE_FILE):
    print(f"❌ Source file {SOURCE_FILE} not found!")
    exit(1)

print(f"🔄 Migrating data from {SOURCE_FILE} to {COLLECTION_NAME}...")

points = []
with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        record = json.loads(line)
        content = record.get('content', '')
        title = record.get('title', '')
        
        # Combined text for sparse encoding
        full_text = f"{title} {content}"
        sparse_vec = sparse_encode(full_text)
        
        point = PointStruct(
            id=i,
            vector={
                "dense": record['embedding'],
                "sparse": sparse_vec
            },
            payload={
                'url': record['url'],
                'title': record['title'],
                'content': record.get('content', '')[:2000],
                'source': record.get('source', 'hs_aalen'),
                'type': record.get('type', 'webpage')
            }
        )
        points.append(point)

# Upsert in batches
batch_size = 100
for i in range(0, len(points), batch_size):
    batch = points[i : i + batch_size]
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch,
    )
    print(f"   [{min(i + batch_size, len(points))}/{len(points)}] Migrated...")

print(f"✅ Successfully migrated {len(points)} points to '{COLLECTION_NAME}'.")

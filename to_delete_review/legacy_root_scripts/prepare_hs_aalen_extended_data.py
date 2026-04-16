#!/usr/bin/env python3
"""
Prepare HS Aalen Extended Data with Embeddings
Generiert SentenceTransformer Embeddings für alle gescrapten Seiten
"""

import json
import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Laden des Modells
logger.info("📦 Loading SentenceTransformer model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Laden der Daten
logger.info("📂 Loading hs_aalen_extended_data.json...")
with open('hs_aalen_extended_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

pages = data['pages']
logger.info(f"✓ Loaded {len(pages)} pages")

# Generiere Embeddings für jede Seite
logger.info("🧠 Generating embeddings...")
output_records = []

for i, page in enumerate(pages, 1):
    # Erstelle combined text für Embedding
    full_text = f"{page['title']} {page['content']}"
    
    # Generiere Embedding
    embedding = model.encode(full_text).tolist()
    
    # Erstelle Record
    record = {
        'url': page['url'],
        'title': page['title'],
        'content': page['content'][:1000],  # Limit content für Payload
        'full_text': full_text,
        'embedding': embedding,
        'source': 'hs_aalen_website',
        'type': 'webpage'
    }
    
    output_records.append(record)
    
    if i % 50 == 0:
        logger.info(f"  [{i}/{len(pages)}] Generated embeddings...")

# Speichere JSONL Format (1 record pro Zeile)
logger.info(f"💾 Saving to hs_aalen_indexed_data.jsonl...")
with open('hs_aalen_indexed_data.jsonl', 'w', encoding='utf-8') as f:
    for record in output_records:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

logger.info(f"✓ Generated embeddings: {len(output_records)} records")
logger.info(f"✓ Embedding dimension: {len(output_records[0]['embedding'])}")
logger.info(f"✓ Done! Data ready for Qdrant indexing")

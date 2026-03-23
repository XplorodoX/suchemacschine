#!/usr/bin/env python3
"""
Prepare ASTA Data with Embeddings
Generiert SentenceTransformer Embeddings für ASTA Inhalte
"""

import json
import logging

from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Laden des Modells
logger.info("📦 Loading SentenceTransformer model...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Laden der ASTA Daten
logger.info("📂 Loading asta_full_data.json...")
try:
    with open('asta_full_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    logger.error("❌ asta_full_data.json not found!")
    logger.error("   Please run asta_full_scraper.py first")
    exit(1)

pages = data['pages']
logger.info(f"✓ Loaded {len(pages)} ASTA pages")

if len(pages) == 0:
    logger.error("❌ No pages loaded!")
    exit(1)

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
        'source': 'asta_website',
        'type': 'asta'
    }
    
    output_records.append(record)
    
    if i % 50 == 0:
        logger.info(f"  [{i}/{len(pages)}] Generated embeddings...")

# Speichere JSONL Format (1 record pro Zeile)
logger.info("💾 Saving to asta_indexed_data.jsonl...")
with open('asta_indexed_data.jsonl', 'w', encoding='utf-8') as f:
    for record in output_records:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

logger.info(f"✓ Generated embeddings: {len(output_records)} records")
logger.info(f"✓ Embedding dimension: {len(output_records[0]['embedding'])}")
logger.info("✓ Done! Data ready for Qdrant indexing")

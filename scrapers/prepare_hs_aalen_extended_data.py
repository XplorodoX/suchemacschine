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
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

import os
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
INPUT_FILE = os.path.join(DATA_DIR, "hs_aalen_extended_data.jsonl")
OUTPUT_FILE = os.path.join(DATA_DIR, "hs_aalen_indexed_data.jsonl")

# Laden der Daten
logger.info(f"📂 Loading {INPUT_FILE}...")
output_records = []

try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            page = json.loads(line)
            
            # Erstelle combined text für Embedding
            # Wir nehmen jetzt mehr Text mit (2500 chars) für besseres Ranking
            title = page.get('title', '')
            content = page.get('content', '')[:2500]
            full_text = f"{title} {content}"
            
            # Generiere Embedding
            embedding = model.encode(full_text).tolist()
            
            # Erstelle Record
            record = {
                'url': page['url'],
                'title': title,
                'content': content,
                'full_text': full_text,
                'embedding': embedding,
                'source': 'hs_aalen_website',
                'type': 'webpage'
            }
            
            output_records.append(record)
            
            if i % 50 == 0:
                logger.info(f"  Processed {i} pages...")
except FileNotFoundError:
    logger.error(f"❌ Input file {INPUT_FILE} not found!")
    exit(1)

# Speichere JSONL Format (1 record pro Zeile)
logger.info(f"💾 Saving to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for record in output_records:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

if output_records:
    logger.info(f"✓ Generated embeddings: {len(output_records)} records")
    logger.info(f"✓ Embedding dimension: {len(output_records[0]['embedding'])}")
else:
    logger.warning("⚠️  No records generated!")

logger.info("✓ Done! Data ready for Qdrant indexing")

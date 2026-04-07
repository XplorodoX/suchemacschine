#!/usr/bin/env python3
"""
Prepare HS Aalen Extended Data with Embeddings
Generiert SentenceTransformer Embeddings für alle gescrapten Seiten
"""

import json
import logging
import re

from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _extract_pdf_section_text(page: dict) -> str:
    """Return concatenated text from sections that contain PDF content."""
    sections = page.get('sections') or []
    pdf_text_parts = []
    for section in sections:
        heading = (section.get('heading') or '').lower()
        if 'pdf' in heading:
            text = (section.get('text') or '').strip()
            if text:
                pdf_text_parts.append(text)
    return "\n".join(pdf_text_parts)


def _snippet_for_pdf_filename(pdf_section_text: str, filename: str) -> str:
    """Try to extract a short snippet for one PDF from the aggregated PDF section text."""
    if not pdf_section_text or not filename:
        return ""

    # Match blocks like: [PDF: some-file.pdf]\n<text>
    pattern = re.compile(
        rf"\[PDF:\s*{re.escape(filename)}\]\s*(.*?)(?=\n\s*\[PDF:|$)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(pdf_section_text)
    if not match:
        return ""

    snippet = re.sub(r"\s+", " ", match.group(1)).strip()
    return snippet[:1200]

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
            pdf_section_text = _extract_pdf_section_text(page)[:2000]
            full_text = f"{title} {content} {pdf_section_text}".strip()
            
            # Generiere Embedding
            embedding = model.encode(full_text).tolist()
            
            # Erstelle Record
            record = {
                'url': page['url'],
                'title': title,
                'content': content,
                'full_text': full_text,
                'embedding': embedding,
                'source': page.get('source', 'hs_aalen_website'),
                'type': page.get('type', 'webpage'),
                'pdf_sources': page.get('pdf_sources', []),
                'pdf_count': page.get('pdf_count', 0),
                'sections': page.get('sections', []),
            }
            
            output_records.append(record)

            # Zusätzlich eigene PDF-Treffer erzeugen, damit PDFs direkt als Suchergebnis erscheinen.
            for pdf in page.get('pdf_sources', []):
                pdf_url = pdf.get('url')
                if not pdf_url:
                    continue

                filename = (pdf.get('filename') or '').strip()
                pdf_title = filename or pdf_url.rsplit('/', 1)[-1]
                pdf_snippet = _snippet_for_pdf_filename(pdf_section_text, filename)
                pdf_text = f"{title} {pdf_title} {pdf_snippet}".strip()
                if not pdf_text:
                    continue

                pdf_record = {
                    'url': pdf_url,
                    'title': pdf_title,
                    'content': pdf_snippet,
                    'full_text': pdf_text,
                    'embedding': model.encode(pdf_text).tolist(),
                    'source': 'hs_aalen_pdfs',
                    'type': 'pdf',
                    'parent_url': page.get('url'),
                }
                output_records.append(pdf_record)
            
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

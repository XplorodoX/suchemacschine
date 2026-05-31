#!/usr/bin/env python3
"""
Prepare HS Aalen Extended Data with Embeddings
Splits long pages into overlapping chunks so every part of a page is searchable.
Each chunk gets its own dense embedding via encode_passage.
"""

import json
import logging
import os
import re

from hybrid_utils import encode_passage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _extract_pdf_section_text(page: dict) -> str:
    sections = page.get('sections') or []
    parts = []
    for section in sections:
        if 'pdf' in (section.get('heading') or '').lower():
            text = (section.get('text') or '').strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


def _snippet_for_pdf_filename(pdf_section_text: str, filename: str) -> str:
    if not pdf_section_text or not filename:
        return ""
    pattern = re.compile(
        rf"\[PDF:\s*{re.escape(filename)}\]\s*(.*?)(?=\n\s*\[PDF:|$)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(pdf_section_text)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()[:1200]


DATA_DIR = os.getenv("DATA_DIR", "/app/data")
INPUT_FILE = os.path.join(DATA_DIR, "hs_aalen_extended_data.jsonl")
OUTPUT_FILE = os.path.join(DATA_DIR, "hs_aalen_indexed_data.jsonl")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks, preferring sentence boundaries."""
    if len(text) <= CHUNK_SIZE:
        return [text] if text.strip() else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        if end < len(text):
            for sep in ['. ', '! ', '? ', '\n', ' ']:
                pos = text.rfind(sep, start + CHUNK_SIZE // 2, end)
                if pos > 0:
                    end = pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP
    return chunks


logger.info(f"📂 Loading {INPUT_FILE}...")
output_records = []

try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        pages = [json.loads(line) for line in f]
except FileNotFoundError:
    logger.error(f"❌ Input file {INPUT_FILE} not found!")
    exit(1)

logger.info(f"✓ {len(pages)} pages loaded — chunking and embedding...")

for i, page in enumerate(pages, 1):
    title = page.get('title', '')
    full_content = page.get('content', '')
    pdf_section_text = _extract_pdf_section_text(page)[:2000]
    source = page.get('source', 'hs_aalen_website')
    page_type = page.get('type', 'webpage')
    pdf_sources = page.get('pdf_sources', [])

    chunks = chunk_text(full_content) or [full_content[:CHUNK_SIZE]]
    chunk_total = len(chunks)

    for chunk_idx, chunk in enumerate(chunks):
        prefix = f"{title} {pdf_section_text}" if chunk_idx == 0 else title
        full_text = f"{prefix} {chunk}".strip()
        output_records.append({
            'url': page['url'],
            'title': title,
            'content': chunk,
            'full_text': full_text,
            'embedding': encode_passage(full_text),
            'source': source,
            'type': page_type,
            'pdf_sources': pdf_sources if chunk_idx == 0 else [],
            'pdf_count': page.get('pdf_count', 0) if chunk_idx == 0 else 0,
            'sections': page.get('sections', []) if chunk_idx == 0 else [],
            'chunk_index': chunk_idx,
            'chunk_total': chunk_total,
        })

    for pdf in pdf_sources:
        pdf_url = pdf.get('url')
        if not pdf_url:
            continue
        filename = (pdf.get('filename') or '').strip()
        pdf_title = filename or pdf_url.rsplit('/', 1)[-1]
        pdf_snippet = _snippet_for_pdf_filename(pdf_section_text, filename)
        pdf_text = f"{title} {pdf_title} {pdf_snippet}".strip()
        if not pdf_text:
            continue
        output_records.append({
            'url': pdf_url,
            'title': pdf_title,
            'content': pdf_snippet,
            'full_text': pdf_text,
            'embedding': encode_passage(pdf_text),
            'source': 'hs_aalen_pdfs',
            'type': 'pdf',
            'parent_url': page.get('url'),
            'chunk_index': 0,
            'chunk_total': 1,
        })

    if i % 50 == 0:
        logger.info(f"  [{i}/{len(pages)}] {len(output_records)} records so far...")

logger.info(f"💾 Saving {len(output_records)} records to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for record in output_records:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

logger.info(f"✓ Generated embeddings: {len(output_records)} records")
logger.info(f"✓ Embedding dimension: {len(output_records[0]['embedding'])}")
logger.info("✓ Done! Data ready for Qdrant indexing")

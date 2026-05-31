#!/usr/bin/env python3
"""Prepare USTA Aalen data: generate embeddings from scraped pages."""

import json
import logging

from hybrid_utils import encode_passage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("📂 Loading usta_full_data.json...")
try:
    with open("usta_full_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    logger.error("❌ usta_full_data.json not found — run usta_scraper.py first")
    exit(1)

pages = data["pages"]
logger.info(f"✓ {len(pages)} Seiten geladen")

if not pages:
    logger.error("❌ Keine Seiten vorhanden")
    exit(1)

logger.info("🧠 Generating embeddings...")
records = []
for i, page in enumerate(pages, 1):
    full_text = f"{page['title']} {page['content']}"
    record = {
        "url": page["url"],
        "title": page["title"],
        "content": page["content"][:1000],
        "full_text": full_text,
        "embedding": encode_passage(full_text),
        "source": "usta_aalen",
        "type": "usta",
    }
    records.append(record)
    if i % 50 == 0:
        logger.info(f"  [{i}/{len(pages)}] Embeddings generiert...")

logger.info("💾 Saving to usta_indexed_data.jsonl...")
with open("usta_indexed_data.jsonl", "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

logger.info(f"✅ {len(records)} Einträge gespeichert")

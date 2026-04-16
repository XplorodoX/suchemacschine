#!/usr/bin/env python3
"""
Comprehensive PDF integration:
1. Read data.jsonl
2. Extract PDFs for curriculum/module pages
3. Generate sentence embeddings
4. Output to processed_data.jsonl
"""

import json
import sys
from pathlib import Path
import requests
from collections import defaultdict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pdf_extractor import extract_pdfs_from_page
from sentence_transformers import SentenceTransformer

INPUT_FILE = PROJECT_ROOT / "data.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "processed_data.jsonl"

# URL patterns for PDF-relevant pages
PDF_RELEVANT_KEYWORDS = [
    "curriculum", "stundenplan", "program", "master", "studienplan",
    "modul", "kurs", "course",  "dokument", "information"
]

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def is_pdf_relevant(url: str) -> bool:
    """Check if page likely contains relevant PDFs."""
    url_lower = url.lower()
    return any(kw in url_lower for kw in PDF_RELEVANT_KEYWORDS)


def create_session():
    """Configured requests session."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "HSAalenSearchBot/1.1 (+https://www.hs-aalen.de)",
    })
    retry = Retry(total=2, connect=2, read=2, backoff_factor=0.3,
                  status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def augment_with_pdfs(record: dict, session: requests.Session) -> dict:
    """Optionally augment record with PDF content."""
    url = record.get("url", "")
    if not url or not is_pdf_relevant(url):
        return record

    try:
        print(f"  Fetching PDFs from {url}...")
        response = session.get(url, timeout=12)
        response.raise_for_status()

        pdf_data = extract_pdfs_from_page(response.text, url, session)
        if pdf_data:
            print(f"    ✓ Found {len(pdf_data)} PDF(s)")
            
            # Store PDF info
            record["pdf_count"] = len(pdf_data)
            record["pdf_sources"] = [
                {"filename": pdf["filename"], "url": pdf["url"]}
                for pdf in pdf_data
            ]
            
            # Augment content with PDF text
            pdf_text = " ".join([pdf["text"] for pdf in pdf_data])
            record["content"] = record.get("content", "") + " " + pdf_text
            
            # Store in sections if available
            if "sections" in record:
                record["sections"].append({
                    "heading": "PDF-Dokumente",
                    "text": "\n".join([f"{pdf['filename']}: {pdf['text'][:300]}" 
                                        for pdf in pdf_data])
                })

    except Exception as e:
        print(f"    ✗ Error fetching PDFs: {e}")
        record["pdf_count"] = 0

    return record


def main():
    """Main processing pipeline."""
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    session = create_session()
    processed_count = 0
    pdf_count = 0

    print(f"\nReading: {INPUT_FILE}")
    print(f"Writing: {OUTPUT_FILE}\n")

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:

        for line in infile:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except:
                continue

            # Augment with PDFs if relevant
            record = augment_with_pdfs(record,session)
            if record.get("pdf_count", 0) > 0:
                pdf_count += 1

            # Generate embedding
            content = record.get("content", "")
            if content and len(content.strip()) > 20:
                try:
                    embedding = model.encode(content).tolist()
                    record["embedding"] = embedding
                except:
                    pass

            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed_count += 1

            if processed_count % 5 == 0:
                print(f"  Progress: {processed_count} records, {pdf_count} with PDFs")

    print(f"\n✓ Complete!")
    print(f"  Total records: {processed_count}")
    print(f"  With PDFs: {pdf_count}")
    print(f"  Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

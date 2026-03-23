from typing import Dict

#!/usr/bin/env python3
"""
Post-processing script to extract PDFs from URLs and augment JSONL data with PDF content.
This runs after the main scraper to add PDF documents to relevant pages.
"""

import json
import sys

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add scripts to path
sys.path.insert(0, '/home/flo/suchemacschine/scripts')
from pdf_extractor import extract_pdfs_from_page

INPUT_FILE = "/home/flo/suchemacschine/data.jsonl"
OUTPUT_FILE = "/home/flo/suchemacschine/data_with_pdfs.jsonl"

PDF_KEYWORDS = [
    "curriculum", "studienplan", "modul", "syllabus", "course", "program",
    "termplan", "stundenplan", "dokument", "richtlinie", "guideline"
]


def is_pdf_relevant_url(url: str) -> bool:
    """Check if URL likely has relevant PDFs."""
    url_lower = url.lower()
    return any(keyword in url_lower for keyword in PDF_KEYWORDS)


def create_session():
    """Create a configured requests session."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "HSAalenSearchBot/1.1 (+https://www.hs-aalen.de)",
    })
    retry = Retry(
        total=3, connect=3, read=3, backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def process_record_with_pdfs(record: Dict, session: requests.Session) -> Dict:
    """Extract PDFs from a record's URL and augment the record."""
    url = record.get("url")
    if not url or not is_pdf_relevant_url(url):
        return record

    try:
        # Download the page
        response = session.get(url, timeout=15)
        response.raise_for_status()

        # Extract PDFs
        pdf_data = extract_pdfs_from_page(response.text, url, session)

        if pdf_data:
            print(f"✓ Found {len(pdf_data)} PDF(s) from {url}")

            # Add PDFs to record
            record["pdf_count"] = len(pdf_data)
            record["pdf_sources"] = [
                {"url": pdf["url"], "filename": pdf["filename"]}
                for pdf in pdf_data
            ]

            # Add PDF content to full text
            pdf_text = " ".join([pdf["text"] for pdf in pdf_data])
            original_content = record.get("content", "")
            record["content"] = original_content + " " + pdf_text
            record["content_length"] = len(record["content"])

            # Add to sections
            if "sections" not in record:
                record["sections"] = []

            pdf_section = {
                "heading": "PDF-Dokumente",
                "text": "\n\n".join([
                    f"[PDF: {pdf['filename']}]\n{pdf['text'][:400]}"
                    for pdf in pdf_data
                ])
            }
            record["sections"].append(pdf_section)

    except Exception as e:
        print(f"✗ Error processing PDFs for {url}: {e}")

    return record


def main():
    """Process all records and add PDF content where available."""
    session = create_session()
    processed_count = 0
    pdf_added_count = 0

    print(f"Reading from: {INPUT_FILE}")
    print(f"Writing to: {OUTPUT_FILE}\n")

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        for i, line in enumerate(infile):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except Exception as e:
                print(f"✗ Line {i}: Invalid JSON - {e}")
                continue

            processed_count += 1

            # Try to add PDFs if relevant URL
            updated_record = process_record_with_pdfs(record, session)
            if updated_record.get("pdf_count", 0) > 0:
                pdf_added_count += 1

            # Write the record (with or without PDFs)
            outfile.write(json.dumps(updated_record, ensure_ascii=False) + "\n")

            if processed_count % 20 == 0:
                print(f"Progress: {processed_count} records processed, {pdf_added_count} with PDFs")

    print("\n✓ Complete!")
    print(f"  Total records: {processed_count}")
    print(f"  Records with PDFs: {pdf_added_count}")
    print(f"  Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

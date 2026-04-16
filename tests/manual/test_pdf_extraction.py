#!/usr/bin/env python3
"""Quick test to verify PDF extraction functionality."""

import sys
<<<<<<<< HEAD:tests/manual/test_pdf_extraction.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
========
>>>>>>>> origin/main:tests/test_pdf_extraction.py

sys.path.insert(0, '/home/flo/suchemacschine/scripts')

from scraper import BrowserRenderer, create_session, extract_content

# Test with a specific curriculum page that might have PDFs
test_urls = [
    "https://www.hs-aalen.de/studienangebot/informatik-kuenstliche-intelligenz/master/informatik/curriculum/",
    "https://www.hs-aalen.de/forschung/institute/",
]

session = create_session()
renderer = BrowserRenderer()

for url in test_urls:
    print(f"\n{'='*60}")
    print(f"Testing: {url}")
    print('='*60)
    
    result = extract_content(session, renderer, url)
    if result:
        pdf_count = result.get('pdf_count', 0)
        print("\n✓ Scraped successfully")
        print(f"  Content length: {result.get('content_length')} chars")
        print(f"  PDFs found: {pdf_count}")
        if result.get('pdf_sources'):
            for pdf in result['pdf_sources']:
                print(f"    - {pdf['filename']}: {pdf['url']}")
    else:
        print("\n✗ Failed to scrape")

renderer.close()

#!/usr/bin/env python3
"""Quick test to verify PDF extraction functionality."""

import sys
sys.path.insert(0, '/Users/merluee/Desktop/suchemacschine/scripts')

from scraper import create_session, extract_content, BrowserRenderer

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
        print(f"\n✓ Scraped successfully")
        print(f"  Content length: {result.get('content_length')} chars")
        print(f"  PDFs found: {pdf_count}")
        if result.get('pdf_sources'):
            for pdf in result['pdf_sources']:
                print(f"    - {pdf['filename']}: {pdf['url']}")
    else:
        print(f"\n✗ Failed to scrape")

renderer.close()

#!/usr/bin/env python3
"""Test PDF extraction in full scraper."""

import json
import sys
sys.path.insert(0, '/Users/merluee/Desktop/suchemacschine/scripts')

from scraper import create_session, extract_content, BrowserRenderer

session = create_session()
renderer = BrowserRenderer()

# Test with curriculum page (should have PDF)
url = "https://www.hs-aalen.de/studienangebot/informatik-kuenstliche-intelligenz/master/informatik/curriculum/"

print(f"Testing full scraper with: {url}\n")
result = extract_content(session, renderer, url)

if result:
    print(f"✓ Scraping successful")
    print(f"  Title: {result.get('title')}")
    print(f"  H1: {result.get('h1')}")
    print(f"  Content length: {result.get('content_length')} chars")
    print(f"  Sections: {len(result.get('sections', []))}")
    print(f"  PDFs found: {result.get('pdf_count', 0)}")
    
    if result.get('pdf_sources'):
        print(f"\n  PDF Files:")
        for pdf in result['pdf_sources']:
            print(f"    - {pdf['filename']}")
    
    # Show first section with PDF content
    for section in result.get('sections', []):
        if 'PDF' in section.get('heading', ''):
            print(f"\n  PDF Section Preview:")
            print(f"    {section['text'][:200]}...")
            break
else:
    print(f"✗ Scraping failed")

renderer.close()

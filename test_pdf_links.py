#!/usr/bin/env python3
"""Quick test for PDF extraction only."""

import sys
import requests
sys.path.insert(0, '/Users/merluee/Desktop/suchemacschine/scripts')

from pdf_extractor import find_pdf_links
from bs4 import BeautifulSoup

session = requests.Session()
session.headers.update({
    "User-Agent": "HSAalenSearchBot/1.1 (+https://www.hs-aalen.de)",
})

test_url = "https://www.hs-aalen.de/studienangebot/informatik-kuenstliche-intelligenz/master/informatik/curriculum/"

print(f"Testing PDF link extraction from: {test_url}")
try:
    response = session.get(test_url, timeout=10)
    response.raise_for_status()
    
    pdf_links = find_pdf_links(response.text, test_url)
    print(f"Found {len(pdf_links)} PDF links:")
    for link in pdf_links[:5]:  # Show first 5
        print(f"  - {link}")
        
except Exception as e:
    print(f"Error: {e}")

#!/usr/bin/env python3
"""Debug PDF link extraction."""

import sys
import requests
sys.path.insert(0, '/Users/merluee/Desktop/suchemacschine/scripts')

from pdf_extractor import find_pdf_links
from bs4 import BeautifulSoup

session = requests.Session()
session.headers.update({
    "User-Agent": "HSAalenSearchBot/1.1 (+https://www.hs-aalen.de)",
})

url = "https://www.hs-aalen.de/studienangebot/informatik-kuenstliche-intelligenz/master/informatik/curriculum/"

print(f"Testing PDF link extraction from: {url}\n")
try:
    response = session.get(url, timeout=10)
    response.raise_for_status()
    
    # Check raw HTML for PDF links
    html = response.text
    
    # Check for .pdf patterns
    import re
    pdf_patterns = re.findall(r'href=["\']([^"\']*\.pdf[^"\']*)["\']', html, re.IGNORECASE)
    print(f"Regex found {len(pdf_patterns)} PDF patterns:")
    for p in pdf_patterns[:5]:
        print(f"  - {p}")
    
    # Now test our function
    pdf_links = find_pdf_links(html, url)
    print(f"\nfind_pdf_links() found {len(pdf_links)} links:")
    for link in pdf_links[:5]:
        print(f"  - {link}")
        
    # Also check all links
    soup = BeautifulSoup(html, "html.parser")
    all_links = soup.find_all("a", href=True)
    pdf_in_all = [link.get("href") for link in all_links if ".pdf" in link.get("href", "").lower()]
    print(f"\nDirect BeautifulSoup search found {len(pdf_in_all)} PDF links:")
    for link in pdf_in_all[:5]:
        print(f"  - {link}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

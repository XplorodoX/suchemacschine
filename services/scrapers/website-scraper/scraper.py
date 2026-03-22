#!/usr/bin/env python3
"""
Website Scraper Service
Scrapt HS-Aalen Website und speichert als JSONL
"""
import os
import json
import time
import hashlib
from datetime import datetime
from typing import List, Dict
from urllib.parse import urljoin, urlparse
from collections import deque

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/data")
TARGET_URLS = os.getenv("TARGET_URLS", "https://www.hs-aalen.de").split(",")
MAX_DEPTH = int(os.getenv("MAX_DEPTH", "3"))
TIMEOUT_MINUTES = int(os.getenv("TIMEOUT_MINUTES", 60))

def create_session():
    """Create requests session with retries."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def extract_text(html: str) -> str:
    """Extract clean text from HTML."""
    soup = BeautifulSoup(html, 'lxml')
    
    # Remove unwanted elements
    for element in soup.find_all(['script', 'style', 'nav', 'footer']):
        element.decompose()
    
    text = soup.get_text(separator=' ', strip=True)
    # Clean up whitespace
    text = ' '.join(text.split())
    return text

def extract_links(html: str, base_url: str) -> List[str]:
    """Extract all links from HTML."""
    soup = BeautifulSoup(html, 'lxml')
    links = []
    
    for a in soup.find_all('a', href=True):
        url = urljoin(base_url, a['href'])
        # Filter to same domain only
        if urlparse(url).netloc == urlparse(base_url).netloc:
            # Remove fragments
            url = url.split('#')[0]
            if url not in links:
                links.append(url)
    
    return links

def scrape_website():
    """Main scraping function."""
    print(f"🕷️  Starting website scraper at {datetime.now()}")
    print(f"   Target URLs: {TARGET_URLS}")
    print(f"   Max depth: {MAX_DEPTH}")
    
    session = create_session()
    visited = set()
    to_visit = deque([(url.strip(), 0) for url in TARGET_URLS])
    results = []
    
    start_time = time.time()
    timeout_sec = TIMEOUT_MINUTES * 60
    
    while to_visit and (time.time() - start_time) < timeout_sec:
        url, depth = to_visit.popleft()
        
        if url in visited or depth > MAX_DEPTH:
            continue
        
        visited.add(url)
        print(f"   [{len(visited)}] Scraping: {url}")
        
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            
            text = extract_text(response.text)
            
            if len(text.strip()) > 100:  # Only save if meaningful content
                results.append({
                    "url": url,
                    "title": BeautifulSoup(response.text, 'lxml').title.string or url,
                    "text": text,
                    "scraped_at": datetime.now().isoformat(),
                    "source": "website"
                })
            
            # Find new links
            if depth < MAX_DEPTH:
                links = extract_links(response.text, url)
                for link in links:
                    if link not in visited:
                        to_visit.append((link, depth + 1))
        
        except Exception as e:
            print(f"   ⚠️  Error scraping {url}: {e}")
        
        time.sleep(0.5)  # Be nice to the server
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "website_data.jsonl")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Scraped {len(results)} pages")
    print(f"   Saved to: {output_file}")

if __name__ == "__main__":
    scrape_website()

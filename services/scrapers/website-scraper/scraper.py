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
DOWNLOAD_PDFS = os.getenv("DOWNLOAD_PDFS", "true").lower() == "true"
PDF_DOWNLOAD_DIR = os.getenv("PDF_DOWNLOAD_DIR", "/pdf_sources")

NON_HTML_EXTENSIONS = (
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".7z", ".jpg", ".jpeg", ".png", ".gif", ".webp",
    ".mp4", ".mp3", ".wav"
)

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
    soup = BeautifulSoup(html, 'html.parser')
    
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

def is_probably_non_html_url(url: str) -> bool:
    """Check URL extension to avoid fetching binary documents as web pages."""
    path = urlparse(url).path.lower()
    # If we want to download PDFs, we don't treat them as "skip-worthy" non-HTML here,
    # but we handle them separately in the main loop.
    if DOWNLOAD_PDFS and path.endswith(".pdf"):
        return False
    return path.endswith(NON_HTML_EXTENSIONS)

def download_pdf(session: requests.Session, url: str) -> bool:
    """Download PDF file to the designated directory."""
    try:
        os.makedirs(PDF_DOWNLOAD_DIR, exist_ok=True)
        filename = os.path.basename(urlparse(url).path)
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"
        
        filepath = os.path.join(PDF_DOWNLOAD_DIR, filename)
        
        # Skip if already exists
        if os.path.exists(filepath):
            return True
            
        print(f"      📥 Downloading PDF: {filename}")
        response = session.get(url, timeout=20)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"      ⚠️  Failed to download PDF {url}: {e}")
        return False

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

        if is_probably_non_html_url(url):
            print(f"   ⏭️  Skipping non-HTML resource: {url}")
            continue
        
        # Handle PDF downloading
        if DOWNLOAD_PDFS and urlparse(url).path.lower().endswith(".pdf"):
            download_pdf(session, url)
            continue
        
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                print(f"   ⏭️  Skipping non-HTML content-type ({content_type or 'unknown'}): {url}")
                continue
            
            text = extract_text(response.text)
            soup = BeautifulSoup(response.text, 'lxml')
            title = url
            if soup.title and soup.title.string:
                title = soup.title.string.strip() or url
            
            if len(text.strip()) > 100:  # Only save if meaningful content
                results.append({
                    "url": url,
                    "title": title,
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

#!/usr/bin/env python3
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import httpx
import requests
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# Add parent directory to sys.path for imports from backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.pdf_extractor import download_and_extract_pdf, chunk_text

logging.basicConfig(
    level=os.getenv("LOGLEVEL", "INFO"), 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
# Note: The search backend uses hs_aalen_hybrid as the main collection for the web app
COLLECTION_NAME = "hs_aalen_hybrid"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
USER_AGENT = "HS-Aalen-Bot/1.0"
GERMAN_STOPWORDS = {"der", "die", "das", "ein", "eine", "einer", "einem", "einen", "und", "oder", "aber", "mit", "ohne", "auf", "in", "im", "am", "an", "zu", "zum", "zur", "von", "für", "ist", "sind", "war", "wie", "was", "wer", "wo", "wann", "warum", "welche", "welcher", "welches", "ich", "du", "er", "sie", "es", "wir", "ihr", "nicht", "kein", "keine", "mehr", "auch", "den", "dem", "des", "bei", "über", "unter", "nach"}

class UniversalScraper:
    def __init__(self, base_url: str, max_pages: int = 500):
        self.base_url = base_url.rstrip('/')
        self.domain = urlparse(base_url).netloc
        self.visited_urls: Set[str] = set()
        self.found_pdf_urls: Set[str] = set()
        self.disallowed_patterns: List[str] = []
        self.sitemaps: List[str] = []
        self.max_pages = max_pages
        
        # Clients
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.model = SentenceTransformer(MODEL_NAME)
        self.httpx_client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={'User-Agent': USER_AGENT}
        )
        self.requests_session = requests.Session()
        self.requests_session.headers.update({'User-Agent': USER_AGENT})

    async def fetch_robots_txt(self):
        """Fetch and parse robots.txt for sitemaps and disallows"""
        robots_url = f"{urlparse(self.base_url).scheme}://{self.domain}/robots.txt"
        logger.info(f"📋 Checking robots.txt at {robots_url}...")
        try:
            resp = await self.httpx_client.get(robots_url)
            if resp.status_code == 200:
                current_agents = []
                relevant_block = False
                
                for line in resp.text.split('\n'):
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    
                    # Sitemap is usually global, but can be in blocks
                    if line.lower().startswith('sitemap:'):
                        self.sitemaps.append(line.split(':', 1)[1].strip())
                        continue

                    if line.lower().startswith('user-agent:'):
                        agent = line.split(':', 1)[1].strip().lower()
                        # If a new block starts, reset
                        if not line.lower().startswith('user-agent:'): # (Redundant check)
                            pass
                        
                        # We care about * or our specific name
                        if agent in ['*', USER_AGENT.lower()]:
                            relevant_block = True
                        else:
                            # If it was relevant but now it's another agent, stop unless it's a multi-agent block
                            # Simplified: if it's a new User-agent line and not ours, we start checking anew
                            relevant_block = (agent in ['*', USER_AGENT.lower()])
                        continue

                    if relevant_block:
                        if line.lower().startswith('disallow:'):
                            pattern = line.split(':', 1)[1].strip()
                            if pattern: self.disallowed_patterns.append(pattern)
                        elif line.lower().startswith('allow:'):
                            # We could support Allow: too if needed, but Disallow is priority
                            pass
            
            # De-duplicate sitemaps
            self.sitemaps = list(set(self.sitemaps))
            if not self.sitemaps:
                self.sitemaps.append(f"{self.base_url}/sitemap.xml")
                
            logger.info(f"✓ Found {len(self.sitemaps)} unique sitemaps and {len(self.disallowed_patterns)} relevant disallows")
        except Exception as e:
            logger.warning(f"⚠️  robots.txt fetch failed: {e}")
            self.sitemaps.append(f"{self.base_url}/sitemap.xml")

    def is_allowed(self, url: str) -> bool:
        """Check if URL is allowed based on robots.txt and domain"""
        parsed = urlparse(url)
        
        # Normalize domains for comparison (ignore www)
        def normalize_domain(d):
            return d.lower().replace('www.', '')

        if parsed.netloc and normalize_domain(parsed.netloc) != normalize_domain(self.domain):
            logger.debug(f"  [Blocked] Domain mismatch: {parsed.netloc} != {self.domain}")
            return False
            
        path = parsed.path or "/"
        for pattern in self.disallowed_patterns:
            # Handle simple wildcard /*
            if pattern == '/*':
                logger.debug(f"  [Blocked] Pattern /* matches everything")
                return False
            
            # Simple prefix match for Disallow rules
            if path.startswith(pattern):
                logger.debug(f"  [Blocked] Pattern {pattern} matches {path}")
                return False
                
        return True

    def table_to_markdown(self, table) -> str:
        """Convert a BeautifulSoup table into Markdown format."""
        rows = []
        for tr in table.find_all("tr"):
            cells = [cell.get_text(strip=True) for cell in tr.find_all(["th", "td"])]
            if cells:
                rows.append("| " + " | ".join(cells) + " |")
        
        if not rows: return ""
        
        # Add separator after header
        if len(rows) > 1:
            header_cells = len(rows[0].split("|")) - 2
            separator = "| " + " | ".join(["---"] * header_cells) + " |"
            rows.insert(1, separator)
            
        return "\n" + "\n".join(rows) + "\n"

    async def discover_urls_from_sitemaps(self) -> List[str]:
        """Recursively discover URLs from sitemaps"""
        all_urls = []
        sitemap_queue = list(self.sitemaps)
        visited_sitemaps = set()

        logger.info("🗺️  Discovering URLs from sitemaps...")
        while sitemap_queue:
            s_url = sitemap_queue.pop(0)
            if s_url in visited_sitemaps: continue
            visited_sitemaps.add(s_url)

            try:
                resp = await self.httpx_client.get(s_url)
                if resp.status_code != 200: continue
                
                soup = BeautifulSoup(resp.text, 'xml')
                
                # If it's a sitemapindex, the <loc> tags point to further sitemaps
                # If it's a urlset, the <loc> tags point to pages
                locs = [loc.text for loc in soup.find_all('loc')]
                
                if soup.find('sitemapindex'):
                    logger.info(f"  [Index] Found {len(locs)} nested sitemaps in {s_url}")
                    sitemap_queue.extend(locs)
                else:
                    logger.info(f"  [URLSet] Found {len(locs)} URLs in {s_url}")
                    all_urls.extend(locs)
            except Exception as e:
                logger.debug(f"  Error parsing sitemap {s_url}: {e}")

        logger.info(f"✓ Total {len(all_urls)} URLs found in sitemaps")
        return all_urls

    def normalize_text(self, t: str) -> str:
        t = (t or "").lower()
        t = unicodedata.normalize("NFKC", t)
        return re.sub(r"\s+", " ", t).strip()

    def tokenize(self, t: str) -> List[str]:
        return [w for w in re.findall(r"[a-zA-Z0-9äöüß]{2,}", self.normalize_text(t)) if w not in GERMAN_STOPWORDS]

    def sparse_encode(self, text: str) -> models.SparseVector:
        tokens = self.tokenize(text)
        if not tokens: return models.SparseVector(indices=[], values=[])
        counts = {}
        for tok in tokens:
            idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % 1000000
            counts[idx] = counts.get(idx, 0) + 1.0
        return models.SparseVector(indices=list(counts.keys()), values=list(counts.values()))

    async def process_url(self, url: str):
        """Process a single URL: Extract content and index to Qdrant"""
        if url in self.visited_urls or not self.is_allowed(url): return
        self.visited_urls.add(url)

        try:
            if url.lower().endswith('.pdf'):
                await self.process_pdf(url)
            else:
                await self.process_html(url)
        except Exception as e:
            logger.error(f"  [Error] Failed to process {url}: {e}")

    def table_to_markdown(self, table) -> str:
        """Convert a BeautifulSoup table into Markdown format."""
        rows = []
        for tr in table.find_all("tr"):
            cells = [cell.get_text(strip=True) for cell in tr.find_all(["th", "td"])]
            if cells:
                rows.append("| " + " | ".join(cells) + " |")
        
        if not rows: return ""
        
        # Add separator after header
        if len(rows) > 1:
            header_cells = len(rows[0].split("|")) - 2
            separator = "| " + " | ".join(["---"] * header_cells) + " |"
            rows.insert(1, separator)
            
        return "\n" + "\n".join(rows) + "\n"

    async def process_html(self, url: str):
        resp = await self.httpx_client.get(url)
        if resp.status_code != 200: return
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Convert tables to markdown before removing tags
        for table in soup.find_all("table"):
            md_table = self.table_to_markdown(table)
            table.replace_with(md_table)

        # PDF discovery from page
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_pdf_url = urljoin(url, href)
            if full_pdf_url.lower().endswith('.pdf') and full_pdf_url not in self.found_pdf_urls:
                self.found_pdf_urls.add(full_pdf_url)

        # Content extraction
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        
        title_tag = soup.find('h1') or soup.find('title')
        title = title_tag.get_text(strip=True) if title_tag else "No title"
        content = soup.get_text(separator=' ', strip=True)
        content = re.sub(r'\s+', ' ', content)
        
        if len(content) < 200: return

        # Indexing (Hybrid)
        chunks = chunk_text(content, max_chars=1200, overlap=150)
        source_name = self.domain.replace('www.', '').split('.')[0]
        
        points = []
        for i, chunk in enumerate(chunks):
            dense_vec = self.model.encode(chunk).tolist()
            sparse_vec = self.sparse_encode(chunk)
            
            point = models.PointStruct(
                id=int(hashlib.md5(f"{url}_{i}".encode()).hexdigest()[:12], 16),
                vector={
                    "dense": dense_vec,
                    "sparse": sparse_vec
                },
                payload={
                    "url": url,
                    "title": f"{title} (Part {i+1})" if len(chunks) > 1 else title,
                    "content": chunk,
                    "source": source_name,
                    "type": "webpage",
                    "timestamp": datetime.now().isoformat()
                }
            )
            points.append(point)
        
        if points:
            self.client.upsert(COLLECTION_NAME, points)
            logger.info(f"  [HTML] {url} indexed ({len(chunks)} chunks)")

    async def process_pdf(self, url: str):
        logger.info(f"  [PDF] Downloading and extracting: {url}")
        text = download_and_extract_pdf(url, session=self.requests_session)
        if not text or len(text) < 100: return

        filename = urlparse(url).path.split("/")[-1]
        title = filename.replace("_", " ").replace("-", " ").replace(".pdf", "").title()
        chunks = chunk_text(text, max_chars=1200, overlap=150)
        source_name = self.domain.replace('www.', '').split('.')[0]

        points = []
        for i, chunk in enumerate(chunks):
            dense_vec = self.model.encode(chunk).tolist()
            sparse_vec = self.sparse_encode(chunk)
            
            point = models.PointStruct(
                id=int(hashlib.md5(f"pdf_{url}_{i}".encode()).hexdigest()[:12], 16),
                vector={
                    "dense": dense_vec,
                    "sparse": sparse_vec
                },
                payload={
                    "url": url,
                    "title": f"{title} (Part {i+1})",
                    "content": chunk,
                    "source": source_name,
                    "type": "pdf",
                    "filename": filename,
                    "timestamp": datetime.now().isoformat()
                }
            )
            points.append(point)
        
        if points:
            self.client.upsert(COLLECTION_NAME, points)
            logger.info(f"  [PDF] {filename} indexed ({len(chunks)} chunks)")

    async def run(self):
        await self.fetch_robots_txt()
        urls = await self.discover_urls_from_sitemaps()
        
        # Merge sitemap URLs and manual discovery
        to_process = [u for u in urls if self.is_allowed(u)]
        logger.info(f"🚀 Starting scrape of {len(to_process)} URLs (max_pages={self.max_pages})")
        
        count = 0
        for url in to_process:
            if count >= self.max_pages: break
            await self.process_url(url)
            count += 1
            if count % 20 == 0: logger.info(f"--- Processed {count}/{min(len(to_process), self.max_pages)} ---")

        # Also process any PDFs found during HTML crawl
        pdf_list = list(self.found_pdf_urls)
        logger.info(f"📎 Processing {len(pdf_list)} manually discovered PDFs...")
        for pdf_url in pdf_list:
            await self.process_url(pdf_url)

        await self.httpx_client.aclose()
        logger.info("✅ Scrape complete")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Universal Website Scrapper')
    parser.add_argument('url', help='Base URL to scrape')
    parser.add_argument('--limit', type=int, default=500, help='Max pages to scrape')
    args = parser.parse_args()
    
    scraper = UniversalScraper(args.url, max_pages=args.limit)
    asyncio.run(scraper.run())

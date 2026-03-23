#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import sys
import re
from typing import Set, List, Dict, Optional
from urllib.parse import urljoin, urlparse

import httpx
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# Add parent directory to sys.path for imports from backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.pdf_extractor import download_and_extract_pdf, chunk_text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "https://www.hs-aalen.de"
COLLECTION_NAME = "hs_aalen_pdfs"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

class PDFComprehensiveScraper:
    def __init__(self, max_pages=100):
        self.visited_urls: Set[str] = set()
        self.found_pdf_urls: Set[str] = set()
        self.max_pages = max_pages
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.model = SentenceTransformer(MODEL_NAME)
        self.point_id_counter = 5000 # Use a high offset to avoid collisions with other collections
        
    async def discover_pdfs_from_sitemap(self):
        """Fetch sitemap and find PDF links"""
        logger.info("🗺️  Checking sitemap for PDFs...")
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(f"{BASE_URL}/sitemap.xml")
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'xml')
                    locs = [loc.text for loc in soup.find_all('loc')]
                    
                    for loc in locs:
                        if loc.lower().endswith('.pdf'):
                            self.found_pdf_urls.add(loc)
                        elif "sitemap" in loc:
                            # Recursive sitemap check
                            sub_resp = await client.get(loc)
                            if sub_resp.status_code == 200:
                                sub_soup = BeautifulSoup(sub_resp.text, 'xml')
                                sub_locs = [sl.text for sl in sub_soup.find_all('loc')]
                                for sl in sub_locs:
                                    if sl.lower().endswith('.pdf'):
                                        self.found_pdf_urls.add(sl)
            
            logger.info(f"✓ Found {len(self.found_pdf_urls)} PDFs in sitemap")
        except Exception as e:
            logger.warning(f"⚠️  Sitemap discovery error: {e}")

    async def crawl_for_pdfs(self):
        """Crawl the website to discover PDF links using Playwright"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            queue = [BASE_URL]
            
            logger.info(f"🕵️  Crawling up to {self.max_pages} pages for PDF links...")
            
            count = 0
            while queue and count < self.max_pages:
                url = queue.pop(0)
                if url in self.visited_urls:
                    continue
                
                self.visited_urls.add(url)
                count += 1
                
                try:
                    page = await browser.new_page()
                    await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                    
                    # Extract all <a> tags
                    links = await page.query_selector_all("a")
                    for link in links:
                        href = await link.get_attribute("href")
                        if not href:
                            continue
                            
                        full_url = urljoin(url, href)
                        parsed = urlparse(full_url)
                        
                        # Normalize
                        full_url = full_url.split("#")[0].split("?")[0].rstrip("/")
                        
                        if full_url.lower().endswith(".pdf"):
                            if full_url not in self.found_pdf_urls:
                                logger.info(f"  [PDF Found] {full_url}")
                                self.found_pdf_urls.add(full_url)
                        elif parsed.netloc == "www.hs-aalen.de" and full_url not in self.visited_urls:
                            if len(queue) < 500: # Limit queue size
                                queue.append(full_url)
                    
                    await page.close()
                    if count % 10 == 0:
                        logger.info(f"  Processed {count}/{self.max_pages} pages...")
                        
                except Exception as e:
                    logger.debug(f"  Error crawling {url}: {e}")
            
            await browser.close()
            logger.info(f"✓ Crawling complete. Total unique PDFs found: {len(self.found_pdf_urls)}")

    def index_pdfs(self):
        """Download, extract, and index all found PDFs"""
        logger.info(f"🚀  Indexing {len(self.found_pdf_urls)} PDFs...")
        
        # Ensure collection exists
        try:
            self.client.get_collection(COLLECTION_NAME)
        except:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            logger.info(f"Created collection {COLLECTION_NAME}")

        success_count = 0
        with requests.Session() as session:
            for pdf_url in list(self.found_pdf_urls):
                try:
                    # Check if already indexed
                    query_res = self.client.scroll(
                        collection_name=COLLECTION_NAME,
                        scroll_filter={"must": [{"key": "url", "match": {"value": pdf_url}}]},
                        limit=1
                    )
                    if query_res[0]:
                        logger.info(f"  [Skip] Already indexed: {pdf_url}")
                        continue

                    text = download_and_extract_pdf(pdf_url, session=session)
                    if not text or len(text) < 50:
                        continue

                    filename = urlparse(pdf_url).path.split("/")[-1]
                    title = filename.replace("_", " ").replace("-", " ").replace(".pdf", "").title()
                    
                    chunks = chunk_text(text)
                    points = []
                    
                    for i, chunk in enumerate(chunks):
                        vector = self.model.encode(chunk).tolist()
                        point = PointStruct(
                            id=self.point_id_counter,
                            vector=vector,
                            payload={
                                "url": pdf_url,
                                "title": f"{title} (Teil {i+1})",
                                "text": chunk,
                                "type": "pdf",
                                "source": "hs_aalen_pdfs",
                                "filename": filename
                            }
                        )
                        points.append(point)
                        self.point_id_counter += 1
                    
                    self.client.upsert(COLLECTION_NAME, points)
                    success_count += 1
                    logger.info(f"  [OK] Indexed {filename} ({len(chunks)} chunks)")
                    
                except Exception as e:
                    logger.error(f"  [Error] Failed to process {pdf_url}: {e}")
        
        logger.info(f"✓ Successfully indexed {success_count} new PDFs")

    async def run(self):
        await self.discover_pdfs_from_sitemap()
        await self.crawl_for_pdfs()
        self.index_pdfs()

if __name__ == "__main__":
    # Limit max pages for initial run to keep it manageable
    max_pages = 50
    if len(sys.argv) > 1:
        max_pages = int(sys.argv[1])
        
    scraper = PDFComprehensiveScraper(max_pages=max_pages)
    asyncio.run(scraper.run())

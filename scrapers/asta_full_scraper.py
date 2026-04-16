from typing import Dict, List, Optional

#!/usr/bin/env python3
"""
ASTA Website Scraper - Kompletter Setup
Scraped ASTA-Website mit Playwright in separate Collection
Konfigurierbar für verschiedene ASTA-Domains
"""

import asyncio
import json
import logging
import re
import sys
from typing import Set
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============ KONFIGURATION ============
# Passe diese URL an wenn du die echte ASTA-Domain kennst
# Bekannte Optionen:
#   - https://vs-hs-aalen.de (Verfasste Studierendenschaft)
#   - https://www.vs-hs-aalen.de
#   - https://asta.hs-aalen.de
#   - https://www.asta-aalen.de
ASTA_BASE_URL = "https://www.vs-hs-aalen.de"  # Verfasste Studierendenschaft

# Optional: Proxy wenn nötig
PROXY = None  # z.B. "http://proxy.example.com:8080"

# Scraping-Limits
MAX_PAGES = 500
PAGE_TIMEOUT = 15000  # millisekunden


class ASTAFullScraper:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.domain = urlparse(base_url).netloc
        self.visited_urls: Set[str] = set()
        self.pages_data: List[Dict] = []
        self.browser = None
        
    async def init_browser(self):
        """Initialisiere Playwright Browser"""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(
            headless=True,
            proxy={"server": PROXY} if PROXY else None
        )
        logger.info("✓ Browser started")
        
    async def close_browser(self):
        """Schließe Browser"""
        if self.browser:
            await self.browser.close()
            logger.info("✓ Browser closed")
    
    async def fetch_page(self, url: str) -> Optional[str]:
        """Fetch page mit Playwright"""
        if url in self.visited_urls:
            return None
        
        if not url.startswith(self.base_url):
            return None
        
        try:
            self.visited_urls.add(url)
            
            page = await self.browser.new_page()
            try:
                await page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=PAGE_TIMEOUT
                )
                content = await page.content()
                return content
            finally:
                await page.close()
                
        except Exception as e:
            logger.debug(f"  Error fetching {url}: {e}")
            return None
    
    async def scrape_with_crawler(self, max_pages: int) -> int:
        """Crawl ASTA-Website wenn keine Sitemap"""
        logger.info(f"🕷️  Starting crawler (max {max_pages} pages)...")
        
        to_visit = [self.base_url]
        count = 0
        
        while to_visit and len(self.visited_urls) < max_pages:
            url = to_visit.pop(0)
            
            if url in self.visited_urls:
                continue
            
            html = await self.fetch_page(url)
            if not html:
                continue
            
            if self._extract_content(html, url):
                count += 1
                if count % 50 == 0:
                    logger.info(f"  [{count}] Pages scraped...")
            
            # Extract links
            soup = BeautifulSoup(html, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                if href.startswith('#') or href.startswith('mailto:'):
                    continue
                
                full_url = urljoin(url, href)
                
                # Only internal links
                if not full_url.startswith(self.base_url):
                    continue
                
                # Remove fragment
                full_url = full_url.split('#')[0]
                
                if full_url not in self.visited_urls:
                    to_visit.append(full_url)
        
        return count
    
    def _extract_content(self, html: str, url: str) -> bool:
        """Extrahiere Seiten-Inhalt"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script/style
            for tag in soup(['script', 'style']):
                tag.decompose()
            
            # Get title
            title_tag = soup.find('h1') or soup.find('title')
            title = title_tag.get_text(strip=True) if title_tag else ""
            
            # Extract text
            content = soup.get_text(separator=' ', strip=True)
            content = re.sub(r'\s+', ' ', content)
            
            # Filter out only completely empty pages
            if not content.strip():
                return False
            
            # Limit to 3000 chars
            content = content[:3000]
            
            self.pages_data.append({
                'url': url,
                'title': title.strip() or "ASTA Page",
                'content': content,
                'source': 'asta_website'
            })
            
            return True
            
        except Exception as e:
            logger.debug(f"  Extraction error: {e}")
            return False
    
    async def run(self):
        """Hauptfunktion"""
        await self.init_browser()
        
        try:
            count = await self.scrape_with_crawler(max_pages=MAX_PAGES)
            logger.info(f"✓ Scraping complete: {count} pages extracted")
            
        finally:
            await self.close_browser()
    
    def save(self, filename: str = "asta_full_data.json"):
        """Speichere Daten"""
        output = {
            'base_url': self.base_url,
            'total_pages': len(self.pages_data),
            'pages': self.pages_data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Saved {len(self.pages_data)} pages to {filename}")


async def main():
    if ASTA_BASE_URL == "https://www.asta-aalen.de":
        logger.warning("⚠️  WARNUNG: ASTA_BASE_URL ist noch nicht konfiguriert!")
        logger.warning("   Bitte editiere diese Datei und setze die richtige ASTA-Domain")
        logger.warning("   Bekannte URLs:")
        logger.warning("   - https://asta.hs-aalen.de")
        logger.warning("   - https://www.asta-aalen.de")
        logger.warning("   - Frag deinen IT-Admin für interne URLs")
        sys.exit(1)
    
    logger.info(f"🎯 Scraping ASTA: {ASTA_BASE_URL}")
    scraper = ASTAScraper(ASTA_BASE_URL)
    await scraper.run()
    scraper.save()


class ASTAScraper(ASTAFullScraper):
    """Alias für Kompatibilität"""
    pass


if __name__ == "__main__":
    asyncio.run(main())

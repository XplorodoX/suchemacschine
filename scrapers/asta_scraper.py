from typing import Dict, List, Optional

#!/usr/bin/env python3
"""
ASTA (Allgemeiner Studierenden-Ausschuss) Web Scraper für HS Aalen
Respektiert robots.txt und scraped nur legal erlaubte Inhalte
Nutzt Sitemap falls vorhanden, fallback auf Crawler
"""

import asyncio
import json
import logging
import re
from typing import Set
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mögliche ASTA-Adressen (probieren wir alle)
POSSIBLE_URLS = [
    "https://www.asta-aalen.de",
    "https://asta.hs-aalen.de",
    "https://www.asta.hs-aalen.de",
    "https://asta-aalen.de",
]

class ASTAScraper:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.domain = urlparse(base_url).netloc
        self.visited_urls: Set[str] = set()
        self.allowed_paths: Set[str] = set()
        self.disallowed_paths: Set[str] = set()
        self.pages_data: List[Dict] = []
        self.session = None
        
    async def init_session(self):
        """Initialisiere HTTP Session"""
        self.session = httpx.AsyncClient(timeout=10.0, follow_redirects=True)
        
    async def close_session(self):
        """Schließe HTTP Session"""
        if self.session:
            await self.session.aclose()
    
    async def check_robots_txt(self) -> bool:
        """
        Lese robots.txt und speichere erlaubte/verbotene Pfade
        Rückgabe: True wenn Domain scrapbar ist
        """
        try:
            robots_url = f"{self.base_url}/robots.txt"
            logger.info(f"📋 Checking robots.txt: {robots_url}")
            
            response = await self.session.get(robots_url)
            if response.status_code != 200:
                logger.info(f"⚠️  robots.txt not found (status {response.status_code}), proceeding with caution")
                return True
            
            # Parse robots.txt
            lines = response.text.split('\n')
            in_global = False
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                if line.lower().startswith('user-agent:'):
                    agent = line.split(':', 1)[1].strip()
                    in_global = agent == '*'
                    continue
                
                if not in_global:
                    continue
                
                if line.lower().startswith('disallow:'):
                    path = line.split(':', 1)[1].strip()
                    if path:
                        self.disallowed_paths.add(path)
                        logger.debug(f"  Disallowed: {path}")
                
                if line.lower().startswith('allow:'):
                    path = line.split(':', 1)[1].strip()
                    if path:
                        self.allowed_paths.add(path)
                        logger.debug(f"  Allowed: {path}")
            
            logger.info(f"✓ robots.txt parsed: {len(self.disallowed_paths)} disallowed, {len(self.allowed_paths)} allowed")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  Could not check robots.txt: {e}")
            return True
    
    def is_path_allowed(self, url: str) -> bool:
        """Prüfe ob Pfad gemäß robots.txt erlaubt ist"""
        path = urlparse(url).path or '/'
        
        # Check disallowed
        for disallowed in self.disallowed_paths:
            if disallowed == '/*':  # Alles verboten
                return False
            if path.startswith(disallowed):
                return False
        
        # Check allowed (whitelist)
        if self.allowed_paths:
            for allowed in self.allowed_paths:
                if allowed == '/*' or path.startswith(allowed):
                    return True
            return False
        
        return True
    
    async def fetch_page(self, url: str) -> Optional[str]:
        """Fetch einzelne Seite"""
        if not url.startswith(self.base_url):
            return None
        
        if url in self.visited_urls:
            return None
        
        if not self.is_path_allowed(url):
            logger.debug(f"  ⊘ Skipped (robots.txt): {url}")
            return None
        
        try:
            self.visited_urls.add(url)
            response = await self.session.get(url)
            
            if response.status_code == 200:
                return response.text
            else:
                logger.debug(f"  ✗ HTTP {response.status_code}: {url}")
                return None
                
        except Exception as e:
            logger.debug(f"  ✗ Error fetching {url}: {e}")
            return None
    
    async def scrape_from_sitemap(self) -> bool:
        """Scrape via Sitemap wenn vorhanden"""
        try:
            sitemap_url = f"{self.base_url}/sitemap.xml"
            logger.info(f"🗺️  Trying sitemap: {sitemap_url}")
            
            response = await self.session.get(sitemap_url)
            if response.status_code != 200:
                logger.info("⚠️  No sitemap found")
                return False
            
            # Parse sitemap
            soup = BeautifulSoup(response.text, 'xml')
            urls = [loc.text for loc in soup.find_all('loc')]
            
            logger.info(f"✓ Found {len(urls)} URLs in sitemap")
            
            for i, url in enumerate(urls, 1):
                if not self.is_path_allowed(url):
                    logger.debug(f"  Skipped (robots.txt): {url}")
                    continue
                
                html = await self.fetch_page(url)
                if html:
                    self._extract_content(html, url)
                    logger.info(f"  [{i}/{len(urls)}] ✓ {url}")
            
            return len(self.pages_data) > 0
            
        except Exception as e:
            logger.warning(f"⚠️  Sitemap scraping failed: {e}")
            return False
    
    async def scrape_with_crawler(self, max_pages: int = 100) -> bool:
        """Fallback: Crawl website wenn keine Sitemap"""
        logger.info(f"🕷️  Starting crawler (max {max_pages} pages)...")
        
        to_visit = [self.base_url]
        
        while to_visit and len(self.visited_urls) < max_pages:
            url = to_visit.pop(0)
            
            if url in self.visited_urls:
                continue
            
            if not self.is_path_allowed(url):
                logger.debug(f"  Skipped (robots.txt): {url}")
                continue
            
            html = await self.fetch_page(url)
            if not html:
                continue
            
            self._extract_content(html, url)
            logger.info(f"  [{len(self.visited_urls)}/{max_pages}] ✓ {url}")
            
            # Extrahiere neue Links
            soup = BeautifulSoup(html, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Skip fragments und mailto
                if href.startswith('#') or href.startswith('mailto:'):
                    continue
                
                full_url = urljoin(url, href)
                
                # Nur internal links
                if not full_url.startswith(self.base_url):
                    continue
                
                # Normalisiere URL (remove fragment)
                full_url = full_url.split('#')[0]
                
                if full_url not in self.visited_urls:
                    to_visit.append(full_url)
        
        return len(self.pages_data) > 0
    
    def _extract_content(self, html: str, url: str):
        """Extrahiere Inhalt einer Seite"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Entferne script/style tags
            for tag in soup(['script', 'style']):
                tag.decompose()
            
            title = soup.find('h1')
            title_text = title.get_text(strip=True) if title else "No title"
            
            content = soup.get_text(separator=' ', strip=True)
            content = re.sub(r'\s+', ' ', content)[:2000]  # Limit to 2000 chars
            
            if content.strip():
                self.pages_data.append({
                    'url': url,
                    'title': title_text,
                    'content': content,
                    'source': 'asta'
                })
                logger.debug(f"    Content extracted: {len(content)} chars")
            
        except Exception as e:
            logger.debug(f"    Content extraction error: {e}")
    
    async def run(self, use_sitemap: bool = True) -> bool:
        """Hauptfunktion"""
        await self.init_session()
        
        try:
            # Check robots.txt
            if not await self.check_robots_txt():
                logger.error("Domain ist nicht scrapbar (robots.txt)")
                return False
            
            # Try sitemap first
            if use_sitemap:
                if await self.scrape_from_sitemap():
                    logger.info("✓ Scraping successful via sitemap!")
                    return True
            
            # Fallback zu Crawler
            if await self.scrape_with_crawler(max_pages=100):
                logger.info("✓ Scraping successful via crawler!")
                return True
            
            logger.warning("⚠️  No content extracted")
            return False
            
        finally:
            await self.close_session()
    
    def save(self, filename: str = "asta_data.json"):
        """Speichere gescrapte Daten"""
        output = {
            'base_url': self.base_url,
            'total_pages': len(self.pages_data),
            'pages': self.pages_data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Saved {len(self.pages_data)} pages to {filename}")


async def find_and_scrape_asta():
    """Versuche ASTA-Website zu finden und scrapen"""
    for base_url in POSSIBLE_URLS:
        logger.info(f"🔍 Trying: {base_url}")
        
        scraper = ASTAScraper(base_url)
        
        try:
            if await scraper.run():
                scraper.save("asta_data.json")
                logger.info(f"✅ Successfully scraped ASTA from {base_url}")
                logger.info(f"   Total pages: {len(scraper.pages_data)}")
                return True
        except Exception as e:
            logger.warning(f"  ✗ Failed: {e}")
            continue
    
    logger.error("❌ Could not reach any ASTA URL. Check URLs or network.")
    return False


if __name__ == "__main__":
    success = asyncio.run(find_and_scrape_asta())
    exit(0 if success else 1)

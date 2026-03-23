from typing import Dict, List, Optional

#!/usr/bin/env python3
"""
HS Aalen Website Scraper - Extended
Scraped Hauptseite UND ASTA-Inhalte
Respektiert robots.txt und nutzt Sitemap
"""

import asyncio
import json
import logging
import re
from typing import Set
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://www.hs-aalen.de"
DOMAIN = "hs-aalen.de"

class HSAalenScraper:
    def __init__(self):
        self.visited_urls: Set[str] = set()
        self.pages_data: List[Dict] = []
        self.session = None
        self.disallowed_patterns = []
        
    async def init_session(self):
        """Initialisiere HTTP Session"""
        self.session = httpx.AsyncClient(
            timeout=15.0, 
            follow_redirects=True,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; HS-Aalen-Bot/1.0)'}
        )
    
    async def close_session(self):
        """Schließe Session"""
        if self.session:
            await self.session.aclose()
    
    async def check_robots_txt(self) -> bool:
        """Prüfe robots.txt"""
        try:
            logger.info("📋 Checking robots.txt...")
            response = await self.session.get(f"{BASE_URL}/robots.txt")
            
            if response.status_code == 200:
                lines = response.text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.lower().startswith('disallow:') and line.startswith('Disallow:'):
                        path = line.split(':', 1)[1].strip()
                        if path:
                            self.disallowed_patterns.append(path)
                
                logger.info(f"✓ Found {len(self.disallowed_patterns)} disallowed patterns")
                return True
        except Exception as e:
            logger.warning(f"⚠️  Could not read robots.txt: {e}")
            return True
    
    def is_allowed(self, url: str) -> bool:
        """Prüfe ob URL erlaubt ist"""
        path = urlparse(url).path
        
        for pattern in self.disallowed_patterns:
            if pattern == '/*':
                return False
            if path.startswith(pattern):
                logger.debug(f"    Blocked by robots.txt: {pattern}")
                return False
        
        return True
    
    async def fetch_page(self, url: str) -> Optional[str]:
        """Hole einzelne Seite"""
        if url in self.visited_urls:
            return None
        
        if not url.startswith(BASE_URL):
            return None
        
        if not self.is_allowed(url):
            return None
        
        try:
            self.visited_urls.add(url)
            response = await self.session.get(url)
            
            if response.status_code == 200:
                return response.text
            return None
        except Exception as e:
            logger.debug(f"  Error fetching {url}: {e}")
            return None
    
    async def scrape_sitemap(self) -> int:
        """Scrape via Sitemap (hierarchical)"""
        try:
            logger.info("🗺️  Fetching sitemap.xml...")
            response = await self.session.get(f"{BASE_URL}/sitemap.xml")
            
            if response.status_code != 200:
                logger.warning("  No sitemap found")
                return 0
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Parse sitemapindex to find subsitemaps
            subsitemaps = [loc.text for loc in soup.find_all('loc')]
            
            if not subsitemaps:
                logger.warning("  No subsitemaps found")
                return 0
            
            logger.info(f"✓ Found {len(subsitemaps)} subsitemaps")
            
            all_urls = []
            
            # Fetch each subsitemap
            for i, sitemap_url in enumerate(subsitemaps, 1):
                logger.info(f"  [{i}] Fetching {sitemap_url[:60]}...")
                
                try:
                    response = await self.session.get(sitemap_url, timeout=10.0)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        urls = [loc.text for loc in soup.find_all('loc')]
                        all_urls.extend(urls)
                        logger.info(f"     Found {len(urls)} content URLs")
                except Exception as e:
                    logger.warning(f"     Error: {e}")
            
            logger.info(f"✓ Total {len(all_urls)} URLs to scrape")
            
            count = 0
            for i, url in enumerate(all_urls[:500], 1):  # Limit to 500
                if not self.is_allowed(url):
                    continue
                
                html = await self.fetch_page(url)
                if html:
                    if self._extract_content(html, url):
                        count += 1
                        if count % 10 == 0:
                            logger.info(f"  [{count}] ✓ Scraped {count} pages...")
            
            return count
            
        except Exception as e:
            logger.warning(f"⚠️  Sitemap error: {e}")
            return 0
    
    def _extract_content(self, html: str, url: str) -> bool:
        """Extrahiere Seiten-Inhalt"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Entferne script/style
            for tag in soup(['script', 'style']):
                tag.decompose()
            
            # Finde Titel
            title_tag = soup.find('h1') or soup.find('title')
            title = title_tag.get_text(strip=True) if title_tag else "No title"
            
            # Extrahiere Text
            content = soup.get_text(separator=' ', strip=True)
            content = re.sub(r'\s+', ' ', content)
            
            if len(content) < 100:
                return False
            
            # Limit auf 3000 chars für bessere Indexierung
            content = content[:3000]
            
            self.pages_data.append({
                'url': url,
                'title': title,
                'content': content,
                'source': 'hs_aalen_website'
            })
            
            return True
            
        except Exception as e:
            logger.debug(f"    Extraction error: {e}")
            return False
    
    async def run(self):
        """Hauptfunktion"""
        await self.init_session()
        
        try:
            await self.check_robots_txt()
            
            # Scrape via Sitemap
            count = await self.scrape_sitemap()
            
            logger.info(f"✓ Scraping complete: {count} pages extracted")
            
        finally:
            await self.close_session()
    
    def save(self, filename: str = "hs_aalen_extended_data.json"):
        """Speichere Daten"""
        output = {
            'base_url': BASE_URL,
            'total_pages': len(self.pages_data),
            'pages': self.pages_data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Saved {len(self.pages_data)} pages to {filename}")


async def main():
    scraper = HSAalenScraper()
    await scraper.run()
    scraper.save()

if __name__ == "__main__":
    asyncio.run(main())

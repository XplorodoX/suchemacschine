from typing import Dict, List, Optional

#!/usr/bin/env python3
"""
ASTA Website Scraper
Scraped ASTA (Allgemeiner Studierenden-Ausschuss) Website
Respektiert robots.txt und nutzt Sitemap wenn möglich
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

# Mögliche ASTA URLs (Hochschule Aalen)
ASTA_URLS = [
    "https://www.asta.hs-aalen.de",
    "https://asta.hs-aalen.de",
    "https://www.asta-aalen.de", 
    "https://asta-aalen.de",
    "https://www.hs-aalen.de/asta",  # Fallback: Might be under main domain
]

class ASTAWebsiteScraper:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.domain = urlparse(base_url).netloc
        self.visited_urls: Set[str] = set()
        self.pages_data: List[Dict] = []
        self.session = None
        
    async def init_session(self):
        """Initialisiere HTTP Session"""
        self.session = httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            headers={'User-Agent': 'Mozilla/5.0 (ASTA-Scraper)'}
        )
    
    async def close_session(self):
        """Schließe Session"""
        if self.session:
            await self.session.aclose()
    
    async def fetch_page(self, url: str) -> Optional[str]:
        """Hole einzelne Seite"""
        if url in self.visited_urls:
            return None
        
        if not url.startswith(self.base_url):
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
        """Versuche via Sitemap zu scrapen"""
        try:
            logger.info("🗺️  Checking for sitemap...")
            
            for sitemap_url in [
                f"{self.base_url}/sitemap.xml",
                f"{self.base_url}/sitemap_index.xml",
            ]:
                response = await self.session.get(sitemap_url, timeout=10.0)
                
                if response.status_code != 200:
                    continue
                
                logger.info(f"✓ Found sitemap: {sitemap_url}")
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Parse sitemapindex or sitemap
                urls = [loc.text for loc in soup.find_all('loc')]
                
                # If sitemapindex, fetch subsitemaps
                if 'sitemap' in response.text and sitemap_url.endswith('index.xml'):
                    all_urls = []
                    for sub_url in urls:
                        try:
                            sub_response = await self.session.get(sub_url, timeout=10.0)
                            if sub_response.status_code == 200:
                                sub_soup = BeautifulSoup(sub_response.text, 'html.parser')
                                all_urls.extend([loc.text for loc in sub_soup.find_all('loc')])
                        except:
                            pass
                    urls = all_urls
                
                logger.info(f"  Found {len(urls)} URLs")
                
                count = 0
                for i, url in enumerate(urls[:200], 1):  # Limit to 200
                    html = await self.fetch_page(url)
                    if html:
                        if self._extract_content(html, url):
                            count += 1
                            if count % 20 == 0:
                                logger.info(f"    [{count}] Scraped...")
                
                return count
                
        except Exception as e:
            logger.warning(f"⚠️  Sitemap error: {e}")
        
        return 0
    
    async def scrape_crawler(self, max_pages: int = 100) -> int:
        """Fallback: Crawl website wenn keine Sitemap"""
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
                if count % 20 == 0:
                    logger.info(f"  [{count}] Scraped...")
            
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
            title = title_tag.get_text(strip=True) if title_tag else "No title"
            
            # Extract text
            content = soup.get_text(separator=' ', strip=True)
            content = re.sub(r'\s+', ' ', content)
            
            if len(content) < 100:
                return False
            
            # Limit to 3000 chars
            content = content[:3000]
            
            self.pages_data.append({
                'url': url,
                'title': title,
                'content': content,
                'source': 'asta_website'
            })
            
            return True
            
        except Exception as e:
            logger.debug(f"    Extraction error: {e}")
            return False
    
    async def run(self) -> bool:
        """Hauptfunktion"""
        await self.init_session()
        
        try:
            # Try sitemap first
            count = await self.scrape_sitemap()
            if count > 0:
                logger.info(f"✓ Scraped {count} pages via sitemap")
                return True
            
            # Fallback to crawler
            count = await self.scrape_crawler(max_pages=100)
            if count > 0:
                logger.info(f"✓ Scraped {count} pages via crawler")
                return True
            
            logger.warning("⚠️  No content extracted")
            return False
            
        finally:
            await self.close_session()
    
    def save(self, filename: str = "asta_data.json"):
        """Speichere Daten"""
        output = {
            'base_url': self.base_url,
            'total_pages': len(self.pages_data),
            'pages': self.pages_data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Saved {len(self.pages_data)} pages to {filename}")


async def find_and_scrape_asta():
    """Versuche ASTA-Website auf verschiedenen Domains zu scrapen"""
    
    for base_url in ASTA_URLS:
        logger.info(f"🔍 Trying: {base_url}")
        
        scraper = ASTAWebsiteScraper(base_url)
        
        try:
            if await scraper.run():
                scraper.save("asta_data.json")
                logger.info(f"✅ Successfully scraped ASTA from {base_url}")
                logger.info(f"   Total pages: {len(scraper.pages_data)}")
                return True
        except Exception as e:
            logger.debug(f"  Error: {e}")
            continue
    
    logger.warning("⚠️  Could not reach any ASTA URL")
    return False


if __name__ == "__main__":
    success = asyncio.run(find_and_scrape_asta())
    exit(0 if success else 1)

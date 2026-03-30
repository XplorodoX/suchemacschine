from typing import Dict, List, Optional

#!/usr/bin/env python3
"""
HS Aalen Extended Website Scraper mit Playwright
Bessere Datenextraktion mit JavaScript-Rendering
"""

import asyncio
import json
import logging
import re
from typing import Set

from bs4 import BeautifulSoup
from playwright.async_api import Browser, async_playwright

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://www.hs-aalen.de"

class HSAalenPlaywrightScraper:
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.visited_urls: Set[str] = set()
        self.pages_data: List[Dict] = []
        
    async def init_browser(self):
        """Initialisiere Browser"""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=True)
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
        
        if not url.startswith(BASE_URL):
            return None
        
        try:
            self.visited_urls.add(url)
            
            page = await self.browser.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=15000)
                content = await page.content()
                return content
            finally:
                await page.close()
                
        except Exception as e:
            logger.debug(f"  Error fetching {url}: {e}")
            return None
    
    async def scrape_from_sitemap(self) -> int:
        """Fetch sitemap and scrape all URLs"""
        try:
            logger.info("🗺️  Fetching sitemap...")
            import httpx
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(f"{BASE_URL}/sitemap.xml")
                
                if response.status_code != 200:
                    logger.warning("  No sitemap found")
                    return 0
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Get subsitemaps
                sitemap_urls = [loc.text for loc in soup.find_all('loc')]
                
                if not sitemap_urls:
                    logger.warning("  No subsitemaps found")
                    return 0
                
                logger.info(f"✓ Found {len(sitemap_urls)} subsitemaps")
                
                all_urls = []
                
                # Fetch each subsitemap
                for sitemap_url in sitemap_urls:
                    try:
                        sub_response = await client.get(sitemap_url, timeout=10)
                        if sub_response.status_code == 200:
                            sub_soup = BeautifulSoup(sub_response.text, 'html.parser')
                            urls = [loc.text for loc in sub_soup.find_all('loc')]
                            all_urls.extend(urls)
                            logger.info(f"  Found {len(urls)} URLs in {sitemap_url.split('=')[-1][:30]}...")
                    except Exception as e:
                        logger.debug(f"  Error fetching subsitemap: {e}")
                
                logger.info(f"✓ Total {len(all_urls)} URLs to scrape")
                
                # Scrape URLs with Playwright
                count = 0
                for i, url in enumerate(all_urls[:150], 1):  # Limit to 150
                    html = await self.fetch_page(url)
                    if html:
                        if self._extract_content(html, url):
                            count += 1
                            if count % 20 == 0:
                                logger.info(f"  [{count}] Scraped...")
                
                return count
                
        except Exception as e:
            logger.warning(f"⚠️  Sitemap error: {e}")
            return 0
    
    def _extract_content(self, html: str, url: str) -> bool:
        """Extract content from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script/style
            for tag in soup(['script', 'style']):
                tag.decompose()
            
            # Get title
            title_tag = soup.find('h1') or soup.find('title')
            title = title_tag.get_text(strip=True) if title_tag else ""
            
            # If no h1, try meta description or page title
            if not title:
                meta_title = soup.find('meta', attrs={'name': 'description'})
                if meta_title:
                    title = meta_title.get('content', '')[:100]
                else:
                    title = "Page"
            
            # Extract text
            content = soup.get_text(separator=' ', strip=True)
            content = re.sub(r'\s+', ' ', content)
            
            # Filter out very short pages
            if len(content) < 150:
                return False
            
            # Limit to 3000 chars
            content = content[:3000]
            
            self.pages_data.append({
                'url': url,
                'title': title.strip(),
                'content': content,
                'source': 'hs_aalen_website'
            })
            
            return True
            
        except Exception as e:
            logger.debug(f"    Extraction error: {e}")
            return False
    
    async def run(self):
        """Main function"""
        await self.init_browser()
        
        try:
            count = await self.scrape_from_sitemap()
            logger.info(f"✓ Scraping complete: {count} pages extracted")
            
        finally:
            await self.close_browser()
    
    def save(self, filename: str = "hs_aalen_extended_data.json"):
        """Save data"""
        output = {
            'base_url': BASE_URL,
            'total_pages': len(self.pages_data),
            'pages': self.pages_data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Saved {len(self.pages_data)} pages to {filename}")


async def main():
    scraper = HSAalenPlaywrightScraper()
    await scraper.run()
    scraper.save()

if __name__ == "__main__":
    asyncio.run(main())

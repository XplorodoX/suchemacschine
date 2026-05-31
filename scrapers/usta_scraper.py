#!/usr/bin/env python3
"""
USTA Aalen Scraper
Scraped https://www.usta-aalen.de/ mit Playwright.
Respektiert robots.txt: /calendar/action~*, /*/action~*, /intranet/ sind ausgeschlossen.
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Set
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://www.usta-aalen.de"
MAX_PAGES = 300
PAGE_TIMEOUT = 15000

DISALLOWED_PATTERNS = [
    re.compile(r"/calendar/action~"),
    re.compile(r"/action~"),
    re.compile(r"controller=ai1ec_exporter_controller"),
    re.compile(r"/intranet/"),
]

MEDIA_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.pdf',
                    '.mp4', '.mp3', '.zip', '.docx', '.xlsx', '.pptx'}


def is_allowed(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path
    ext = path[path.rfind('.'):].lower() if '.' in path.rsplit('/', 1)[-1] else ''
    if ext in MEDIA_EXTENSIONS:
        return False
    full = path + ("?" + parsed.query if parsed.query else "")
    return not any(p.search(full) for p in DISALLOWED_PATTERNS)


def extract_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    title = ""
    if soup.title:
        title = soup.title.get_text(strip=True)
    for h in soup.find_all(["h1", "h2"]):
        t = h.get_text(strip=True)
        if t:
            title = t
            break

    main = soup.find("main") or soup.find("article") or soup.find(id=re.compile(r"content|main", re.I)) or soup.body
    content = main.get_text(separator=" ", strip=True) if main else soup.get_text(separator=" ", strip=True)
    content = re.sub(r"\s{2,}", " ", content).strip()
    return title, content


class USTAScraper:
    def __init__(self):
        self.domain = urlparse(BASE_URL).netloc
        self.visited: Set[str] = set()
        self.pages: List[Dict] = []
        self.browser = None

    async def init(self):
        pw = await async_playwright().start()
        self.browser = await pw.chromium.launch(headless=True)
        logger.info("✓ Browser started")

    async def close(self):
        if self.browser:
            await self.browser.close()

    def _same_domain(self, url: str) -> bool:
        return urlparse(url).netloc == self.domain

    async def scrape_page(self, url: str) -> tuple[str, str, List[str]]:
        ctx = await self.browser.new_context()
        page = await ctx.new_page()
        links: List[str] = []
        try:
            await page.goto(url, timeout=PAGE_TIMEOUT, wait_until="domcontentloaded")
            html = await page.content()
            title, content = extract_text(html)
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"])
                href = href.split("#")[0].rstrip("/")
                if href and self._same_domain(href) and href not in self.visited and is_allowed(href):
                    links.append(href)
        except Exception as e:
            logger.warning(f"  Fehler bei {url}: {e}")
            title, content = "", ""
        finally:
            await ctx.close()
        return title, content, links

    async def run(self):
        await self.init()
        queue = [BASE_URL]
        self.visited.add(BASE_URL)

        while queue and len(self.pages) < MAX_PAGES:
            url = queue.pop(0)
            logger.info(f"[{len(self.pages)+1}] {url}")
            title, content, links = await self.scrape_page(url)

            if len(content) > 100:
                self.pages.append({"url": url, "title": title, "content": content})

            for link in links:
                if link not in self.visited:
                    self.visited.add(link)
                    queue.append(link)

        await self.close()
        logger.info(f"✓ {len(self.pages)} Seiten gescraped")


async def main():
    scraper = USTAScraper()
    await scraper.run()

    output = {"source": "usta_aalen", "pages": scraper.pages}
    with open("usta_full_data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"✓ Gespeichert: usta_full_data.json ({len(scraper.pages)} Seiten)")


if __name__ == "__main__":
    asyncio.run(main())

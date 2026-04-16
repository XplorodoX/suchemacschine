"""
scraper_improved.py — Asynchronous version with 10 concurrent workers.

Fixes:
  1. Always expand dynamic content (accordions, tabs, details)
  2. Integrated PDF extraction directly in the main scraper loop
  3. Better section deduplication and content length tracking
  4. CONCURRENCY: 10 simultaneous requests using asyncio + httpx + playwright async
"""

import asyncio
import json
import os
import re
import time
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SITEMAPS = [
    "https://www.hs-aalen.de/sitemap.xml?sitemap=news&cHash=9328a3bb7b686fd62683a27aa38768ed",
    "https://www.hs-aalen.de/sitemap.xml?sitemap=pages&cHash=067b9fa27c3c31031830cfd9d62c4858",
    "https://www.hs-aalen.de/sitemap.xml?sitemap=event&cHash=ab7c558440b5e2921e559a5daa55cab0",
]
ROOT_SITEMAP_INDEX = "https://www.hs-aalen.de/sitemap.xml"

# Local paths for scraper_improved.py
OUTPUT_FILE = "/home/flo/suchemacschine/data.jsonl"
REPORT_FILE = "/home/flo/suchemacschine/scrape_report.json"

# Quality gate and thresholds
MIN_CONTENT_LENGTH = 350
JS_RENDER_THRESHOLD = 1500
CONCURRENCY_LIMIT = 5

NOISE_PATTERNS = [
    r"AI Suche",
    r"Hallo, ich bin deine AI Suche.*",
    r"Häufig gesucht:.*",
    r"Suchbegriff eingeben",
    r"Link in Zwischenablage kopiert",
    r"Kopieren",
    r"Schließen",
    r"Teilen",
    r"Facebook",
    r"LinkedIn",
    r"Email",
    r"Url",
]

PDF_RELEVANT_KEYWORDS = [
    "curriculum", "studienplan", "modul", "stundenplan", "dokument",
    "downloads", "formulare", "ordnung", "satzung", "prüfung", "info",
    "studienangebot", "fachbereich", "service", "international"
]

# ---------------------------------------------------------------------------
# Browser renderer (Playwright Async)
# ---------------------------------------------------------------------------
class BrowserRenderer:
    """Playwright async renderer that also expands dynamic content."""

    def __init__(self):
        self.playwright = None
        self.browser = None

    async def start(self):
        if self.browser:
            return
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)

    async def render_html(self, url: str) -> Optional[str]:
        try:
            await self.start()
            context = await self.browser.new_context(
                user_agent="HSAalenSearchBot/1.2 (+https://www.hs-aalen.de)",
                locale="de-DE",
            )
            page = await context.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            await asyncio.sleep(0.8)
            await page.wait_for_load_state("networkidle", timeout=10000)

            # Expanded JS interaction logic
            await page.evaluate(
                """
                () => {
                  document.querySelectorAll('details').forEach((d) => d.open = true);
                  const expandSelectors = [
                    'button[aria-expanded="false"]',
                    '[role="button"][aria-expanded="false"]',
                    'a[aria-expanded="false"]',
                    '.accordion button',
                    '.accordion-button.collapsed',
                    '.tab button',
                    '[data-bs-toggle="collapse"]',
                    '[data-toggle="collapse"]',
                    '.collapsible',
                    '.read-more-button',
                    '.show-more',
                    '[class*="expand"]',
                    '[class*="accordion"]',
                  ];
                  expandSelectors.forEach((sel) => {
                    document.querySelectorAll(sel).forEach((el) => {
                      try { el.click(); } catch (_) {}
                    });
                  });
                  document.querySelectorAll('.collapse:not(.show)').forEach((el) => {
                    el.classList.add('show');
                    el.style.display = '';
                    el.style.visibility = 'visible';
                    el.style.height = 'auto';
                    el.style.overflow = 'visible';
                  });
                  document.querySelectorAll('[style*="display: none"], [style*="display:none"]').forEach((el) => {
                    const tag = el.tagName.toLowerCase();
                    if (!['script', 'style', 'noscript'].includes(tag)) {
                      el.style.display = '';
                    }
                  });
                  document.querySelectorAll('[role="tab"]:not(.active)').forEach((tab) => {
                    try { tab.click(); } catch (_) {}
                  });
                }
                """
            )
            await asyncio.sleep(0.8)
            html = await page.content()
            await context.close()
            return html
        except Exception as e:
            # Silent on purpose or logged differently? Added local prefix.
            print(f"    JS rendering failed for {url}: {e}")
            return None

    async def close(self):
        if self.browser:
            await self.browser.close()
            self.browser = None
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None


# ---------------------------------------------------------------------------
# PDF extraction helpers (Async)
# ---------------------------------------------------------------------------
def is_pdf_relevant(url: str) -> bool:
    url_lower = url.lower()
    return any(kw in url_lower for kw in PDF_RELEVANT_KEYWORDS)


def find_pdf_links_simple(html: str, page_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    seen = set()
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag.get("href", "").strip()
        if href.lower().endswith(".pdf"):
            absolute = urljoin(page_url, href)
            if absolute not in seen:
                seen.add(absolute)
                links.append(absolute)
    return links


async def download_pdf_text(pdf_url: str, client: httpx.AsyncClient) -> Optional[str]:
    try:
        resp = await client.get(pdf_url, timeout=20.0, follow_redirects=True)
        if resp.status_code != 200:
            return None
        
        def _extract_sync(content):
            import tempfile
            from pypdf import PdfReader
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(content)
                path = tmp.name
            try:
                reader = PdfReader(path)
                pages = [p.extract_text() for p in reader.pages if p.extract_text()]
                return "\n".join(pages)
            finally:
                try: os.unlink(path)
                except: pass

        text = await asyncio.to_thread(_extract_sync, resp.content)
        return text if text and len(text.strip()) > 80 else None
    except Exception as e:
        print(f"    PDF fallback extraction failed for {pdf_url}: {e}")
        return None


async def augment_with_pdfs(record: dict, client: httpx.AsyncClient) -> dict:
    url = record.get("url", "")
    html_content = record.get("_raw_html", "")
    if not url or not html_content or not is_pdf_relevant(url):
        record.pop("_raw_html", None)
        return record
    pdf_links = find_pdf_links_simple(html_content, url)
    if not pdf_links:
        record.pop("_raw_html", None)
        return record
    pdf_sources, pdf_texts = [], []
    for pdf_url in pdf_links[:8]:
        filename = urlparse(pdf_url).path.split("/")[-1]
        text = await download_pdf_text(pdf_url, client)
        if text:
            pdf_sources.append({"url": pdf_url, "filename": filename})
            pdf_texts.append(f"[Dokument: {filename}]\n{text[:3000]}")
            print(f"    [PDF] extracted {len(text)} chars from {filename}")
    if pdf_sources:
        record["pdf_count"] = len(pdf_sources)
        record["pdf_sources"] = pdf_sources
        combined = "\n\n".join(pdf_texts)
        record["content"] = (record.get("content") or "") + "\n\n" + combined
        if "sections" not in record: record["sections"] = []
        record["sections"].append({"heading": "PDF-Dokumente", "text": combined})
        record["content_length"] = len(record["content"])
    record.pop("_raw_html", None)
    return record


# ---------------------------------------------------------------------------
# Sitemap discovery (Async)
# ---------------------------------------------------------------------------
async def discover_sitemaps(client: httpx.AsyncClient):
    try:
        response = await client.get(ROOT_SITEMAP_INDEX, timeout=30.0)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "xml")
        sitemap_urls = [loc.text.strip() for loc in soup.find_all("loc") if loc.text]
        if sitemap_urls:
            print(f"Discovered {len(sitemap_urls)} sitemap files from index")
        return sitemap_urls
    except Exception as e:
        print(f"Could not discover sitemap index ({e}), fallback to configured list")
        return []


async def get_urls_from_sitemap(client: httpx.AsyncClient, sitemap_url: str):
    print(f"Fetching sitemap: {sitemap_url}")
    try:
        response = await client.get(sitemap_url, timeout=30.0)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "xml")
        urls = [loc.text.strip() for loc in soup.find_all("loc") if loc.text]
        print(f"Found {len(urls)} URLs in {sitemap_url}")
        return urls
    except Exception as e:
        print(f"Error fetching sitemap {sitemap_url}: {e}")
        return []


# ---------------------------------------------------------------------------
# Content extraction helpers
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    cleaned = text or ""
    for pattern in NOISE_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()

def path_of(url: str) -> str: return urlparse(url).path.lower()
def is_person_page(url: str) -> bool: return "/person/" in path_of(url)
def is_event_page(url: str) -> bool: return "/veranstaltungen/veranstaltung/" in path_of(url)

def extract_jsonld_sections(soup):
    sections = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw_json = (script.string or script.get_text() or "").strip()
        if not raw_json: continue
        try: payload = json.loads(raw_json)
        except: continue
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict): continue
            parts = []
            for field in ["name", "description", "articleBody", "jobTitle", "startDate", "endDate"]:
                value = item.get(field)
                if isinstance(value, str):
                    t = clean_text(value)
                    if t: parts.append(t)
            location = item.get("location")
            if isinstance(location, dict):
                loc_name = location.get("name")
                if isinstance(loc_name, str) and loc_name.strip():
                    parts.append(clean_text(f"Ort: {loc_name}"))
            if parts:
                text = clean_text(" ".join(parts))
                if len(text) >= 40:
                    sections.append({"heading": "Strukturierte Daten", "text": text})
    return sections

def maybe_add_template_section(url: str, main_container, h1: str):
    sections = []
    main_text = clean_text(main_container.get_text(separator=" ", strip=True))
    if is_person_page(url):
        if main_text: sections.append({"heading": h1 or "Person", "text": main_text})
    if is_event_page(url) and main_text:
        trimmed = re.split(r"Weitere Veranstaltungen", main_text, maxsplit=1, flags=re.IGNORECASE)[0]
        trimmed = clean_text(trimmed)
        if trimmed: sections.append({"heading": h1 or "Veranstaltung", "text": trimmed})
    return sections

def extract_sections(main_container):
    _BLOCK_TAGS = {"h1", "h2", "h3", "p", "li", "dt", "dd", "td", "th", "div"}
    _HEADING_TAGS = {"h1", "h2", "h3"}
    def _is_leaf_div(el): return not el.find(_BLOCK_TAGS)
    sections, current_heading, current_text_parts = [], "Allgemein", []
    for element in main_container.find_all(list(_BLOCK_TAGS), recursive=True):
        if element.name == "div" and not _is_leaf_div(element): continue
        text = clean_text(element.get_text(separator=" ", strip=True))
        if not text: continue
        if element.name in _HEADING_TAGS:
            f_text = clean_text(" ".join(current_text_parts)) if current_text_parts else current_heading
            sections.append({"heading": current_heading, "text": f_text})
            current_text_parts = []
            current_heading = text
        else: current_text_parts.append(text)
    f_text = clean_text(" ".join(current_text_parts)) if current_text_parts else current_heading
    sections.append({"heading": current_heading, "text": f_text})
    deduped = []
    seen = set()
    for s in sections:
        h, t = s.get("heading", ""), s.get("text", "")
        min_len = 4 if len(h) >= 4 else 20
        key = (h, t)
        if key in seen or len(t) < min_len: continue
        seen.add(key)
        deduped.append(s)
    return deduped

def extract_structured_from_html(html: str, url: str):
    soup = BeautifulSoup(html, "html.parser")
    title = clean_text(soup.title.get_text()) if soup.title else ""
    jsonld = extract_jsonld_sections(soup)
    for el in soup(["script", "style", "nav", "footer", "form", "aside", "noscript", "iframe"]): el.decompose()
    main = soup.find("main") or soup.find("article") or soup.find(id=re.compile(r"content|main", re.I)) or soup.body
    if not main: return {"url": url, "title": title, "h1": "", "headings": [], "sections": [], "content": ""}
    for el in main.find_all(class_=re.compile(r"breadcrumb|share|social|nav|menu|pagination|button|modal|ai-search|chat|cookie", re.I)): el.decompose()
    h1_tag = main.find("h1")
    h1 = clean_text(h1_tag.get_text()) if h1_tag else ""
    sects = extract_sections(main)
    sects.extend(maybe_add_template_section(url, main, h1))
    sects.extend(jsonld)
    final_sects, seen = [], set()
    for s in sects:
        h, t = clean_text(s.get("heading", "")), clean_text(s.get("text", ""))
        if not t or (h, t) in seen: continue
        seen.add((h, t))
        final_sects.append({"heading": h or "Allgemein", "text": t})
    if not final_sects:
        fallback = clean_text(main.get_text(separator=" ", strip=True))
        if fallback: final_sects = [{"heading": h1 or "Allgemein", "text": fallback}]
    c_parts = []
    for s in final_sects:
        h, t = s.get("heading", ""), s.get("text", "")
        if h and h != "Allgemein" and h not in t: c_parts.append(h)
        c_parts.append(t)
    return {"url": url, "title": title, "h1": h1, "headings": [s["heading"] for s in final_sects], "sections": final_sects, "content": clean_text(" ".join(c_parts))}

# ---------------------------------------------------------------------------
# Async Worker
# ---------------------------------------------------------------------------
async def extract_content(client: httpx.AsyncClient, renderer: BrowserRenderer, url: str) -> Optional[dict]:
    try:
        response = await client.get(url, timeout=35.0, follow_redirects=True)
        response.raise_for_status()
        raw_html = response.text
        data = extract_structured_from_html(raw_html, url)
        static_len = len(data.get("content", ""))
        final_data, final_html, used_js, js_attempted, rend_len = data, raw_html, False, False, static_len
        if static_len < JS_RENDER_THRESHOLD:
            js_attempted = True
            rend_html = await renderer.render_html(url)
            if rend_html:
                rend_data = extract_structured_from_html(rend_html, url)
                rend_len = len(rend_data.get("content", ""))
                if rend_len > static_len:
                    final_data, final_html, used_js = rend_data, rend_html, True
        final_data.update({"_raw_html": final_html, "used_js_render": used_js, "js_render_attempted": js_attempted,
                          "static_content_length": static_len, "rendered_content_length": rend_len,
                          "content_gain_from_js": max(0, rend_len - static_len)})
        final_data = await augment_with_pdfs(final_data, client)
        final_data["content_length"] = len(final_data.get("content", ""))
        return final_data
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

async def worker(url_queue: asyncio.Queue, results: list, client: httpx.AsyncClient, renderer: BrowserRenderer, sem: asyncio.Semaphore, total: int):
    while True:
        try:
            idx, url = await url_queue.get()
            async with sem:
                print(f"Scraping [{idx+1}/{total}]: {url}")
                extracted = await extract_content(client, renderer, url)
                if extracted: results.append(extracted)
            url_queue.task_done()
        except asyncio.CancelledError: break
        except Exception as e:
            print(f"Worker error: {e}")
            url_queue.task_done()

async def main_async():
    renderer = BrowserRenderer()
    async with httpx.AsyncClient(headers={"User-Agent": "HSAalenSearchBot/1.2", "Accept-Language": "de,en;q=0.8"},
                               timeout=httpx.Timeout(30.0, connect=10.0), follow_redirects=True) as client:
        discovered = await discover_sitemaps(client)
        urls = sorted(set(sum([await get_urls_from_sitemap(client, s) for s in (discovered if discovered else SITEMAPS)], [])))
        total = len(urls)
        print(f"Total unique URLs: {total}")
        q = asyncio.Queue()
        for i, u in enumerate(urls): await q.put((i, u))
        results, sem = [], asyncio.Semaphore(CONCURRENCY_LIMIT)
        workers = [asyncio.create_task(worker(q, results, client, renderer, sem, total)) for _ in range(CONCURRENCY_LIMIT)]
        processed = 0
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            while not q.empty() or processed < total:
                await asyncio.sleep(2)
                while results:
                    rec = results.pop(0)
                    processed += 1
                    if len(rec.get("content", "")) >= MIN_CONTENT_LENGTH:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        if processed % 10 == 0: print(f"  Progress: {processed}/{total} saved")
                if q.empty() and not results and all(w.done() for w in workers): break
        for w in workers: w.cancel()
        await renderer.close()
        print(f"Done! Saved to {OUTPUT_FILE}")

if __name__ == "__main__": asyncio.run(main_async())

"""
scraper_improved.py — Drop-in replacement for scrapers/scraper.py

Fixes:
  1. Always expand dynamic content (accordions, tabs, details) — not just when content < 350 chars
  2. Integrated PDF extraction directly in the main scraper loop
  3. Better section deduplication and content length tracking
"""

import json
import re
import time
from typing import Optional
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SITEMAPS = [
    "https://www.hs-aalen.de/sitemap.xml?sitemap=news&cHash=9328a3bb7b686fd62683a27aa38768ed",
    "https://www.hs-aalen.de/sitemap.xml?sitemap=pages&cHash=067b9fa27c3c31031830cfd9d62c4858",
    "https://www.hs-aalen.de/sitemap.xml?sitemap=event&cHash=ab7c558440b5e2921e559a5daa55cab0",
]
ROOT_SITEMAP_INDEX = "https://www.hs-aalen.de/sitemap.xml"

OUTPUT_FILE = "/home/flo/suchemacschine/data.jsonl"
REPORT_FILE = "/home/flo/suchemacschine/scrape_report.json"

# FIX 1 — raise threshold so short-but-expandable pages also get JS rendering
MIN_CONTENT_LENGTH = 350       # Still used as quality gate for saving records
JS_RENDER_THRESHOLD = 1500     # JS rendering kicks in whenever static content < 1500 chars
                               # (was: only when < 350 — missed all pages with partial content)

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

# FIX 2 — PDF extraction trigger: URL keywords that hint at PDF-bearing pages
PDF_RELEVANT_KEYWORDS = [
    "curriculum", "studienplan", "modul", "stundenplan", "dokument",
    "downloads", "formulare", "ordnung", "satzung", "prüfung", "info",
    "studienangebot", "fachbereich", "service", "international"
]


# ---------------------------------------------------------------------------
# Browser renderer (Playwright)
# ---------------------------------------------------------------------------
class BrowserRenderer:
    """Playwright renderer that also expands dynamic content."""

    def __init__(self):
        self.playwright = None
        self.browser = None

    @property
    def available(self) -> bool:
        return sync_playwright is not None

    def start(self):
        if not self.available or self.browser:
            return
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)

    def render_html(self, url: str) -> Optional[str]:
        if not self.available:
            return None
        try:
            self.start()
            context = self.browser.new_context(
                user_agent="HSAalenSearchBot/1.2 (+https://www.hs-aalen.de)",
                locale="de-DE",
            )
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            page.wait_for_timeout(800)
            page.wait_for_load_state("networkidle", timeout=10000)

            # FIX 1 — Expanded JS interaction logic
            # Now handles many more accordion/collapse patterns used on HS Aalen
            page.evaluate(
                """
                () => {
                  // 1. Native <details> elements
                  document.querySelectorAll('details').forEach((d) => d.open = true);

                  // 2. Generic expand buttons (aria-expanded pattern)
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
                      try {
                        el.click();
                      } catch (_) {}
                    });
                  });

                  // 3. Force-show hidden content containers (Bootstrap collapse etc.)
                  document.querySelectorAll('.collapse:not(.show)').forEach((el) => {
                    el.classList.add('show');
                    el.style.display = '';
                    el.style.visibility = 'visible';
                    el.style.height = 'auto';
                    el.style.overflow = 'visible';
                  });

                  // 4. Reveal content hidden via inline style
                  document.querySelectorAll('[style*="display: none"], [style*="display:none"]').forEach((el) => {
                    const tag = el.tagName.toLowerCase();
                    if (!['script', 'style', 'noscript'].includes(tag)) {
                      el.style.display = '';
                    }
                  });

                  // 5. Tab panels — activate all tabs so their content is readable
                  document.querySelectorAll('[role="tab"]:not(.active)').forEach((tab) => {
                    try { tab.click(); } catch (_) {}
                  });
                }
                """
            )
            # Wait for animations/transitions to complete
            page.wait_for_timeout(800)
            html = page.content()
            context.close()
            return html
        except Exception as e:
            print(f"JS rendering failed for {url}: {e}")
            return None

    def close(self):
        if self.browser:
            self.browser.close()
            self.browser = None
        if self.playwright:
            self.playwright.stop()
            self.playwright = None


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------
def create_session():
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "HSAalenSearchBot/1.1 (+https://www.hs-aalen.de)",
            "Accept-Language": "de,en;q=0.8",
        }
    )
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# ---------------------------------------------------------------------------
# PDF extraction helpers (FIX 2)
# ---------------------------------------------------------------------------
def is_pdf_relevant(url: str) -> bool:
    """Check if the page is likely to contain useful PDFs."""
    url_lower = url.lower()
    return any(kw in url_lower for kw in PDF_RELEVANT_KEYWORDS)


def find_pdf_links_simple(html: str, page_url: str) -> list[str]:
    """Extract all PDF links from an HTML string."""
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


def download_pdf_text(pdf_url: str, session: requests.Session) -> Optional[str]:
    """Download a PDF and return its text content."""
    # Try to import the project's own extractor; fall back to a simple version
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
        from pdf_extractor import download_and_extract_pdf
        return download_and_extract_pdf(pdf_url, session)
    except ImportError:
        pass

    # Minimal fallback using pypdf
    try:
        import tempfile
        from pypdf import PdfReader

        resp = session.get(pdf_url, timeout=20)
        resp.raise_for_status()
        if "pdf" not in resp.headers.get("content-type", "").lower() and not pdf_url.lower().endswith(".pdf"):
            return None

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(resp.content)
            path = tmp.name

        try:
            reader = PdfReader(path)
            pages = [p.extract_text() for p in reader.pages if p.extract_text()]
            text = "\n".join(pages)
            return text if len(text.strip()) > 80 else None
        finally:
            import os
            try:
                os.unlink(path)
            except Exception:
                pass
    except Exception as e:
        print(f"    PDF fallback extraction failed for {pdf_url}: {e}")
        return None


def augment_with_pdfs(record: dict, session: requests.Session) -> dict:
    """
    FIX 2 — Download and embed PDF content directly into the scraped record.

    Only runs when the URL matches PDF-relevant keywords to keep scraping fast.
    Adds pdf_sources, pdf_count, and appends PDF text to content/sections.
    """
    url = record.get("url", "")
    html_content = record.get("_raw_html", "")  # stored temporarily below

    if not url or not html_content or not is_pdf_relevant(url):
        record.pop("_raw_html", None)
        return record

    pdf_links = find_pdf_links_simple(html_content, url)
    if not pdf_links:
        record.pop("_raw_html", None)
        return record

    pdf_sources = []
    pdf_texts = []

    for pdf_url in pdf_links[:8]:  # Cap at 8 PDFs per page
        filename = urlparse(pdf_url).path.split("/")[-1]
        print(f"  [PDF] Extracting: {filename}")
        text = download_pdf_text(pdf_url, session)
        if text:
            pdf_sources.append({"url": pdf_url, "filename": filename})
            # Truncate individual PDFs to avoid blowing up the record size
            pdf_texts.append(f"[Dokument: {filename}]\n{text[:3000]}")
            print(f"    ✓ {len(text)} chars extracted from {filename}")
        else:
            print(f"    ✗ No text from {filename}")

    if pdf_sources:
        record["pdf_count"] = len(pdf_sources)
        record["pdf_sources"] = pdf_sources

        # Append PDF text to existing content
        combined_pdf_text = "\n\n".join(pdf_texts)
        record["content"] = (record.get("content") or "") + "\n\n" + combined_pdf_text

        # Add as a dedicated section so section-aware chunkers include it
        if "sections" not in record:
            record["sections"] = []
        record["sections"].append({
            "heading": "PDF-Dokumente",
            "text": combined_pdf_text,
        })

        record["content_length"] = len(record["content"])

    record.pop("_raw_html", None)
    return record


# ---------------------------------------------------------------------------
# Sitemap discovery
# ---------------------------------------------------------------------------
def discover_sitemaps(session):
    try:
        response = session.get(ROOT_SITEMAP_INDEX, timeout=(10, 30))
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "xml")
        sitemap_urls = [loc.text.strip() for loc in soup.find_all("loc") if loc.text]
        if sitemap_urls:
            print(f"Discovered {len(sitemap_urls)} sitemap files from index")
        return sitemap_urls
    except Exception as e:
        print(f"Could not discover sitemap index ({e}), fallback to configured list")
        return []


def get_urls_from_sitemap(session, sitemap_url):
    print(f"Fetching sitemap: {sitemap_url}")
    try:
        response = session.get(sitemap_url, timeout=(10, 30))
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


def path_of(url: str) -> str:
    return urlparse(url).path.lower()


def is_person_page(url: str) -> bool:
    return "/person/" in path_of(url)


def is_event_page(url: str) -> bool:
    return "/veranstaltungen/veranstaltung/" in path_of(url)


def extract_jsonld_sections(soup):
    sections = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw_json = (script.string or script.get_text() or "").strip()
        if not raw_json:
            continue
        try:
            import json as _json
            payload = _json.loads(raw_json)
        except Exception:
            continue

        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            parts = []
            for field in ["name", "description", "articleBody", "jobTitle", "startDate", "endDate"]:
                value = item.get(field)
                if isinstance(value, str):
                    text = clean_text(value)
                    if text:
                        parts.append(text)
            location = item.get("location")
            if isinstance(location, dict):
                loc_name = location.get("name")
                if isinstance(loc_name, str) and loc_name.strip():
                    parts.append(clean_text(f"Ort: {loc_name}"))
            if parts:
                section_text = clean_text(" ".join(parts))
                if len(section_text) >= 40:
                    sections.append({"heading": "Strukturierte Daten", "text": section_text})
    return sections


def maybe_add_template_section(url: str, main_container, h1: str):
    sections = []
    main_text = clean_text(main_container.get_text(separator=" ", strip=True))

    if is_person_page(url):
        if main_text:
            sections.append({"heading": h1 or "Person", "text": main_text})

    if is_event_page(url) and main_text:
        trimmed = re.split(r"Weitere Veranstaltungen", main_text, maxsplit=1, flags=re.IGNORECASE)[0]
        trimmed = clean_text(trimmed)
        if trimmed:
            sections.append({"heading": h1 or "Veranstaltung", "text": trimmed})

    return sections


def extract_sections(main_container):
    """
    Walk the main content container and build sections keyed by the most
    recent heading.

    Changes vs. the original:
    1. Heading-only sections are NO LONGER silently dropped.  When a heading
       has no following body text the heading string itself is stored as the
       section text so it ends up in the searchable `content` field.
    2. Leaf <div> elements are now scanned in addition to the original tag
       list.  This captures card/timeline descriptions that live in
       `<div class="…__description">` wrappers without being duplicated from
       their parent containers.
    3. The minimum-length gate now uses 4 chars (instead of 20) when the
       heading is itself meaningful, so short-but-real headings survive.
    """

    _BLOCK_TAGS = {"h1", "h2", "h3", "p", "li", "dt", "dd", "td", "th", "div"}
    _HEADING_TAGS = {"h1", "h2", "h3"}

    def _is_leaf_div(el):
        """Return True when a <div> contains no nested block elements."""
        return not el.find(_BLOCK_TAGS)

    sections = []
    current_heading = "Allgemein"
    current_text_parts = []

    for element in main_container.find_all(list(_BLOCK_TAGS), recursive=True):
        # Only process leaf divs to avoid duplicating text from nested blocks.
        if element.name == "div" and not _is_leaf_div(element):
            continue

        text = clean_text(element.get_text(separator=" ", strip=True))
        if not text:
            continue

        if element.name in _HEADING_TAGS:
            # Flush the current accumulator.  If it's empty we still emit the
            # heading so that it contributes to the searchable content.
            flush_text = clean_text(" ".join(current_text_parts)) if current_text_parts else current_heading
            sections.append({"heading": current_heading, "text": flush_text})
            current_text_parts = []
            current_heading = text
        else:
            current_text_parts.append(text)

    # Final flush
    flush_text = clean_text(" ".join(current_text_parts)) if current_text_parts else current_heading
    sections.append({"heading": current_heading, "text": flush_text})

    deduped_sections = []
    seen: set = set()
    for section in sections:
        heading = section.get("heading", "")
        text = section.get("text", "")
        # Accept short text when the heading itself is meaningful (>= 4 chars).
        min_len = 4 if len(heading) >= 4 else 20
        key = (heading, text)
        if key in seen or len(text) < min_len:
            continue
        seen.add(key)
        deduped_sections.append(section)

    return deduped_sections


def extract_structured_from_html(html: str, url: str):
    soup = BeautifulSoup(html, "html.parser")
    title = clean_text(soup.title.get_text()) if soup.title else ""
    jsonld_sections = extract_jsonld_sections(soup)

    for element in soup(["script", "style", "nav", "footer", "form", "aside", "noscript", "iframe"]):
        element.decompose()

    main_container = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id=re.compile(r"content|main", re.I))
        or soup.body
    )

    if not main_container:
        return {"url": url, "title": title, "h1": "", "headings": [], "sections": [], "content": ""}

    exclude_patterns = r"breadcrumb|share|social|nav|menu|pagination|button|modal|ai-search|chat|cookie"
    for element in main_container.find_all(class_=re.compile(exclude_patterns, re.I)):
        element.decompose()

    h1_tag = main_container.find("h1")
    h1 = clean_text(h1_tag.get_text()) if h1_tag else ""

    sections = extract_sections(main_container)
    sections.extend(maybe_add_template_section(url, main_container, h1))
    sections.extend(jsonld_sections)

    deduped_sections = []
    seen: set = set()
    for section in sections:
        heading = clean_text(section.get("heading", ""))
        text = clean_text(section.get("text", ""))
        if not text:
            continue
        key = (heading, text)
        if key in seen:
            continue
        seen.add(key)
        deduped_sections.append({"heading": heading or "Allgemein", "text": text})

    sections = deduped_sections

    if not sections:
        fallback_text = clean_text(main_container.get_text(separator=" ", strip=True))
        if fallback_text:
            sections = [{"heading": h1 or "Allgemein", "text": fallback_text}]

    # ---------------------------------------------------------------------------
    # FIX: include heading text in the searchable content string
    # Previously only section["text"] was joined; now non-generic headings are
    # prepended so queries like "Magic: The Gathering" can match.
    # ---------------------------------------------------------------------------
    content_parts = []
    for section in sections:
        heading = section.get("heading", "")
        text = section.get("text", "")
        # Avoid duplicating when heading == text (heading-only sections)
        if heading and heading != "Allgemein" and heading not in text:
            content_parts.append(heading)
        content_parts.append(text)

    content = clean_text(" ".join(content_parts))
    headings = [section["heading"] for section in sections if section.get("heading")]

    return {
        "url": url,
        "title": title,
        "h1": h1,
        "headings": headings,
        "sections": sections,
        "content": content,
    }


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------
def extract_content(session, renderer: BrowserRenderer, url: str) -> Optional[dict]:
    print(f"Scraping: {url}")
    try:
        response = session.get(url, timeout=(10, 35))
        response.raise_for_status()
        raw_html = response.text
        static_data = extract_structured_from_html(raw_html, url)
        static_len = len(static_data.get("content", ""))

        used_js = False
        js_attempted = False
        rendered_len = static_len
        final_data = static_data
        final_html = raw_html

        # FIX 1 — raise threshold: also render JS for partially-filled pages
        if renderer.available and static_len < JS_RENDER_THRESHOLD:
            js_attempted = True
            rendered_html = renderer.render_html(url)
            if rendered_html:
                rendered_data = extract_structured_from_html(rendered_html, url)
                rendered_len = len(rendered_data.get("content", ""))
                if rendered_len > static_len:
                    final_data = rendered_data
                    final_html = rendered_html
                    used_js = True
                    print(f"  JS gain: {static_len} → {rendered_len} chars (+{rendered_len - static_len})")

        # Store raw HTML temporarily for PDF extraction
        final_data["_raw_html"] = final_html

        final_data["used_js_render"] = used_js
        final_data["js_render_attempted"] = js_attempted
        final_data["static_content_length"] = static_len
        final_data["rendered_content_length"] = rendered_len
        final_data["content_gain_from_js"] = max(0, rendered_len - static_len)

        # FIX 2 — integrated PDF extraction
        final_data = augment_with_pdfs(final_data, session)

        final_data["content_length"] = len(final_data.get("content", ""))
        return final_data

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    session = create_session()
    renderer = BrowserRenderer()

    discovered_sitemaps = discover_sitemaps(session)
    sitemap_list = discovered_sitemaps if discovered_sitemaps else SITEMAPS

    all_urls = []
    for sitemap in sitemap_list:
        all_urls.extend(get_urls_from_sitemap(session, sitemap))

    unique_urls = sorted(set(all_urls))
    print(f"Total unique URLs to scrape: {len(unique_urls)}")

    failed_urls = []
    empty_content_urls = []
    saved_records = 0
    js_render_attempted_count = 0
    js_render_used_count = 0
    pdf_pages_count = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, url in enumerate(unique_urls):
            extracted = extract_content(session, renderer, url)
            if extracted and extracted.get("content"):
                if len(extracted.get("content", "")) < MIN_CONTENT_LENGTH:
                    empty_content_urls.append(url)
                else:
                    f.write(json.dumps(extracted, ensure_ascii=False) + "\n")
                    saved_records += 1
                    if extracted.get("js_render_attempted"):
                        js_render_attempted_count += 1
                    if extracted.get("used_js_render"):
                        js_render_used_count += 1
                    if extracted.get("pdf_count", 0) > 0:
                        pdf_pages_count += 1
            elif extracted is not None:
                empty_content_urls.append(url)
            else:
                failed_urls.append(url)

            if i % 10 == 0:
                print(f"Progress: {i}/{len(unique_urls)} | saved: {saved_records} | pdfs: {pdf_pages_count}")

            time.sleep(0.1)

    renderer.close()

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_urls": len(unique_urls),
        "saved_records": saved_records,
        "failed_count": len(failed_urls),
        "empty_content_count": len(empty_content_urls),
        "js_render_attempted_count": js_render_attempted_count,
        "js_render_used_count": js_render_used_count,
        "pdf_pages_count": pdf_pages_count,
        "failed_urls": failed_urls,
        "empty_content_urls": empty_content_urls,
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)

    print(f"\nDone! Saved: {saved_records} | PDFs embedded: {pdf_pages_count} | JS used: {js_render_used_count}")


if __name__ == "__main__":
    main()

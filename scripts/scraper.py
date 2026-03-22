import json
import re
import time
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None

# Sitemaps provided by the user
SITEMAPS = [
    "https://www.hs-aalen.de/sitemap.xml?sitemap=news&cHash=9328a3bb7b686fd62683a27aa38768ed",
    "https://www.hs-aalen.de/sitemap.xml?sitemap=pages&cHash=067b9fa27c3c31031830cfd9d62c4858",
    "https://www.hs-aalen.de/sitemap.xml?sitemap=event&cHash=ab7c558440b5e2921e559a5daa55cab0",
]
ROOT_SITEMAP_INDEX = "https://www.hs-aalen.de/sitemap.xml"

OUTPUT_FILE = "/Users/merluee/Desktop/suchemacschine/data.jsonl"
REPORT_FILE = "/Users/merluee/Desktop/suchemacschine/scrape_report.json"
MIN_CONTENT_LENGTH = 350
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


class BrowserRenderer:
    """Optional Playwright renderer for pages with JS-dependent content."""

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

            # Expand common collapsible elements to include hidden sub-content.
            page.evaluate(
                """
                () => {
                  document.querySelectorAll('details').forEach((d) => d.open = true);
                  const selectors = [
                    'button[aria-expanded="false"]',
                    '[role="button"][aria-expanded="false"]',
                    '.accordion button',
                    '.tab button'
                  ];
                  selectors.forEach((sel) => {
                    document.querySelectorAll(sel).forEach((el) => {
                      try { el.click(); } catch (_) {}
                    });
                  });
                }
                """
            )
            page.wait_for_timeout(500)
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


def discover_sitemaps(session):
    """Loads the root sitemap index and returns contained sitemap URLs."""
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


def clean_text(text: str) -> str:
    cleaned = (text or "")
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
            payload = json.loads(raw_json)
        except Exception:
            continue

        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue

            parts = []
            for field in [
                "name",
                "description",
                "articleBody",
                "jobTitle",
                "startDate",
                "endDate",
            ]:
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
    sections = []
    current_heading = "Allgemein"
    current_text_parts = []

    for element in main_container.find_all(["h1", "h2", "h3", "p", "li", "dt", "dd", "td", "th"], recursive=True):
        text = clean_text(element.get_text(separator=" ", strip=True))
        if not text:
            continue

        if element.name in {"h1", "h2", "h3"}:
            if current_text_parts:
                sections.append(
                    {
                        "heading": current_heading,
                        "text": clean_text(" ".join(current_text_parts)),
                    }
                )
                current_text_parts = []
            current_heading = text
        else:
            current_text_parts.append(text)

    if current_text_parts:
        sections.append({"heading": current_heading, "text": clean_text(" ".join(current_text_parts))})

    deduped_sections = []
    seen = set()
    for section in sections:
        key = (section["heading"], section["text"])
        if key in seen or len(section["text"]) < 20:
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

    main_container = soup.find("main") or soup.find("article") or soup.find(id=re.compile(r"content|main", re.I))
    if not main_container:
        main_container = soup.body

    if not main_container:
        return {
            "url": url,
            "title": title,
            "h1": "",
            "headings": [],
            "sections": [],
            "content": "",
        }

    exclude_patterns = r"breadcrumb|share|social|nav|menu|pagination|button|modal|ai-search|chat|cookie"
    for element in main_container.find_all(class_=re.compile(exclude_patterns, re.I)):
        element.decompose()

    h1_tag = main_container.find("h1")
    h1 = clean_text(h1_tag.get_text()) if h1_tag else ""

    sections = extract_sections(main_container)
    sections.extend(maybe_add_template_section(url, main_container, h1))
    sections.extend(jsonld_sections)

    deduped_sections = []
    seen = set()
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

    content = clean_text(" ".join(section["text"] for section in sections))
    headings = [section["heading"] for section in sections if section.get("heading")]

    return {
        "url": url,
        "title": title,
        "h1": h1,
        "headings": headings,
        "sections": sections,
        "content": content,
    }


def extract_content(session, renderer: BrowserRenderer, url):
    print(f"Scraping content from: {url}")
    try:
        response = session.get(url, timeout=(10, 35))
        response.raise_for_status()
        static_data = extract_structured_from_html(response.text, url)
        static_len = len(static_data.get("content", ""))

        used_js = False
        js_attempted = False
        rendered_len = static_len
        final_data = static_data

        # Fallback to JS rendering when static HTML is suspiciously short.
        if renderer.available and static_len < MIN_CONTENT_LENGTH:
            js_attempted = True
            rendered_html = renderer.render_html(url)
            if rendered_html:
                rendered_data = extract_structured_from_html(rendered_html, url)
                rendered_len = len(rendered_data.get("content", ""))
                if rendered_len > static_len:
                    final_data = rendered_data
                    used_js = True

        final_data["used_js_render"] = used_js
        final_data["js_render_attempted"] = js_attempted
        final_data["static_content_length"] = static_len
        final_data["rendered_content_length"] = rendered_len
        final_data["content_gain_from_js"] = max(0, rendered_len - static_len)
        final_data["content_length"] = len(final_data.get("content", ""))
        return final_data
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


def main():
    session = create_session()
    renderer = BrowserRenderer()

    discovered_sitemaps = discover_sitemaps(session)
    sitemap_list = discovered_sitemaps if discovered_sitemaps else SITEMAPS

    all_urls = []
    for sitemap in sitemap_list:
        all_urls.extend(get_urls_from_sitemap(session, sitemap))

    # Remove duplicates but keep deterministic order for reproducibility.
    unique_urls = sorted(set(all_urls))
    print(f"Total unique URLs to scrape: {len(unique_urls)}")

    failed_urls = []
    empty_content_urls = []
    saved_records = 0
    js_render_attempted_count = 0
    js_render_used_count = 0
    js_positive_gain_count = 0
    total_content_gain = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, url in enumerate(unique_urls):
            extracted = extract_content(session, renderer, url)
            if extracted and extracted.get("content"):
                record = extracted
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                saved_records += 1
                if record.get("js_render_attempted"):
                    js_render_attempted_count += 1
                if record.get("used_js_render"):
                    js_render_used_count += 1
                gain = int(record.get("content_gain_from_js", 0) or 0)
                if gain > 0:
                    js_positive_gain_count += 1
                    total_content_gain += gain
            elif extracted is not None and extracted.get("content", "") == "":
                empty_content_urls.append(url)
            else:
                failed_urls.append(url)

            # Rate limiting / polite scraping
            if i % 10 == 0:
                print(f"Progress: {i}/{len(unique_urls)}")

            # Small delay to be nice to the server
            time.sleep(0.1)

    renderer.close()

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sitemaps_used": sitemap_list,
        "total_urls_from_sitemaps": len(all_urls),
        "unique_urls": len(unique_urls),
        "saved_records": saved_records,
        "failed_count": len(failed_urls),
        "empty_content_count": len(empty_content_urls),
        "js_render_attempted_count": js_render_attempted_count,
        "js_render_used_count": js_render_used_count,
        "js_positive_gain_count": js_positive_gain_count,
        "total_content_gain": total_content_gain,
        "failed_urls": failed_urls,
        "empty_content_urls": empty_content_urls,
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)

    print(f"Scraping complete. Results saved to {OUTPUT_FILE}")
    print(f"Scrape report written to {REPORT_FILE}")


if __name__ == "__main__":
    main()

"""
Generic document loaders for the search framework.

Each loader implements a load() method that returns a list of Document dicts:

  {
    "url": str,
    "title": str,
    "content": str,           # full plain text
    "sections": [             # optional — improves contextual chunking
      {"heading": str, "text": str}
    ],
    "h1": str,                # optional
    "type": str,              # "webpage" | "timetable" | "document" | "asta"
    "metadata": dict,         # source-specific extra fields
  }

Available loaders:
  WebsiteLoader   — recursive website crawler (requests + BeautifulSoup)
  StarplanLoader  — Starplan iCal API (HS Aalen timetable system)
  FolderLoader    — local files (.pdf, .txt, .md, .docx)
  UrlListLoader   — explicit list of URLs
"""

from __future__ import annotations

import re
import time
import logging
from pathlib import Path
from typing import Generator
from urllib.parse import urljoin, urlparse, urlencode
from urllib.robotparser import RobotFileParser

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url: str, timeout: int = 15, headers: dict | None = None) -> str | None:
    """HTTP GET with simple retry logic. Returns response text or None."""
    try:
        import requests
    except ImportError:
        raise ImportError("Install requests: pip install requests")

    default_headers = {
        "User-Agent": "SearchFramework/1.0 (educational search engine; contact: info@hs-aalen.de)"
    }
    if headers:
        default_headers.update(headers)

    for attempt in range(3):
        try:
            r = requests.get(url, headers=default_headers, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            if attempt == 2:
                log.warning("GET %s failed after 3 attempts: %s", url, e)
            else:
                time.sleep(1.5 ** attempt)
    return None


def _parse_html_sections(soup) -> tuple[str, str, str, list[dict]]:
    """
    Extract (title, h1, content, sections) from a BeautifulSoup object.
    Sections are built from <section> tags or heading hierarchy.
    """
    # Remove boilerplate elements
    for tag in soup.find_all(["nav", "footer", "script", "style", "noscript", "aside"]):
        tag.decompose()

    title = (soup.find("title") or soup.new_tag("x")).get_text(strip=True)
    h1_tag = soup.find("h1")
    h1 = h1_tag.get_text(strip=True) if h1_tag else ""

    sections = []

    # Strategy 1: explicit <section> tags
    section_tags = soup.find_all("section")
    if section_tags:
        for sec in section_tags:
            heading_tag = sec.find(re.compile(r"^h[1-6]$"))
            heading = heading_tag.get_text(strip=True) if heading_tag else "Allgemein"
            text = sec.get_text(separator=" ", strip=True)
            if text and len(text) > 30:
                sections.append({"heading": heading, "text": text})

    # Strategy 2: heading hierarchy in main content
    if not sections:
        main = soup.find("main") or soup.find("article") or soup.find("body")
        if main:
            current_heading = "Allgemein"
            current_text_parts: list[str] = []
            for child in main.descendants:
                if not hasattr(child, "name"):
                    continue
                if re.match(r"^h[1-6]$", child.name or ""):
                    # Save previous block
                    block = " ".join(current_text_parts).strip()
                    if block and len(block) > 30:
                        sections.append({"heading": current_heading, "text": block})
                    current_heading = child.get_text(strip=True)
                    current_text_parts = []
                elif child.name == "p":
                    txt = child.get_text(separator=" ", strip=True)
                    if txt:
                        current_text_parts.append(txt)
            # Flush last block
            block = " ".join(current_text_parts).strip()
            if block and len(block) > 30:
                sections.append({"heading": current_heading, "text": block})

    content = soup.get_text(separator=" ", strip=True)
    return title, h1, content, sections


# ---------------------------------------------------------------------------
# WebsiteLoader
# ---------------------------------------------------------------------------

class WebsiteLoader:
    """
    Recursively crawls a website and returns one Document per page.

    Config keys:
      url            — seed URL
      max_depth      — crawl depth (default: 3)
      respect_robots_txt — honour robots.txt (default: True)
      max_pages      — hard cap on number of pages (default: 500)
      js_rendering   — use Playwright for JS-heavy sites (default: False)
    """

    def __init__(self, config: dict):
        self.seed_url = config["url"].rstrip("/")
        self.max_depth = int(config.get("max_depth", 3))
        self.respect_robots = config.get("respect_robots_txt", True)
        self.max_pages = int(config.get("max_pages", 500))
        self.js_rendering = config.get("js_rendering", False)
        self.result_type = config.get("result_type", "webpage")
        self.domain = urlparse(self.seed_url).netloc
        self._robot_parser: RobotFileParser | None = None

    def _robot_allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        if self._robot_parser is None:
            rp = RobotFileParser()
            robots_url = f"{urlparse(self.seed_url).scheme}://{self.domain}/robots.txt"
            rp.set_url(robots_url)
            try:
                rp.read()
            except Exception:
                pass
            self._robot_parser = rp
        return self._robot_parser.can_fetch("*", url)

    def _extract_links(self, soup, base_url: str) -> list[str]:
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
                continue
            full = urljoin(base_url, href).split("#")[0].split("?")[0]
            parsed = urlparse(full)
            if parsed.netloc == self.domain and parsed.scheme in ("http", "https"):
                # Skip non-text resources
                if not re.search(r"\.(jpg|jpeg|png|gif|svg|ico|css|js|woff|woff2|ttf|zip|rar)$", full, re.I):
                    links.append(full)
        return links

    def _scrape_url(self, url: str) -> dict | None:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("Install beautifulsoup4: pip install beautifulsoup4 lxml")

        html = _get(url)
        if not html:
            return None
        soup = BeautifulSoup(html, "lxml")
        title, h1, content, sections = _parse_html_sections(soup)
        if not content or len(content.strip()) < 50:
            return None
        return {
            "url": url,
            "title": title,
            "h1": h1,
            "content": content,
            "sections": sections,
            "type": self.result_type,
            "metadata": {"domain": self.domain},
        }, soup

    def load(self) -> list[dict]:
        log.info("WebsiteLoader: starting crawl of %s (depth=%d)", self.seed_url, self.max_depth)
        visited: set[str] = set()
        # queue: (url, depth)
        queue: list[tuple[str, int]] = [(self.seed_url, 0)]
        documents: list[dict] = []

        while queue and len(documents) < self.max_pages:
            url, depth = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            if not self._robot_allowed(url):
                log.debug("Blocked by robots.txt: %s", url)
                continue

            result = self._scrape_url(url)
            if result is None:
                continue

            doc, soup = result
            documents.append(doc)
            log.debug("[%d/%d] Scraped: %s", len(documents), self.max_pages, url)

            if depth < self.max_depth:
                for link in self._extract_links(soup, url):
                    if link not in visited:
                        queue.append((link, depth + 1))

            time.sleep(0.3)  # polite crawl delay

        log.info("WebsiteLoader: finished — %d pages from %s", len(documents), self.seed_url)
        return documents


# ---------------------------------------------------------------------------
# StarplanLoader  (HS Aalen / HTW Aalen timetable system)
# ---------------------------------------------------------------------------

class StarplanLoader:
    """
    Loads lecture timetable data from the Starplan scheduling system via its
    JSON API + iCal export. Used for HTW/HS Aalen's Vorlesungsplan.

    Config keys:
      base_url           — e.g. "https://vorlesungen.htw-aalen.de/splan"
      planning_unit_id   — integer, default 50
      max_org_groups     — 0 = no limit
      max_planning_groups— 0 = no limit
    """

    def __init__(self, config: dict):
        self.base_url = config["base_url"].rstrip("/")
        self.pu_id = int(config.get("planning_unit_id", 50))
        self.max_og = int(config.get("max_org_groups", 0))
        self.max_pg = int(config.get("max_planning_groups", 0))
        self.result_type = config.get("result_type", "timetable")

    def _api(self, params: dict) -> list | dict | None:
        import json
        url = f"{self.base_url}/json?" + urlencode(params)
        raw = _get(url)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def _get_org_groups(self) -> list[dict]:
        data = self._api({"m": "getogs", "pu": self.pu_id})
        if not isinstance(data, list):
            return []
        return data[: self.max_og] if self.max_og else data

    def _get_planning_groups(self, og_id: int) -> list[dict]:
        data = self._api({"m": "getPgsExt", "pu": self.pu_id, "og": og_id})
        if not isinstance(data, list):
            return []
        return data[: self.max_pg] if self.max_pg else data

    def _get_ical(self, pg_id: int) -> str | None:
        url = (
            f"{self.base_url}/ical?"
            f"lan=de&puid={self.pu_id}&type=pg&pgid={pg_id}"
        )
        return _get(url)

    @staticmethod
    def _parse_ical(ical_text: str, program_name: str, pg_id: int, base_url: str) -> list[dict]:
        """Parse iCal text into timetable documents."""
        DAYS_DE = {
            "MO": "Montag", "TU": "Dienstag", "WE": "Mittwoch",
            "TH": "Donnerstag", "FR": "Freitag", "SA": "Samstag", "SO": "Sonntag",
        }

        def _unfold(text: str) -> str:
            # iCal line folding: continuation lines start with space/tab
            return re.sub(r"\r?\n[ \t]", "", text)

        def _dtstamp_to_day_time(dtvalue: str) -> tuple[str, str]:
            # DTSTART;TZID=...:20240415T090000 or DTSTART:20240415T090000
            dt = re.sub(r"^[^:]*:", "", dtvalue)
            try:
                year, month, day = dt[:4], dt[4:6], dt[6:8]
                hour, minute = dt[9:11], dt[11:13]
                import datetime
                weekday_abbr = datetime.date(int(year), int(month), int(day)).strftime("%a").upper()[:2]
                day_name = DAYS_DE.get(weekday_abbr, weekday_abbr)
                return day_name, f"{hour}:{minute}"
            except Exception:
                return "", ""

        ical_text = _unfold(ical_text)
        documents = []
        event: dict[str, str] = {}
        in_event = False

        for raw_line in ical_text.splitlines():
            line = raw_line.strip()
            if line == "BEGIN:VEVENT":
                in_event = True
                event = {}
            elif line == "END:VEVENT":
                in_event = False
                # Build timetable document
                summary = event.get("SUMMARY", "")
                location = event.get("LOCATION", "")
                description = event.get("DESCRIPTION", "")
                dtstart = event.get("DTSTART", "")
                dtend = event.get("DTEND", "")

                day, start_time = _dtstamp_to_day_time(dtstart)
                _, end_time = _dtstamp_to_day_time(dtend)
                time_str = f"{start_time} - {end_time}" if end_time else start_time

                full_text = (
                    f"Studiengang: {program_name} | Tag: {day} | "
                    f"Zeit: {time_str} | Raum: {location} | Veranstaltung: {summary}"
                )
                if description:
                    full_text += f" | {description[:200]}"

                ical_url = f"{base_url}/ical?lan=de&puid=50&type=pg&pgid={pg_id}"
                documents.append({
                    "url": ical_url,
                    "title": f"{program_name} – {summary}",
                    "content": full_text,
                    "sections": [],
                    "h1": program_name,
                    "type": "timetable",
                    "metadata": {
                        "program": program_name,
                        "day": day,
                        "time": time_str,
                        "room": location,
                        "lecture_info": summary,
                        "source": "starplan_timetable",
                    },
                })
            elif in_event and ":" in line:
                key, _, val = line.partition(":")
                # Strip TZID params: DTSTART;TZID=Europe/Berlin → store as DTSTART
                clean_key = key.split(";")[0]
                event[clean_key] = val

        return documents

    def load(self) -> list[dict]:
        log.info("StarplanLoader: loading timetables from %s (pu=%d)", self.base_url, self.pu_id)
        org_groups = self._get_org_groups()
        log.info("  Found %d org groups", len(org_groups))

        all_documents: list[dict] = []

        for og in org_groups:
            og_id = og.get("id") or og.get("ogid")
            og_name = og.get("name", str(og_id))
            if og_id is None:
                continue

            planning_groups = self._get_planning_groups(og_id)
            for pg in planning_groups:
                pg_id = pg.get("id") or pg.get("pgid")
                pg_name = pg.get("name", og_name)
                if pg_id is None:
                    continue

                ical = self._get_ical(pg_id)
                if not ical:
                    continue

                docs = self._parse_ical(ical, pg_name, pg_id, self.base_url)
                all_documents.extend(docs)
                log.debug("  %s: %d events", pg_name, len(docs))
                time.sleep(0.2)

        log.info("StarplanLoader: loaded %d timetable entries", len(all_documents))
        return all_documents


# ---------------------------------------------------------------------------
# FolderLoader
# ---------------------------------------------------------------------------

class FolderLoader:
    """
    Loads all matching files from a local directory.

    Config keys:
      path        — directory path
      extensions  — list of file extensions to include (default: all supported)
      recursive   — scan subdirectories (default: True)
    """

    SUPPORTED = {".txt", ".md", ".pdf", ".docx"}

    def __init__(self, config: dict):
        self.path = Path(config["path"])
        self.extensions = set(config.get("extensions", list(self.SUPPORTED)))
        self.recursive = config.get("recursive", True)
        self.result_type = config.get("result_type", "document")

    @staticmethod
    def _table_to_markdown(table: list[list]) -> str:
        """Convert a pdfplumber table (list of rows) to a Markdown table string."""
        rows = [[str(cell).strip() if cell is not None else "" for cell in row] for row in table]
        if not rows:
            return ""
        header = rows[0]
        sep = ["---"] * len(header)
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        for row in rows[1:]:
            # Pad short rows to header width
            padded = row + [""] * (len(header) - len(row))
            lines.append("| " + " | ".join(padded) + " |")
        return "\n".join(lines)

    def _extract(self, file: Path) -> str | None:
        ext = file.suffix.lower()
        try:
            if ext in (".txt", ".md"):
                return file.read_text(encoding="utf-8", errors="ignore")
            if ext == ".pdf":
                import pdfplumber
                page_parts: list[str] = []
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        # Extract tables first; replace their bounding boxes so
                        # extract_text() doesn't double-extract their content.
                        tables = page.extract_tables()
                        table_bboxes = [tbl.bbox for tbl in page.find_tables()] if tables else []

                        parts: list[str] = []

                        # Plain text — crop away table regions to avoid duplication
                        text_page = page
                        for bbox in table_bboxes:
                            try:
                                text_page = text_page.filter(
                                    lambda obj, b=bbox: not (
                                        b[0] <= obj.get("x0", 0) <= b[2]
                                        and b[1] <= obj.get("top", 0) <= b[3]
                                    )
                                )
                            except Exception:
                                pass  # filter may fail on some objects; ignore

                        plain = text_page.extract_text()
                        if plain and plain.strip():
                            parts.append(plain.strip())

                        # Markdown tables
                        for tbl in tables:
                            md = self._table_to_markdown(tbl)
                            if md:
                                parts.append(md)

                        if parts:
                            page_parts.append("\n\n".join(parts))

                return "\n\n---\n\n".join(page_parts) or None
            if ext == ".docx":
                import docx
                doc = docx.Document(file)
                return "\n".join(p.text for p in doc.paragraphs).strip() or None
        except Exception as e:
            log.warning("Could not extract %s: %s", file, e)
        return None

    def load(self) -> list[dict]:
        log.info("FolderLoader: scanning %s", self.path)
        glob = self.path.rglob if self.recursive else self.path.glob
        documents = []
        for ext in self.extensions:
            for file in glob(f"*{ext}"):
                text = self._extract(file)
                if text and len(text) > 50:
                    documents.append({
                        "url": file.as_uri(),
                        "title": file.stem.replace("_", " ").replace("-", " "),
                        "content": text,
                        "sections": [],
                        "h1": "",
                        "type": self.result_type,
                        "metadata": {"filename": file.name, "extension": ext},
                    })
        log.info("FolderLoader: loaded %d documents from %s", len(documents), self.path)
        return documents


# ---------------------------------------------------------------------------
# UrlListLoader
# ---------------------------------------------------------------------------

class UrlListLoader:
    """
    Scrapes an explicit list of URLs (no recursive crawling).

    Config keys:
      list   — list of URL strings
    """

    def __init__(self, config: dict):
        self.urls: list[str] = config.get("list", [])
        self.result_type = config.get("result_type", "webpage")

    def load(self) -> list[dict]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("Install beautifulsoup4: pip install beautifulsoup4 lxml")

        documents = []
        for url in self.urls:
            html = _get(url)
            if not html:
                continue
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")
            title, h1, content, sections = _parse_html_sections(soup)
            if content and len(content) > 50:
                documents.append({
                    "url": url,
                    "title": title,
                    "h1": h1,
                    "content": content,
                    "sections": sections,
                    "type": self.result_type,
                    "metadata": {},
                })
            time.sleep(0.3)

        log.info("UrlListLoader: scraped %d/%d URLs", len(documents), len(self.urls))
        return documents


# ---------------------------------------------------------------------------
# Loader factory
# ---------------------------------------------------------------------------

LOADER_TYPES: dict[str, type] = {
    "website": WebsiteLoader,
    "starplan": StarplanLoader,
    "folder": FolderLoader,
    "urls": UrlListLoader,
}


def get_loader(source_config: dict):
    """Return the right loader instance for a source config dict."""
    source_type = source_config.get("type")
    loader_class = LOADER_TYPES.get(source_type)
    if loader_class is None:
        raise ValueError(
            f"Unknown source type: '{source_type}'. "
            f"Available: {list(LOADER_TYPES.keys())}"
        )
    return loader_class(source_config)

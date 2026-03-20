import json
import re
import time

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Sitemaps provided by the user
SITEMAPS = [
    "https://www.hs-aalen.de/sitemap.xml?sitemap=news&cHash=9328a3bb7b686fd62683a27aa38768ed",
    "https://www.hs-aalen.de/sitemap.xml?sitemap=pages&cHash=067b9fa27c3c31031830cfd9d62c4858",
    "https://www.hs-aalen.de/sitemap.xml?sitemap=event&cHash=ab7c558440b5e2921e559a5daa55cab0",
]
ROOT_SITEMAP_INDEX = "https://www.hs-aalen.de/sitemap.xml"

OUTPUT_FILE = "/Users/merluee/Desktop/suchemacschine/data.jsonl"
REPORT_FILE = "/Users/merluee/Desktop/suchemacschine/scrape_report.json"


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


def extract_content(session, url):
    print(f"Scraping content from: {url}")
    try:
        response = session.get(url, timeout=(10, 35))
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove obvious non-content elements site-wide first
        for element in soup(["script", "style", "nav", "footer", "form", "aside", "noscript"]):
            element.decompose()

        # Find the most likely "main" container
        # We look for <main>, article tags, or IDs like 'content', 'main'
        main_container = soup.find("main") or soup.find("article") or soup.find(id=re.compile(r"content|main", re.I))

        if not main_container:
            # Fallback to body but with heavy cleaning
            main_container = soup.body

        if main_container:
            # Remove anything that looks like a menu, breadcrumb, or share widget
            exclude_patterns = r"breadcrumb|share|social|nav|menu|pagination|button|modal"
            for element in main_container.find_all(class_=re.compile(exclude_patterns, re.I)):
                element.decompose()

            # Extract plain text with a separator to avoid word merging
            text = main_container.get_text(separator=" ", strip=True)
        else:
            text = ""

        # Final text cleaning: remove multiple spaces, newlines, and non-printable characters
        text = re.sub(r"\s+", " ", text).strip()

        return text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


def main():
    session = create_session()

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

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, url in enumerate(unique_urls):
            content = extract_content(session, url)
            if content:
                # Store as a JSON object on each line
                record = {"url": url, "content": content}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                saved_records += 1
            elif content == "":
                empty_content_urls.append(url)
            else:
                failed_urls.append(url)

            # Rate limiting / polite scraping
            if i % 10 == 0:
                print(f"Progress: {i}/{len(unique_urls)}")

            # Small delay to be nice to the server
            time.sleep(0.1)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sitemaps_used": sitemap_list,
        "total_urls_from_sitemaps": len(all_urls),
        "unique_urls": len(unique_urls),
        "saved_records": saved_records,
        "failed_count": len(failed_urls),
        "empty_content_count": len(empty_content_urls),
        "failed_urls": failed_urls,
        "empty_content_urls": empty_content_urls,
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)

    print(f"Scraping complete. Results saved to {OUTPUT_FILE}")
    print(f"Scrape report written to {REPORT_FILE}")


if __name__ == "__main__":
    main()

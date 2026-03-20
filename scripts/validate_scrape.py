import json
from collections import Counter
from datetime import datetime

import requests
from bs4 import BeautifulSoup

ROOT_SITEMAP_INDEX = "https://www.hs-aalen.de/sitemap.xml"
DATA_FILE = "/Users/merluee/Desktop/suchemacschine/data.jsonl"
OUTPUT_REPORT = "/Users/merluee/Desktop/suchemacschine/validation_report.json"


def fetch_sitemap_urls() -> list[str]:
    response = requests.get(ROOT_SITEMAP_INDEX, timeout=(10, 30))
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "xml")
    sitemap_urls = [loc.text.strip() for loc in soup.find_all("loc") if loc.text]

    all_urls = []
    for sitemap_url in sitemap_urls:
        sitemap_response = requests.get(sitemap_url, timeout=(10, 30))
        sitemap_response.raise_for_status()
        sitemap_soup = BeautifulSoup(sitemap_response.content, "xml")
        all_urls.extend([loc.text.strip() for loc in sitemap_soup.find_all("loc") if loc.text])

    return sorted(set(all_urls))


def load_data_records() -> list[dict]:
    records = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main():
    sitemap_urls = fetch_sitemap_urls()
    records = load_data_records()

    data_urls = [r.get("url", "").strip() for r in records if r.get("url")]
    data_url_set = set(data_urls)
    sitemap_url_set = set(sitemap_urls)

    missing_in_data = sorted(sitemap_url_set - data_url_set)
    extra_in_data = sorted(data_url_set - sitemap_url_set)

    url_counts = Counter(data_urls)
    duplicate_urls = sorted([url for url, c in url_counts.items() if c > 1])

    empty_content_urls = sorted(
        [r.get("url", "") for r in records if not (r.get("content") or "").strip()]
    )

    report = {
        "timestamp": datetime.now().isoformat(),
        "sitemap_url_count": len(sitemap_urls),
        "data_record_count": len(records),
        "unique_data_url_count": len(data_url_set),
        "missing_in_data_count": len(missing_in_data),
        "extra_in_data_count": len(extra_in_data),
        "duplicate_url_count": len(duplicate_urls),
        "empty_content_count": len(empty_content_urls),
        "missing_in_data": missing_in_data,
        "extra_in_data": extra_in_data,
        "duplicate_urls": duplicate_urls,
        "empty_content_urls": empty_content_urls,
    }

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Validation complete")
    print(f"Sitemap URLs: {report['sitemap_url_count']}")
    print(f"Data records: {report['data_record_count']}")
    print(f"Unique data URLs: {report['unique_data_url_count']}")
    print(f"Missing in data: {report['missing_in_data_count']}")
    print(f"Extra in data: {report['extra_in_data_count']}")
    print(f"Duplicate URLs: {report['duplicate_url_count']}")
    print(f"Empty content: {report['empty_content_count']}")
    print(f"Report: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()

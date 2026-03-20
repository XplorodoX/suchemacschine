import requests
from bs4 import BeautifulSoup
import json
import re
import time
from urllib.parse import urljoin

# Sitemaps provided by the user
SITEMAPS = [
    "https://www.hs-aalen.de/sitemap.xml?sitemap=news&cHash=9328a3bb7b686fd62683a27aa38768ed",
    "https://www.hs-aalen.de/sitemap.xml?sitemap=pages&cHash=067b9fa27c3c31031830cfd9d62c4858",
    "https://www.hs-aalen.de/sitemap.xml?sitemap=event&cHash=ab7c558440b5e2921e559a5daa55cab0"
]

OUTPUT_FILE = "/Users/merluee/Desktop/suchemacschine/data.jsonl"

def get_urls_from_sitemap(sitemap_url):
    print(f"Fetching sitemap: {sitemap_url}")
    try:
        response = requests.get(sitemap_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'xml')
        urls = [loc.text for loc in soup.find_all('loc')]
        print(f"Found {len(urls)} URLs in {sitemap_url}")
        return urls
    except Exception as e:
        print(f"Error fetching sitemap {sitemap_url}: {e}")
        return []

def extract_content(url):
    print(f"Scraping content from: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove obvious non-content elements site-wide first
        for element in soup(["script", "style", "nav", "footer", "header", "form", "aside", "noscript"]):
            element.decompose()

        # Find the most likely "main" container
        # We look for <main>, article tags, or IDs like 'content', 'main'
        main_container = soup.find('main') or soup.find('article') or soup.find(id=re.compile(r'content|main', re.I))
        
        if not main_container:
            # Fallback to body but with heavy cleaning
            main_container = soup.body

        if main_container:
            # Clean specifically within the main container
            # Remove anything that looks like a menu, breadcrumb, or share widget
            for element in main_container.find_all(class_=re.compile(r'breadcrumb|share|social|nav|menu|pagination|button|modal', re.I)):
                element.decompose()
            
            # Extract plain text with a separator to avoid word merging
            text = main_container.get_text(separator=' ', strip=True)
        else:
            text = ""

        # Final text cleaning: remove multiple spaces, newlines, and non-printable characters
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def main():
    all_urls = []
    for sitemap in SITEMAPS:
        all_urls.extend(get_urls_from_sitemap(sitemap))
    
    # Remove duplicates
    unique_urls = list(set(all_urls))
    print(f"Total unique URLs to scrape: {len(unique_urls)}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for i, url in enumerate(unique_urls):
            content = extract_content(url)
            if content:
                # Store as a JSON object on each line
                record = {
                    "url": url,
                    "content": content
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            # Rate limiting / polite scraping
            if i % 10 == 0:
                print(f"Progress: {i}/{len(unique_urls)}")
            
            # Small delay to be nice to the server
            time.sleep(0.1)

    print(f"Scraping complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

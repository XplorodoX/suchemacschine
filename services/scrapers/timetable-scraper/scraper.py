#!/usr/bin/env python3
"""
Timetable Scraper Service
Scrapt Stundenpläne von HS Aalen Hochschulportal
"""
import os
import json
from datetime import datetime
from typing import List, Dict

import requests

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/data")
API_URL = os.getenv("API_URL", "https://hochschulportal.hs-aalen.de")

def scrape_timetables():
    """Scrape timetable data from HS Aalen."""
    print(f"📅 Starting timetable scraper at {datetime.now()}")
    print(f"   API URL: {API_URL}")
    
    results = []
    
    # Common degree programs at HS Aalen
    programs = [
        "INF",  # Informatik
        "WIN",  # Wirtschaftsinformatik
        "ET",   # Elektrotechnik
        "MB",   # Maschinenbau
    ]
    
    # Semesters
    semesters = ["1", "2", "3", "4", "5", "6", "7"]
    
    for program in programs:
        for semester in semesters:
            try:
                # Try to fetch from common timetable endpoints
                url = f"{API_URL}/api/stundenplan/{program}/{semester}"
                print(f"   Fetching: {program}/{semester}...", end=" ", flush=True)
                
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    events = data.get("events", [])
                    if events:
                        results.append({
                            "program": program,
                            "semester": semester,
                            "events": events,
                            "scraped_at": datetime.now().isoformat(),
                            "source": "timetable",
                            "url": url
                        })
                        print(f"✓ ({len(events)} events)")
                    else:
                        print("- (empty)")
                else:
                    print(f"✗ (HTTP {response.status_code})")
            
            except Exception as e:
                print(f"✗ (Error: {e})")
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "timetable_data.jsonl")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Scraped {len(results)} timetables")
    print(f"   Saved to: {output_file}")

if __name__ == "__main__":
    scrape_timetables()

#!/usr/bin/env python3
"""Quick test of Starplan iCal URLs"""

import requests
import re

base_url = "https://vorlesungen.htw-aalen.de/splan"

# First, get the program list
print("Fetching program list...")
r = requests.get(f"{base_url}/mobile?lan=de&sel=pg&pu=50", timeout=10)
r.encoding = 'ISO-8859-1'

# Extract one program ID for testing
prog_match = re.search(r'value="(\d+)"[^>]*>([^<]+)<', r.text)
if prog_match:
    prog_id = prog_match.group(1)
    prog_name = prog_match.group(2).strip()
    
    print(f"\nTesting iCal export for: {prog_name} (ID: {prog_id})\n")
    
    # Test different URL patterns
    urls_to_test = [
        f"{base_url}/export?type=ical&sel=pg&og={prog_id}&pu=50",
        f"{base_url}/mobile?type=ical&sel=pg&og={prog_id}&pu=50",
        f"{base_url}?export=ical&og={prog_id}&pu=50",
        f"{base_url}/export/ics?og={prog_id}&pu=50",
    ]
    
    for i, url in enumerate(urls_to_test, 1):
        print(f"{i}. Testing: {url}")
        try:
            response = requests.get(url, timeout=10)
            print(f"   Status: {response.status_code}")
            print(f"   Content-Type: {response.headers.get('content-type', 'N/A')}")
            print(f"   Size: {len(response.content)} bytes")
            
            # Check if it's valid iCal
            try:
                content = response.text if isinstance(response.content, bytes) else response.content
                if 'BEGIN:VCALENDAR' in content:
                    print(f"   ✓ Valid iCal format!")
                    
                    # Count events
                    event_count = content.count('BEGIN:VEVENT')
                    print(f"   ✓ Contains {event_count} events")
                    
                    # Show sample
                    lines = content.split('\n')
                    for line in lines[:15]:
                        print(f"   {line}")
                    break
                else:
                    print(f"   ✗ Not valid iCal")
            except Exception as e:
                print(f"   ✗ Error parsing: {e}")
        
        except Exception as e:
            print(f"   ✗ Error: {e}")
        
        print()
else:
    print("Could not find program IDs in HTML")

#!/usr/bin/env python3
"""Test Starplan API endpoints to understand the data structure"""

import json

import requests


def test_starplan_endpoints():
    base_url = "https://vorlesungen.htw-aalen.de/splan"
    
    # Probiere bekannte Starplan-API-Patterns
    endpoints = [
        ("/data/infos", "GET"),
        ("/data/events", "GET"),
        ("/export?type=json", "GET"),
        ("/getEvents", "GET"),
        ("/api/events", "GET"),
        ("/mobile?lan=de&export=json&sel=pg&pu=50", "GET"),
        ("/mobile/data", "POST"),
    ]
    
    print("=" * 70)
    print("Testing Starplan API Endpoints")
    print("=" * 70)
    
    for endpoint, method in endpoints:
        try:
            url = base_url + endpoint
            print(f"\n[{method}] {endpoint}")
            print(f"    Full URL: {url}")
            
            if method == "GET":
                r = requests.get(url, timeout=5)
            else:
                r = requests.post(url, timeout=5)
            
            print(f"    Status: {r.status_code}")
            print(f"    Content-Type: {r.headers.get('content-type', 'N/A')}")
            
            if r.status_code == 200:
                try:
                    data = r.json()
                    print("    ✓ JSON Response!")
                    print(f"    Keys: {list(data.keys())[:8]}")
                    print(f"    Sample size: {len(json.dumps(data))} bytes")
                except:
                    print("    ✓ 200 OK but not JSON")
                    print(f"    Response size: {len(r.text)} bytes")
                    if len(r.text) < 500:
                        print(f"    Content: {r.text[:200]}")
            
        except requests.exceptions.Timeout:
            print("    ✗ Timeout")
        except requests.exceptions.ConnectionError:
            print("    ✗ Connection error")
        except Exception as e:
            print(f"    ✗ Error: {str(e)[:60]}")

if __name__ == "__main__":
    test_starplan_endpoints()
    
    print("\n" + "=" * 70)
    print("Next: Try to discover what data parameters are available")
    print("=" * 70)
    
    # Versuche, die aktuelle Seite mit verschiedenen Parametern zu laden
    print("\nTrying the main endpoint with different selectors...")
    
    selectors = ["pg", "c", "r", "s"]  # program, course, room, staff
    for sel in selectors:
        try:
            url = f"https://vorlesungen.htw-aalen.de/splan/mobile?lan=de&sel={sel}&pu=50&export=json"
            r = requests.get(url, timeout=5)
            print(f"  sel={sel}: {r.status_code} - {r.headers.get('content-type', '')}")
        except Exception as e:
            print(f"  sel={sel}: Error - {str(e)[:40]}")

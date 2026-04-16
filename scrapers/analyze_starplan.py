#!/usr/bin/env python3
"""Parse Starplan HTML to extract timetable data"""

import json
import re

import requests
from bs4 import BeautifulSoup


def extract_timetable_data(url):
    """Extract schedule data from Starplan HTML"""
    try:
        r = requests.get(url, timeout=10)
        r.encoding = 'ISO-8859-1'
        soup = BeautifulSoup(r.text, 'html.parser')
        
        # Versuche die Tabelle zu finden
        tables = soup.find_all('table')
        print(f"Found {len(tables)} tables\n")
        
        # Suche nach select/dropdown-Elementen für verfügbare Optionen
        selects = soup.find_all('select')
        print(f"Found {len(selects)} select elements:")
        for select in selects[:5]:
            print(f"  - {select.get('name')}: {[opt.text for opt in select.find_all('option')[:5]]}")
        
        # Suche nach Events/Einträgen
        
        # Pattern 1: Suche nach Tabellenzellen mit Zeitinformationen  
        cells = soup.find_all(['td', 'div'], class_=re.compile(r'(event|lec|course|class)', re.I))
        print(f"\nFound {len(cells)} potential event cells")
        if cells:
            for cell in cells[:3]:
                print(f"  Sample: {str(cell)[:200]}")
        
        # Pattern 2: Suche nach Text-Patterns für Vorlesungsinfos
        # "8:00" - "10:00" oder ähnlich
        time_pattern = re.compile(r'\b\d{1,2}:\d{2}\b')
        day_pattern = re.compile(r'\b(Montag|Dienstag|Mittwoch|Donnerstag|Freitag)\b', re.IGNORECASE)
        
        text = soup.get_text()
        times = time_pattern.findall(text)
        days = day_pattern.findall(text)
        
        print(f"\nTimes found: {set(times)}")
        print(f"Days found: {set(days)}")
        
        # Versuche, aus dem JavaScript die Event-Daten zu extrahieren
        scripts = soup.find_all('script', type='text/javascript')
        print(f"\nFound {len(scripts)} JavaScript blocks")
        
        for i, script in enumerate(scripts[:2]):
            if script.string:
                # Suche nach JSON-Objekten in JavaScript
                json_pattern = re.compile(r'\{[^{}]*?"(name|title|start|end|time)"[^{}]*?\}', re.IGNORECASE)
                matches = json_pattern.findall(script.string)
                if matches:
                    print(f"  Script {i}: Found potential JSON data")
        
        # Versuche, alle Links zu extrahieren (könnten zu spezifischen Vorlesungen führen)
        links = soup.find_all('a', href=re.compile(r'sel='))
        print(f"\nFound {len(links)} data-selection links")
        for link in links[:5]:
            href = link.get('href', '')
            text = link.get_text(strip=True)
            if text and len(text) < 100:
                print(f"  {text}: {href[:80]}")
        
        return {
            'tables_count': len(tables),
            'selects': [s.get('name') for s in selects],
            'events_found': len(cells),
            'times': list(set(times)),
            'days': list(set(days))
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {}

if __name__ == "__main__":
    print("=" * 70)
    print("Analyzing Starplan HTML Structure")
    print("=" * 70)
    
    # Analysiere verschiedene Selektor-Typen
    selectors = [
        ("Programs", "pg", "Studiengänge"),
        ("Courses", "c", "Vorlesungen"),  
        ("Rooms", "r", "Räume"),
        ("Staff", "s", "Dozenten"),
    ]
    
    for label, sel, german in selectors:
        print(f"\n{'='*70}")
        print(f"Analyzing {label} ({german}) - sel={sel}")
        print(f"{'='*70}")
        
        url = f"https://vorlesungen.htw-aalen.de/splan/mobile?lan=de&sel={sel}&pu=50"
        result = extract_timetable_data(url)
        print(f"\nResult: {json.dumps(result, ensure_ascii=False, indent=2)}")

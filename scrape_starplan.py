#!/usr/bin/env python3
"""Extract timetable data from Starplan"""

import requests
from bs4 import BeautifulSoup
import json
import re
from urllib.parse import urljoin

def extract_starplan_data(url):
    """Extract structured timetable data from Starplan"""
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        r = requests.get(url, headers=headers, timeout=10)
        r.encoding = 'ISO-8859-1'
        soup = BeautifulSoup(r.text, 'html.parser')
        
        data = {
            'url': url,
            'title': soup.title.string if soup.title else 'Starplan',
            'programs': [],
            'courses': [],
            'rooms': [],
            'staff': [],
            'timetable': {}
        }
        
        # Extrahiere alle Select-Elemente
        # og = Organisationsgruppe (Studiengänge)
        og_select = soup.find('select', {'name': 'og'})
        if og_select:
            options = og_select.find_all('option')
            for option in options:
                if option.get('value', '').strip() != '-1':
                    data['programs'].append({
                        'name': option.text.strip(),
                        'id': option.get('value', ''),
                        'url': f"https://vorlesungen.htw-aalen.de/splan/mobile?lan=de&sel=pg&og={option.get('value')}&pu=50"
                    })
        
        # Extrahiere Tabellendaten für Stundenplan
        tables = soup.find_all('table')
        for table_idx, table in enumerate(tables):
            rows = table.find_all('tr')
            
            if len(rows) > 0:
                # Erste Reihe ist normalerweise Header mit Zeiten
                header_row = rows[0]
                headers = [cell.text.strip() for cell in header_row.find_all(['th', 'td'])]
                
                data['timetable'][f'table_{table_idx}'] = {
                    'headers': headers,
                    'rows': []
                }
                
                # Extrahiere Datenreihen
                for row_idx, row in enumerate(rows[1:]):
                    cells = row.find_all(['td', 'th'])
                    row_data = [cell.text.strip() for cell in cells]
                    data['timetable'][f'table_{table_idx}']['rows'].append(row_data)
        
        # Suche nach Event-Divs (Vorlesungen)
        event_pattern = re.compile(r'(event|lec|course|event_|lv_)', re.IGNORECASE)
        event_divs = soup.find_all('div', class_=event_pattern)
        
        for div in event_divs[:10]:
            text = div.get_text(strip=True)
            if text:
                # Versuche, Informationen zu extrahieren
                lines = [l.strip() for l in text.split('\n') if l.strip()]
                if lines:
                    data['courses'].append({
                        'raw_text': text[:200],
                        'lines': lines[:5]
                    })
        
        # Suche nach Links zu Räumen oder Dozenten
        room_links = soup.find_all('a', href=re.compile(r'sel=r'))
        course_links = soup.find_all('a', href=re.compile(r'sel=c'))
        staff_links = soup.find_all('a', href=re.compile(r'sel=s'))
        
        return data
        
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    print("=" * 70)
    print("Extracting Starplan Data")
    print("=" * 70)
    
    # Teste verschiedene Views
    urls = [
        ("Default view", "https://vorlesungen.htw-aalen.de/splan/mobile?lan=de&act=tt&sel=pg"),
        ("Sommersemester 2026", "https://vorlesungen.htw-aalen.de/splan/mobile?lan=de&sel=pg&pu=50"),
        ("Mit Zeitangaben", "https://vorlesungen.htw-aalen.de/splan/mobile?lan=de&act=tt&sel=pg&pu=50&sd=true&dfc=2026-03-22"),
    ]
    
    for label, url in urls:
        print(f"\n{'='*70}")
        print(f"{label}")
        print(f"{'='*70}")
        
        data = extract_starplan_data(url)
        
        print(f"\nPrograms found: {len(data.get('programs', []))}")
        if data.get('programs'):
            for prog in data['programs'][:5]:
                print(f"  - {prog['name']} (ID: {prog['id']})")
        
        print(f"\nTimetable tables: {len(data.get('timetable', {}))}")
        for table_key, table_data in list(data.get('timetable', {}).items())[:2]:
            print(f"  {table_key}:")
            print(f"    Headers: {table_data['headers'][:5]}")
            print(f"    Rows: {len(table_data['rows'])}")
            if table_data['rows']:
                print(f"    Sample row: {table_data['rows'][0][:5]}")
        
        print(f"\nCourses/Events found: {len(data.get('courses', []))}")
        if data.get('courses'):
            for course in data['courses'][:3]:
                print(f"  - {course.get('raw_text', '')[:80]}")
        
        # Speichere Ergebnis als JSON
        with open(f'/tmp/starplan_{url.split("=")[-1][:10]}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"\n✓ Saved to /tmp/starplan_*.json")

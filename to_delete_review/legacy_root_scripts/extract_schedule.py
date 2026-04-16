#!/usr/bin/env python3
"""Extract actual timetable data after selecting a program"""

import asyncio
from playwright.async_api import async_playwright
import json
import re

async def extract_schedule():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        await page.set_viewport_size({"width": 375, "height": 812})
        
        print("=" * 70)
        print("Loading Starplan with selected program")
        print("=" * 70)
        
        # URL mit ausgewähltem Studiengang (AI = Artifical Intelligence / Informatik)
        # Versuche mit Programm-Parameter
        url = "https://vorlesungen.htw-aalen.de/splan/mobile?lan=de&act=tt&sel=pg&pu=50&sd=true&dfc=2026-03-22&loc=1&sa=false&cb=o"
        
        print(f"Loading: {url}\n")
        await page.goto(url, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(2000)
        
        # Versuche, List-Items oder Tabellen-Zeilen zu finden
        # Starplan zeigt normalerweise eine Liste oder Tabelle von Studiengängen
        
        list_items = await page.locator('li, tr, .event, [data-id]').all()
        print(f"List/Row items found: {len(list_items)}\n")
        
        # Versuche, auf das erste Element zu klicken (sollte ein Studiengang sein)
        try:
            # Suche nach einem Button oder Link mit Text, die einen Studiengang zu sein scheinen
            links = await page.locator('a[href*="og="]').all()
            print(f"Program links found: {len(links)}")
            
            if links:
                for i, link in enumerate(links[:5]):
                    text = await link.text_content()
                    href = await link.get_attribute('href')
                    print(f"  [{i}] {text.strip()[:50]} → {href[:80]}")
                    
                    # Klicke auf den ersten Studiengang (z.B. "AI" - Informatik)
                    if i == 0 and "og=" in href:
                        print(f"\nClicking on: {text.strip()}")
                        await link.click()
                        await page.wait_for_timeout(2000)
                        break
        except Exception as e:
            print(f"Error clicking: {e}")
        
        # Jetzt sollte die Schedule visible sein
        # Suche nach Zeilen mit Zeitangaben
        schedule_content = await page.query_selector_all('*')
        
        # Versuche, die HTML als Text zu bekommen
        html = await page.content()
        
        # Extrahiere Zeitangaben und Vorlesungs-Infos
        text_content = await page.evaluate('() => document.body.innerText')
        
        lines = text_content.split('\n')
        schedule_lines = []
        
        for i, line in enumerate(lines):
            # Suche nach Zeitangaben (8:00, 9:00, etc.) oder Wochentagen
            if re.search(r'\d{1,2}:\d{2}', line) or re.search(r'(Montag|Dienstag|Mittwoch|Donnerstag|Freitag)', line, re.I):
                schedule_lines.append((i, line.strip()))
        
        print(f"\nSchedule lines found: {len(schedule_lines)}")
        for idx, line in schedule_lines[:20]:
            print(f"  {line[:100]}")
        
        # Versuche, alle Texte zu tabelle zu finden
        tables = await page.locator('table').all()
        print(f"\nTables found: {len(tables)}")
        
        for i, table in enumerate(tables[:3]):
            rows = await table.locator('tr').all()
            print(f"  Table {i}: {len(rows)} rows")
            
            # Zeige die erste Zeile
            if rows:
                cells = await rows[0].locator('td, th').all()
                cell_texts = []
                for cell in cells[:8]:
                    text = await cell.text_content()
                    cell_texts.append(text.strip()[:30])
                print(f"    Header: {cell_texts}")
        
        # Versuche auch, Daten aus data-Attributen zu extrahieren
        data_elements = await page.locator('[data-*]').all()
        print(f"\nElements with data-* attributes: {len(data_elements)}")
        
        if data_elements:
            for i, elem in enumerate(data_elements[:3]):
                attrs = await elem.evaluate('el => Object.entries(el.dataset)')
                if attrs:
                    print(f"  [{i}] {attrs}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(extract_schedule())

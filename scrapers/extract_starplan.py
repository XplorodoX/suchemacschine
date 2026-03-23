#!/usr/bin/env python3
"""Use Playwright to analyze Starplan dynamic content"""

import asyncio
import re

from playwright.async_api import async_playwright


async def extract_starplan_data():
    async with async_playwright() as p:
        # Starten Sie einen Browser
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Setzen Sie die Viewport-Größe (die Seite wird als "mobilé" angefordert)
        await page.set_viewport_size({"width": 375, "height": 812})
        
        print("=" * 70)
        print("Analyzing Starplan with Playwright (JavaScript rendering)")
        print("=" * 70)
        
        selectors = [
            ("Studiengänge", "pg"),
            ("Kurse", "c"),
            ("Räume", "r"),
            ("Dozenten", "s"),
        ]
        
        for label, sel in selectors:
            print(f"\n{'='*70}")
            print(f"Fetching: {label} (sel={sel})")
            print(f"{'='*70}")
            
            url = f"https://vorlesungen.htw-aalen.de/splan/mobile?lan=de&sel={sel}&pu=50"
            print(f"URL: {url}\n")
            
            try:
                # Navigiere zur URL
                await page.goto(url, wait_until="networkidle", timeout=30000)
                
                # Warte kurz, damit alle JavaScripts laden
                await page.wait_for_timeout(2000)
                
                # Extrahiere die Seiteninhalte
                await page.content()
                
                # Extrahiere Tabellendaten
                # Suche nach divs mit Event-Daten
                event_elements = await page.locator('[class*="event"], [class*="lec"], [class*="course"]').all()
                print(f"Event elements found: {len(event_elements)}")
                
                if event_elements:
                    for i, elem in enumerate(event_elements[:3]):
                        text = await elem.text_content()
                        print(f"  [{i}] {text[:100]}")
                
                # Suche nach Links in der Seite  
                links = await page.locator('a').all()
                print(f"Links found: {len(links)}")
                
                # Zeige die Dropdown-Optionen
                selects = await page.locator('select').all()
                print(f"Selectors found: {len(selects)}")
                
                for i, select_elem in enumerate(selects[:2]):
                    name = await select_elem.get_attribute('name')
                    options = await select_elem.locator('option').all()
                    print(f"  [{name}]: {len(options)} options")
                    for j, option in enumerate(options[:5]):
                        text = await option.text_content()
                        value = await option.get_attribute('value')
                        print(f"      [{j}] {text} (value={value})")
                
                # Versuche, JSON-Daten aus dem Document zu extrahieren
                json_search = await page.evaluate("""
                    () => {
                        const scripts = document.querySelectorAll('script');
                        let found = [];
                        for (let script of scripts) {
                            if (script.textContent && script.textContent.includes('data') && script.textContent.includes('{')) {
                                // Suche nach JSON-Pattern
                                const match = script.textContent.match(/\\{[^{}]{10,}\\}/);
                                if (match) {
                                    found.push({
                                        type: 'script',
                                        preview: match[0].substring(0, 200)
                                    });
                                }
                            }
                        }
                        return found;
                    }
                """)
                
                if json_search:
                    print(f"\nJSON data in scripts: {len(json_search)}")
                    for j, item in enumerate(json_search[:2]):
                        print(f"  [{j}] {item['preview'][:150]}")
                
                # Hole den vollständigen gerenderten HTML
                html = await page.content()
                print(f"\nPage size: {len(html)} bytes")
                
                # Extrahiere Text-Pattern
                text = await page.text_content()
                times = re.findall(r'\b\d{1,2}:\d{2}\b', text)
                if times:
                    print(f"Times found: {set(times)}")
                
                days_de = re.findall(r'\b(Montag|Dienstag|Mittwoch|Donnerstag|Freitag)\b', text, re.IGNORECASE)
                if days_de:
                    print(f"Weekdays found: {set(days_de)}")
                
            except Exception as e:
                print(f"Error: {e}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(extract_starplan_data())

#!/usr/bin/env python3
"""Find actual iCal download mechanism using Playwright"""

import asyncio
import re

from playwright.async_api import async_playwright


async def find_ical_download():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        await page.set_viewport_size({"width": 375, "height": 812})
        
        print("=" * 70)
        print("Finding iCal Download Mechanism")
        print("=" * 70)
        
        # Load a program's timetable
        url = "https://vorlesungen.htw-aalen.de/splan/mobile?lan=de&sel=pg&og=1630&pu=50&act=tt"
        
        print(f"\nLoading: {url}\n")
        await page.goto(url, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(1000)
        
        # Look for export/download buttons or links
        print("Looking for export/download elements...\n")
        
        # Find all buttons and links
        buttons = await page.locator('button, a, [role="button"]').all()
        print(f"Found {len(buttons)} buttons/links")
        
        # Filter for export/download related
        export_elements = []
        for btn in buttons:
            try:
                text = await btn.text_content()
                title = await btn.get_attribute('title')
                onclick = await btn.get_attribute('onclick')
                href = await btn.get_attribute('href')
                
                # Check if related to export
                if any(keyword in (text + title + onclick + href).lower() for keyword in ['export', 'download', 'ical', 'ics', 'calendar']):
                    export_elements.append({
                        'text': text,
                        'title': title,
                        'onclick': onclick,
                        'href': href,
                        'element': btn
                    })
            except:
                pass
        
        print(f"Found {len(export_elements)} export-related elements:\n")
        
        for i, elem in enumerate(export_elements, 1):
            print(f"{i}. Text: {elem['text']}")
            print(f"   Title: {elem['title']}")
            print(f"   Href: {elem['href']}")
            print(f"   Onclick: {elem['onclick']}")
            print()
        
        # Check for any iframes or hidden content
        iframes = await page.locator('iframe').all()
        print(f"\nIframes found: {len(iframes)}")
        
        # Look in page source for export URLs
        page_source = await page.content()
        
        # Search for common export URL patterns
        patterns = [
            r'export[^"\']*\.php[^"\']*',
            r'/splan/[^"\']*\.ics[^"\']*',
            r'download[^"\']*ics[^"\']*',
            r'type=ical[^"\']*',
        ]
        
        print("\nSearching page source for export patterns...")
        for pattern in patterns:
            matches = re.findall(pattern, page_source, re.IGNORECASE)
            if matches:
                print(f"\n  Pattern '{pattern}':")
                for match in matches[:3]:
                    print(f"    - {match[:100]}")
        
        # Try clicking the print/export button if it exists
        try:
            # Look for "Exportieren" button (German for "Export")
            export_btn = await page.locator('text=Exportieren').first
            if export_btn:
                await export_btn.scroll_into_view_if_needed()
                await export_btn.click()
                
                print("\nClicked 'Exportieren' button...")
                await page.wait_for_timeout(1000)
                
                # Check what happened (modal, menu, etc.)
                menu = await page.locator('.menu, .dropdown, [role="menu"]').first
                if menu:
                    print("Dropdown menu appeared!")
                    items = await menu.locator('[role="menuitem"], li, a').all()
                    print(f"Menu items ({len(items)}):")
                    for item in items[:10]:
                        text = await item.text_content()
                        print(f"  - {text}")
        except:
            pass
        
        # Also check network requests
        print("\n" + "="*70)
        print("Monitoring network requests for iCal downloads...")
        print("="*70)
        
        # Set up response listener
        responses_found = []
        
        def handle_response(response):
            if 'ical' in response.url.lower() or 'ics' in response.url.lower() or response.headers.get('content-type', '').lower().__contains__('calendar'):
                responses_found.append({
                    'url': response.url,
                    'status': response.status,
                    'content-type': response.headers.get('content-type', 'N/A')
                })
        
        page.on("response", handle_response)
        
        # Try navigating with different export parameters
        test_urls = [
            "https://vorlesungen.htw-aalen.de/splan/mobile?lan=de&sel=pg&og=1630&pu=50&act=tt&export=ical",
            "https://vorlesungen.htw-aalen.de/splan/mobile?lan=de&sel=pg&og=1630&pu=50&pdf=1",
        ]
        
        for test_url in test_urls:
            try:
                print(f"\nTrying: {test_url}")
                await page.goto(test_url, wait_until="networkidle", timeout=10000)
            except:
                pass
        
        print(f"\nResponses with 'ical' or 'calendar': {len(responses_found)}")
        for resp in responses_found:
            print(f"  - {resp['url']}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(find_ical_download())

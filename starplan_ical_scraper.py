#!/usr/bin/env python3
"""
Starplan iCal Scraper - Extract timetable data from iCal/ICS files

This scraper:
1. Fetches program list from Starplan
2. Extracts iCal download links for each program
3. Parses iCal events
4. Structures lecture information for indexing
"""

import asyncio
from playwright.async_api import async_playwright
import requests
import json
import re
from datetime import datetime
from icalendar import Calendar
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class StarplanIcalScraper:
    def __init__(self):
        self.base_url = "https://vorlesungen.htw-aalen.de/splan/mobile"
        self.data = {
            'programs': [],
            'lectures': [],
            'extracted_at': datetime.now().isoformat()
        }
    
    async def get_program_ical_links(self):
        """Extract iCal download links for all programs"""
        programs_with_links = []
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            await page.set_viewport_size({"width": 375, "height": 812})
            
            try:
                logger.info("Loading Starplan home page...")
                
                # Load main page
                url = f"{self.base_url}?lan=de&act=tt&sel=pg&pu=50"
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await page.wait_for_timeout(1000)
                
                # Extract programs
                logger.info("Extracting programs...")
                og_select = await page.locator('select[name="og"]').first.evaluate(
                    'el => Array.from(el.options).map(o => ({name: o.text, id: o.value}))'
                )
                
                for option in og_select:
                    if option['id'] and option['id'] != '-1':
                        program = {
                            'name': option['name'].strip(),
                            'id': option['id'],
                            'ical_url': None,
                            'type': 'program'
                        }
                        
                        # Navigate to program to find iCal link
                        try:
                            prog_url = f"{self.base_url}?lan=de&sel=pg&og={option['id']}&pu=50&act=tt"
                            await page.goto(prog_url, wait_until="networkidle", timeout=15000)
                            await page.wait_for_timeout(500)
                            
                            # Look for iCal/export links
                            # Common patterns: export?type=ical, .ics, ical.php, etc.
                            ical_link = None
                            
                            # Try direct links
                            links = await page.locator('a[href*="ics"], a[href*="ical"], a[href*="export"]').all()
                            
                            for link in links:
                                href = await link.get_attribute('href')
                                if href and ('ics' in href.lower() or 'ical' in href.lower() or 'export' in href.lower()):
                                    # Convert relative URLs to absolute
                                    if href.startswith('/'):
                                        ical_link = 'https://vorlesungen.htw-aalen.de' + href
                                    elif href.startswith('http'):
                                        ical_link = href
                                    else:
                                        ical_link = 'https://vorlesungen.htw-aalen.de/splan/' + href
                                    break
                            
                            # If no direct link found, try JavaScript method
                            if not ical_link:
                                # Most Starplan system have hidden export functions
                                # Try common URLs
                                test_urls = [
                                    f"https://vorlesungen.htw-aalen.de/splan/export?type=ical&sel=pg&og={option['id']}&pu=50",
                                    f"https://vorlesungen.htw-aalen.de/splan/mobile/export?og={option['id']}&pu=50&type=ical",
                                    f"https://vorlesungen.htw-aalen.de/splan?og={option['id']}&pu=50&type=ical",
                                ]
                                
                                for test_url in test_urls:
                                    try:
                                        r = requests.head(test_url, timeout=5)
                                        if r.status_code in [200, 303, 307]:
                                            ical_link = test_url
                                            break
                                    except:
                                        pass
                            
                            if ical_link:
                                program['ical_url'] = ical_link
                                logger.info(f"  ✓ {program['name']}: Found iCal link")
                            else:
                                logger.warning(f"  ⚠ {program['name']}: No iCal link found")
                        
                        except Exception as e:
                            logger.warning(f"  ✗ {program['name']}: {e}")
                        
                        programs_with_links.append(program)
                
                logger.info(f"Found {len(programs_with_links)} programs total")
                
            finally:
                await browser.close()
        
        return programs_with_links
    
    def parse_ical(self, ical_content, program_name):
        """Parse iCal content and extract lecture events"""
        lectures = []
        
        try:
            cal = Calendar.from_ical(ical_content)
            
            for component in cal.walk():
                if component.name == "VEVENT":
                    event = {
                        'program': program_name,
                        'title': None,
                        'start': None,
                        'end': None,
                        'location': None,
                        'instructor': None,
                        'description': None,
                        'raw': {}
                    }
                    
                    # Extract standard iCal properties
                    if 'summary' in component:
                        event['title'] = str(component['summary'])
                    
                    if 'dtstart' in component:
                        dt = component['dtstart'].dt
                        event['start'] = dt.isoformat() if hasattr(dt, 'isoformat') else str(dt)
                    
                    if 'dtend' in component:
                        dt = component['dtend'].dt
                        event['end'] = dt.isoformat() if hasattr(dt, 'isoformat') else str(dt)
                    
                    if 'location' in component:
                        event['location'] = str(component['location'])
                    
                    # Instructor might be in different fields depending on Starplan version
                    if 'organizer' in component:
                        event['instructor'] = str(component['organizer'])
                    elif 'description' in component and 'leiter' in str(component['description']).lower():
                        event['instructor'] = str(component['description'])
                    
                    if 'description' in component:
                        event['description'] = str(component['description'])
                    
                    # Store raw properties for debugging
                    for key in component:
                        if key not in ['summary', 'dtstart', 'dtend', 'location', 'organizer', 'description']:
                            event['raw'][key] = str(component[key])
                    
                    if event['title']:  # Only add if we have at least a title
                        lectures.append(event)
            
            logger.info(f"Parsed {len(lectures)} events from iCal")
            
        except Exception as e:
            logger.warning(f"Error parsing iCal: {e}")
        
        return lectures
    
    async def scrape(self):
        """Main scraping function"""
        # Step 1: Get iCal links
        programs = await self.get_program_ical_links()
        self.data['programs'] = programs
        
        # Step 2: Download and parse iCal files
        logger.info("\n" + "="*70)
        logger.info("Downloading and parsing iCal files...")
        logger.info("="*70)
        
        for program in programs[:20]:  # Limit to first 20 programs for performance
            if program['ical_url']:
                try:
                    logger.info(f"\nDownloading {program['name']} ({program['id']})...")
                    
                    r = requests.get(program['ical_url'], timeout=30)
                    r.encoding = 'utf-8'
                    
                    if r.status_code == 200:
                        lectures = self.parse_ical(r.content, program['name'])
                        self.data['lectures'].extend(lectures)
                        logger.info(f"  ✓ {len(lectures)} lectures added")
                    else:
                        logger.warning(f"  ✗ HTTP {r.status_code}")
                
                except Exception as e:
                    logger.warning(f"  ✗ Error: {str(e)[:80]}")
            else:
                logger.warning(f"  ⚠ No iCal URL available")
        
        return self.data
    
    def save(self, filename='starplan_ical_data.json'):
        """Save extracted data to JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        logger.info(f"Data saved to {filename}")

async def main():
    logger.info("=" * 70)
    logger.info("Starplan iCal Scraper")
    logger.info("=" * 70)
    
    scraper = StarplanIcalScraper()
    data = await scraper.scrape()
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)
    logger.info(f"Programs: {len(data['programs'])}")
    logger.info(f"Total lectures extracted: {len(data['lectures'])}")
    
    if data['lectures']:
        logger.info("\nSample lectures:")
        for lecture in data['lectures'][:5]:
            logger.info(f"  - {lecture['title']} ({lecture['program']})")
            logger.info(f"    Raum: {lecture['location']}")
            logger.info(f"    Zeit: {lecture['start']}")
    
    # Save data
    scraper.save()
    
    return data

if __name__ == "__main__":
    asyncio.run(main())

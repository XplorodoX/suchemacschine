#!/usr/bin/env python3
"""
Starplan Scraper - Extract timetable data from HTW Aalen Starplan system

This scraper:
1. Fetches the Starplan page
2. Extracts program/course information
3. Extracts timetable data
4. Indexes the data for search
"""

import asyncio
import json
import logging
import re
from datetime import datetime

from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class StarplanScraper:
    def __init__(self):
        self.base_url = "https://vorlesungen.htw-aalen.de/splan/mobile"
        self.data = {
            'programs': [],
            'timetables': {},
            'lectures': [],
            'extracted_at': datetime.now().isoformat()
        }
    
    async def scrape(self):
        """Main scraping function"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Set mobile viewport
            await page.set_viewport_size({"width": 375, "height": 812})
            
            try:
                logger.info("Loading Starplan home page...")
                
                # Load the main page
                url = f"{self.base_url}?lan=de&act=tt&sel=pg&pu=50&sd=true&dfc=2026-03-22&loc=1&sa=false&cb=o"
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await page.wait_for_timeout(1000)
                
                # Extract programs from the select dropdown
                logger.info("Extracting programs...")
                programs = await self.extract_programs(page)
                self.data['programs'] = programs
                logger.info(f"Found {len(programs)} programs")
                
                # For each program, extract timetable
                logger.info("Extracting timetables...")
                logger.info(f"📚 Processing all {len(programs)} programs (full scrape)...")
                for i, program in enumerate(programs, 1):  # All 89 programs
                    try:
                        timetable = await self.extract_timetable(page, program)
                        if timetable:
                            self.data['timetables'][program['id']] = timetable
                            logger.info(f"  [{i}/{len(programs)}] ✓ {program['name']}: {len(timetable.get('lectures', []))} lectures")
                    except Exception as e:
                        logger.warning(f"  [{i}/{len(programs)}] ✗ {program['name']}: {e}")
                
            finally:
                await browser.close()
            
            return self.data
    
    async def extract_programs(self, page):
        """Extract study programs from the page"""
        programs = []
        
        try:
            # Get the select element for organization groups
            og_select = await page.locator('select[name="og"]').first.evaluate(
                'el => Array.from(el.options).map(o => ({name: o.text, id: o.value}))'
            )
            
            for option in og_select:
                if option['id'] and option['id'] != '-1':
                    programs.append({
                        'name': option['name'].strip(),
                        'id': option['id'],
                        'type': 'program'
                    })
            
        except Exception as e:
            logger.warning(f"Could not extract programs: {e}")
            
            # Fallback: try to find links manually
            try:
                links = await page.locator('a[href*="og="]').all()
                for link in links[:20]:
                    href = await link.get_attribute('href')
                    text = await link.text_content()
                    
                    if href and 'og=' in href:
                        # Extract the og ID from URL
                        match = re.search(r'og=(\d+)', href)
                        if match:
                            programs.append({
                                'name': text.strip(),
                                'id': match.group(1),
                                'type': 'program',
                                'url': href
                            })
            except:
                pass
        
        return programs
    
    async def extract_timetable(self, page, program):
        """Extract timetable for a specific program"""
        timetable = {
            'program_id': program['id'],
            'program_name': program['name'],
            'days': {},
            'lectures': [],
            'schedule_table': None
        }
        
        try:
            # Navigate to the program's timetable
            url = f"{self.base_url}?lan=de&sel=pg&og={program['id']}&pu=50&act=tt"
            await page.goto(url, wait_until="networkidle", timeout=15000)
            await page.wait_for_timeout(1000)
            
            # Extract schedule table
            schedule_data = await page.evaluate("""
                () => {
                    const table = document.querySelector('table');
                    if (!table) return null;
                    
                    const rows = table.querySelectorAll('tr');
                    const data = {
                        headers: [],
                        rows: []
                    };
                    
                    // Get headers
                    if (rows.length > 0) {
                        const headerCells = rows[0].querySelectorAll('th, td');
                        for (let cell of headerCells) {
                            data.headers.push(cell.textContent.trim());
                        }
                    }
                    
                    // Get data rows
                    for (let i = 1; i < rows.length; i++) {
                        const cells = rows[i].querySelectorAll('td, th');
                        const rowData = [];
                        for (let cell of cells) {
                            rowData.push(cell.textContent.trim());
                        }
                        if (rowData.length > 0) {
                            data.rows.push(rowData);
                        }
                    }
                    
                    return data;
                }
            """)
            
            timetable['schedule_table'] = schedule_data
            
            # Try to extract lecture information from the page text
            text = await page.evaluate('() => document.body.innerText')
            
            # Parse lecture times and information
            lines = text.split('\n')
            current_day = None
            current_time = None
            
            for line in lines:
                line = line.strip()
                
                # Detect day
                if re.search(r'(Montag|Dienstag|Mittwoch|Donnerstag|Freitag)', line, re.I):
                    current_day = line
                
                # Detect time
                if re.match(r'^\d{1,2}:\d{2}', line):
                    current_time = line
                
                # Detect course/room info (usually indented or special format)
                if current_day and current_time and line and not re.match(r'^\d{1,2}:\d{2}', line):
                    # This might be a lecture
                    lecture = {
                        'day': current_day,
                        'time': current_time,
                        'info': line,
                        'program': program['name']
                    }
                    
                    # Only add non-empty lectures
                    if line not in ['Stundenpläne', 'Semester:', 'Auswahl:', 'Exportieren:', '']:
                        timetable['lectures'].append(lecture)
            
        except Exception as e:
            logger.warning(f"Error extracting timetable for {program['name']}: {e}")
        
        return timetable
    
    def save(self, filename='starplan_data.json'):
        """Save extracted data to JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        logger.info(f"Data saved to {filename}")

async def main():
    logger.info("=" * 70)
    logger.info("Starplan Timetable Scraper")
    logger.info("=" * 70)
    
    scraper = StarplanScraper()
    data = await scraper.scrape()
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)
    logger.info(f"Programs: {len(data['programs'])}")
    logger.info(f"Timetables extracted: {len(data['timetables'])}")
    
    total_lectures = sum(len(t['lectures']) for t in data['timetables'].values())
    logger.info(f"Total lectures: {total_lectures}")
    
    # Save data
    scraper.save()
    
    # Print sample
    if data['programs']:
        logger.info("\nSample Programs:")
        for prog in data['programs'][:5]:
            logger.info(f"  - {prog['name']} (ID: {prog['id']})")
    
    if data['timetables']:
        logger.info("\nSample Timetables:")
        for prog_id, tt in list(data['timetables'].items())[:3]:
            logger.info(f"  - {tt['program_name']}: {len(tt['lectures'])} lectures")

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Starplan Multi-Semester Scraper
Scraped Starplan für verschiedene Semester (SoSe26, WS26, SoSe25, etc.)
und erstellt separate Datensets für jedes Semester
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Semester-Konfiguration
SEMESTERS = {
    'SoSe26': {'start_date': '2026-03-22', 'name': 'Sommersemester 2026'},
    'WS25': {'start_date': '2025-10-15', 'name': 'Wintersemester 2025/26'},
    'SoSe25': {'start_date': '2025-03-22', 'name': 'Sommersemester 2025'},
    'WS24': {'start_date': '2024-10-15', 'name': 'Wintersemester 2024/25'},
}


class MultiSemesterScraper:
    def __init__(self):
        self.base_url = "https://vorlesungen.htw-aalen.de/splan/mobile"
        self.all_data = {}
        
    async def scrape_semester(self, semester_code: str, semester_info: dict):
        """Scrape Starplan für ein bestimmtes Semester"""
        logger.info(f"\n🎓 Scraping {semester_code}: {semester_info['name']}")
        
        try:
            from playwright.async_api import async_playwright
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.set_viewport_size({"width": 375, "height": 812})
                
                try:
                    # URL mit Semester-Datum
                    url = f"{self.base_url}?lan=de&act=tt&sel=pg&pu=50&sd=true&dfc={semester_info['start_date']}&loc=1&sa=false&cb=o"
                    
                    logger.info(f"  Loading {semester_code}...")
                    await page.goto(url, wait_until="networkidle", timeout=30000)
                    await page.wait_for_timeout(1000)
                    
                    # Extract programs
                    programs = await self._extract_programs(page)
                    logger.info(f"  Found {len(programs)} programs")
                    
                    semester_data = {
                        'semester': semester_code,
                        'semester_name': semester_info['name'],
                        'start_date': semester_info['start_date'],
                        'programs': programs,
                        'timetables': {},
                        'lectures': [],
                        'extracted_at': datetime.now().isoformat()
                    }
                    
                    # Extract timetables
                    logger.info(f"  Extracting timetables...")
                    for i, program in enumerate(programs, 1):
                        try:
                            timetable = await self._extract_timetable(page, program, semester_info)
                            if timetable:
                                semester_data['timetables'][program['id']] = timetable
                                lectures = timetable.get('lectures', [])
                                semester_data['lectures'].extend(lectures)
                                
                                if i % 10 == 0 or i == 1:
                                    logger.info(f"    [{i}/{len(programs)}] {program['name']}: {len(lectures)} lectures")
                        except Exception as e:
                            logger.debug(f"    ✗ {program['name']}: {e}")
                    
                    self.all_data[semester_code] = semester_data
                    logger.info(f"  ✅ {semester_code}: {len(semester_data['lectures'])} total lectures")
                    
                finally:
                    await browser.close()
                    
        except Exception as e:
            logger.error(f"  ❌ Error scraping {semester_code}: {e}")
    
    async def _extract_programs(self, page):
        """Extract programs from page"""
        programs = []
        try:
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
        
        return programs
    
    async def _extract_timetable(self, page, program, semester_info):
        """Extract timetable for program"""
        timetable = {
            'program_id': program['id'],
            'program_name': program['name'],
            'lectures': [],
            'schedule_table': None
        }
        
        try:
            # Important: keep semester date for every program request.
            url = (
                f"{self.base_url}?lan=de&act=tt&sel=pg&og={program['id']}&pu=50"
                f"&sd=true&dfc={semester_info['start_date']}&loc=1&sa=false&cb=o"
            )
            await page.goto(url, wait_until="networkidle", timeout=15000)
            await page.wait_for_timeout(1000)
            
            # Table extraction (if available)
            schedule_data = await page.evaluate("""
                () => {
                    const tables = document.querySelectorAll('table');
                    if (!tables.length) return null;
                    const table = tables[0];
                    
                    const rows = table.querySelectorAll('tr');
                    const lectures = [];
                    
                    for (let i = 0; i < rows.length; i++) {
                        const cells = rows[i].querySelectorAll('td, th');
                        if (cells.length >= 3) {
                            lectures.push({
                                day: (cells[0]?.textContent || '').trim(),
                                time: (cells[1]?.textContent || '').trim(),
                                name: (cells[2]?.textContent || '').trim(),
                                lecturer: (cells[3]?.textContent || '').trim(),
                                room: (cells[4]?.textContent || '').trim(),
                                info: (cells[5]?.textContent || '').trim()
                            });
                        }
                    }
                    
                    return lectures;
                }
            """)

            parsed_lectures = []

            # Text parsing fallback for mobile views without a structured table.
            body_text = await page.evaluate('() => document.body.innerText')
            lines = [ln.strip() for ln in body_text.split('\n') if ln.strip()]

            current_day = None
            current_time = None
            day_pattern = re.compile(r'(Montag|Dienstag|Mittwoch|Donnerstag|Freitag)', re.I)
            time_pattern = re.compile(r'^\d{1,2}:\d{2}(\s*-\s*\d{1,2}:\d{2})?')

            for line in lines:
                if day_pattern.search(line):
                    current_day = line
                    continue

                if time_pattern.match(line):
                    current_time = line
                    continue

                if not current_day or not current_time:
                    continue

                if len(line) < 4:
                    continue

                lower = line.lower()
                if any(skip in lower for skip in ["stundenpl", "semester", "auswahl", "export", "anzeige"]):
                    continue

                parsed_lectures.append({
                    'day': current_day,
                    'time': current_time,
                    'name': line,
                    'lecturer': '',
                    'room': '',
                    'info': line,
                    'program': program['name'],
                    'semester': semester_info['name'],
                })

            # Prefer richer table data when it contains meaningful entries.
            valid_table_lectures = []
            if schedule_data:
                for lecture in schedule_data:
                    merged = {
                        'day': lecture.get('day', ''),
                        'time': lecture.get('time', ''),
                        'name': lecture.get('name', ''),
                        'lecturer': lecture.get('lecturer', ''),
                        'room': lecture.get('room', ''),
                        'info': lecture.get('info', ''),
                        'program': program['name'],
                        'semester': semester_info['name'],
                    }
                    # Filter table headers/empty rows
                    content = f"{merged['name']} {merged['info']}".strip()
                    if len(content) >= 4 and not content.lower().startswith("tag"):
                        valid_table_lectures.append(merged)

            timetable['schedule_table'] = schedule_data
            timetable['lectures'] = valid_table_lectures if valid_table_lectures else parsed_lectures
            
        except Exception as e:
            logger.debug(f"Could not extract timetable for {program['name']}: {e}")
        
        return timetable
    
    async def run(self):
        """Scrape all configured semesters"""
        logger.info(f"🎓 Starting Multi-Semester Scraping")
        logger.info(f"Configured semesters: {', '.join(SEMESTERS.keys())}")
        
        for semester_code, semester_info in SEMESTERS.items():
            await self.scrape_semester(semester_code, semester_info)
        
        return self.all_data
    
    def save_by_semester(self):
        """Save data für jedes Semester separat"""
        for semester_code, data in self.all_data.items():
            filename = f"starplan_{semester_code}_data.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Saved {filename}")
        
        # Auch combined file für alle Semester
        filename = "starplan_all_semesters_data.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.all_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved {filename}")


async def main():
    scraper = MultiSemesterScraper()
    await scraper.run()
    scraper.save_by_semester()
    
    logger.info(f"\n✅ Multi-semester scraping complete!")
    logger.info(f"Total semesters: {len(scraper.all_data)}")
    for semester, data in scraper.all_data.items():
        logger.info(f"  {semester}: {len(data['lectures'])} lectures")


if __name__ == "__main__":
    asyncio.run(main())

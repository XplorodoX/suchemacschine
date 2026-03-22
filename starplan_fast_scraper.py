#!/usr/bin/env python3
"""
Starplan iCal Scraper - Fast version using URL patterns
"""

import requests
import json
import re
from datetime import datetime
from icalendar import Calendar
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class FastStarplanIcalScraper:
    def __init__(self):
        self.base_url = "https://vorlesungen.htw-aalen.de/splan"
        self.data = {
            'programs': [],
            'lectures': [],
            'extracted_at': datetime.now().isoformat()
        }
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_programs(self):
        """Extract programs from main page"""
        programs = []
        
        try:
            logger.info("Fetching program list...")
            
            url = f"{self.base_url}/mobile?lan=de&sel=pg&pu=50"
            r = self.session.get(url, timeout=10)
            r.encoding = 'ISO-8859-1'
            
            # Extract program IDs from HTML
            # Look for: og=<ID>
            prog_pattern = r'og=(\d+)["\'\&]'
            matches = re.findall(prog_pattern, r.text)
            
            # Also try to extract names from select element
            select_pattern = r'<select[^>]*name="og"[^>]*>.*?</select>'
            select_match = re.search(select_pattern, r.text, re.DOTALL)
            
            program_names = {}
            if select_match:
                # Extract options
                option_pattern = r'<option[^>]*value="(\d+)"[^>]*>([^<]+)</option>'
                for prog_id, name in re.findall(option_pattern, select_match.group()):
                    if prog_id != '-1':
                        program_names[prog_id] = name.strip()
            
            # Build program list
            seen_ids = set()
            for prog_id in matches:
                if prog_id not in seen_ids and prog_id != '-1':
                    seen_ids.add(prog_id)
                    program = {
                        'id': prog_id,
                        'name': program_names.get(prog_id, f'Program {prog_id}'),
                        'ical_urls': []
                    }
                    programs.append(program)
            
            logger.info(f"Found {len(programs)} programs")
            
            # Test common iCal URL patterns for first few programs
            logger.info("Testing iCal URL patterns...")
            for prog in programs[:5]:
                ical_urls = self.try_ical_urls(prog['id'])
                prog['ical_urls'] = ical_urls
                if ical_urls:
                    logger.info(f"  ✓ {prog['name']}: {len(ical_urls)} possible URL(s)")
            
        except Exception as e:
            logger.error(f"Error fetching programs: {e}")
        
        return programs
    
    def try_ical_urls(self, program_id, semester='50'):
        """Try common iCal URL patterns for a program"""
        possible_urls = [
            # Pattern 1: export endpoint with parameters
            f"{self.base_url}/export?type=ical&sel=pg&og={program_id}&pu={semester}",
            f"{self.base_url}/mobile?type=ical&sel=pg&og={program_id}&pu={semester}",
            f"{self.base_url}?export=ical&og={program_id}&pu={semester}",
            
            # Pattern 2: direct .ics file
            f"{self.base_url}/download?program={program_id}&format=ics",
            f"{self.base_url}/ics?og={program_id}",
            
            # Pattern 3: with additional parameters
            f"{self.base_url}/export?format=ics&sel=pg&og={program_id}&pu={semester}&lan=de",
        ]
        
        working_urls = []
        for url in possible_urls:
            try:
                r = self.session.head(url, timeout=5, allow_redirects=True)
                if r.status_code in [200, 301, 302, 303, 307]:
                    working_urls.append(url)
            except:
                pass
        
        return working_urls
    
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
                    }
                    
                    if 'summary' in component:
                        event['title'] = str(component['summary']).strip()
                    
                    if 'dtstart' in component:
                        dt = component['dtstart'].dt
                        event['start'] = dt.isoformat() if hasattr(dt, 'isoformat') else str(dt)
                    
                    if 'dtend' in component:
                        dt = component['dtend'].dt
                        event['end'] = dt.isoformat() if hasattr(dt, 'isoformat') else str(dt)
                    
                    if 'location' in component:
                        loc = str(component['location']).strip()
                        if loc:
                            event['location'] = loc
                    
                    if 'organizer' in component:
                        event['instructor'] = str(component['organizer']).strip()
                    
                    if 'description' in component:
                        event['description'] = str(component['description']).strip()
                    
                    if event['title']:
                        lectures.append(event)
            
        except Exception as e:
            logger.warning(f"Error parsing iCal: {e}")
        
        return lectures
    
    def download_and_parse_ical(self, program):
        """Download iCal for a program and parse it"""
        lectures = []
        
        for url in program['ical_urls']:
            try:
                logger.info(f"  Downloading {program['name']} from {url[:60]}...")
                r = self.session.get(url, timeout=30)
                
                if r.status_code == 200 and ('icalendar' in r.headers.get('content-type', '').lower() or r.content.startswith(b'BEGIN:VCALENDAR')):
                    parsed = self.parse_ical(r.content, program['name'])
                    lectures.extend(parsed)
                    logger.info(f"    ✓ {len(parsed)} events")
                    break  # Success, no need to try other URLs
                
            except Exception as e:
                logger.debug(f"    ✗ {str(e)[:60]}")
        
        return lectures
    
    def scrape_all(self, max_programs=None):
        """Scrape all programs"""
        # Get programs
        programs = self.get_programs()
        self.data['programs'] = programs
        
        if max_programs:
            programs = programs[:max_programs]
        
        logger.info("\n" + "="*70)
        logger.info(f"Downloading iCal files for {len(programs)} programs...")
        logger.info("="*70)
        
        # Download and parse in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self.download_and_parse_ical, prog): prog 
                for prog in programs if prog['ical_urls']
            }
            
            for future in as_completed(futures):
                prog = futures[future]
                try:
                    lectures = future.result()
                    self.data['lectures'].extend(lectures)
                except Exception as e:
                    logger.warning(f"Error processing {prog['name']}: {e}")
        
        return self.data
    
    def save(self, filename='starplan_ical_data.json'):
        """Save extracted data to JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        logger.info(f"Data saved to {filename}")

def main():
    logger.info("=" * 70)
    logger.info("Starplan Fast iCal Scraper")
    logger.info("=" * 70)
    
    scraper = FastStarplanIcalScraper()
    
    # Scrape programs and test iCal URLs
    data = scraper.scrape_all(max_programs=20)  # Start with 20 programs
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)
    logger.info(f"Programs: {len(data['programs'])}")
    logger.info(f"Total lectures extracted: {len(data['lectures'])}")
    
    if data['lectures']:
        logger.info("\nSample lectures (first 10):")
        for lecture in data['lectures'][:10]:
            logger.info(f"  [{lecture['program']}] {lecture['title']}")
            if lecture['location']:
                logger.info(f"    Ort: {lecture['location']}")
            if lecture['instructor']:
                logger.info(f"    Dozent: {lecture['instructor']}")
    
    # Save data
    scraper.save()
    
    return data

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Starplan iCal Scraper - Lightweight version without dependencies
Parses iCal using regex instead of icalendar library
"""

import requests
import json
import re
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SimpleIcalParser:
    """Simple iCal parser using regex"""
    
    @staticmethod
    def parse_ical_event(event_text):
        """Parse a single VEVENT block"""
        event = {
            'title': None,
            'start': None,
            'end': None,
            'location': None,
            'instructor': None,
            'description': None,
        }
        
        # Extract SUMMARY
        m = re.search(r'SUMMARY:([^\r\n]+)', event_text)
        if m:
            event['title'] = m.group(1).strip()
        
        # Extract DTSTART (various formats: 20260322T080000Z, 20260322T080000)
        m = re.search(r'DTSTART[^:]*:([^\r\n]+)', event_text)
        if m:
            event['start'] = m.group(1).strip()
        
        # Extract DTEND
        m = re.search(r'DTEND[^:]*:([^\r\n]+)', event_text)
        if m:
            event['end'] = m.group(1).strip()
        
        # Extract LOCATION
        m = re.search(r'LOCATION:([^\r\n]+)', event_text)
        if m:
            event['location'] = m.group(1).strip()
        
        # Extract ORGANIZER or ATTENDEE (might contain instructor info)
        m = re.search(r'ORGANIZER[^:]*:([^\r\n]+)', event_text)
        if m:
            org = m.group(1).strip()
            # Remove CN= prefix if present
            org = re.sub(r'^CN=', '', org)
            org = re.sub(r'mailto:', '', org)
            event['instructor'] = org
        
        # Extract DESCRIPTION
        m = re.search(r'DESCRIPTION:([^\r\n]+)', event_text)
        if m:
            event['description'] = m.group(1).strip()
        
        return event if event['title'] else None
    
    @staticmethod
    def parse_ical_content(content):
        """Parse iCal content and extract all events"""
        events = []
        
        # Decode if bytes
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='ignore')
        
        # Split into VEVENT blocks
        event_blocks = re.split(r'BEGIN:VEVENT', content)[1:]
        
        for block in event_blocks:
            # Get just the event part (until END:VEVENT)
            event_text = block.split('END:VEVENT')[0]
            event = SimpleIcalParser.parse_ical_event(event_text)
            if event:
                events.append(event)
        
        return events

class StarplanIcalScraper:
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
            logger.info("Fetching program list from Starplan...")
            
            url = f"{self.base_url}/mobile?lan=de&sel=pg&pu=50"
            r = self.session.get(url, timeout=10)
            r.encoding = 'ISO-8859-1'
            
            # Extract program IDs and names from select element
            select_pattern = r'<select[^>]*name="og"[^>]*>(.*?)</select>'
            select_match = re.search(select_pattern, r.text, re.DOTALL | re.IGNORECASE)
            
            if select_match:
                # Extract options
                option_pattern = r'<option[^>]*value="([^"]*)"[^>]*>([^<]+)</option>'
                for prog_id, name in re.findall(option_pattern, select_match.group(1)):
                    if prog_id and prog_id != '-1':
                        programs.append({
                            'id': prog_id,
                            'name': name.strip(),
                            'ical_url': self.construct_ical_url(prog_id)
                        })
            
            logger.info(f"Found {len(programs)} programs")
            
        except Exception as e:
            logger.error(f"Error fetching programs: {e}")
        
        return programs
    
    def construct_ical_url(self, program_id, semester='50'):
        """Construct likely iCal URL for a program"""
        # Try common Starplan export pattern first
        return f"{self.base_url}/export?type=ical&sel=pg&og={program_id}&pu={semester}"
    
    def download_ical(self, program):
        """Download iCal file for a program"""
        try:
            logger.info(f"  Downloading {program['name']}...")
            
            r = self.session.get(program['ical_url'], timeout=30)
            
            if r.status_code == 200:
                # Check if response looks like iCal format
                content = r.content if isinstance(r.content, bytes) else r.text.encode()
                
                if b'BEGIN:VCALENDAR' in content or 'BEGIN:VCALENDAR' in r.text:
                    logger.info(f"    ✓ Valid iCal received")
                    return r.content
                else:
                    logger.warning(f"    ⚠ Response doesn't look like iCal format")
            else:
                logger.warning(f"    ✗ HTTP {r.status_code}")
        
        except requests.Timeout:
            logger.warning(f"    ✗ Timeout")
        except Exception as e:
            logger.warning(f"    ✗ Error: {str(e)[:60]}")
        
        return None
    
    def scrape_all(self, max_programs=None):
        """Scrape all programs and their iCal files"""
        
        # Get program list
        programs = self.get_programs()
        self.data['programs'] = programs
        
        if max_programs:
            programs = programs[:max_programs]
        
        logger.info("\n" + "="*70)
        logger.info(f"Downloading iCal files for {len(programs)} programs...")
        logger.info("="*70 + "\n")
        
        # Download and parse each program's iCal
        for program in programs:
            ical_content = self.download_ical(program)
            
            if ical_content:
                lectures = SimpleIcalParser.parse_ical_content(ical_content)
                
                # Add program info to each lecture
                for lecture in lectures:
                    lecture['program'] = program['name']
                
                self.data['lectures'].extend(lectures)
                logger.info(f"    → {len(lectures)} events extracted\n")
        
        return self.data
    
    def save(self, filename='starplan_data.json'):
        """Save extracted data"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Data saved to {filename}")

def main():
    logger.info("=" * 70)
    logger.info("Starplan iCal Scraper - Lightweight Version")
    logger.info("=" * 70 + "\n")
    
    scraper = StarplanIcalScraper()
    data = scraper.scrape_all(max_programs=30)  # Scrape first 30 programs
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)
    logger.info(f"Programs found: {len(data['programs'])}")
    logger.info(f"Programs with iCal downloads: {sum(1 for p in data['programs'] if p.get('ical_url'))}")
    logger.info(f"Total lectures/events extracted: {len(data['lectures'])}")
    
    if data['lectures']:
        logger.info("\nSample lectures (first 15):")
        for i, lecture in enumerate(data['lectures'][:15], 1):
            logger.info(f"\n  {i}. {lecture['title']}")
            logger.info(f"     Studiengang: {lecture['program']}")
            if lecture['location']:
                logger.info(f"     Ort: {lecture['location']}")
            if lecture['instructor']:
                logger.info(f"     Dozent: {lecture['instructor']}")
            if lecture['start']:
                logger.info(f"     Start: {lecture['start']}")
    
    # Save
    scraper.save()
    logger.info("\n✓ Done!")

if __name__ == "__main__":
    main()

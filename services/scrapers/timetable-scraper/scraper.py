#!/usr/bin/env python3
"""
Timetable Scraper Service
Scrapt Stundenplaene ueber Starplan JSON + iCal Endpoints
"""
import os
import json
from datetime import datetime
from typing import List, Dict, Any

import requests
from icalendar import Calendar

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/data")
STARPLAN_BASE_URL = os.getenv("STARPLAN_BASE_URL", "https://vorlesungen.htw-aalen.de/splan")
PLANNING_UNIT_ID = int(os.getenv("PLANNING_UNIT_ID", "50"))
MAX_ORG_GROUPS = int(os.getenv("MAX_ORG_GROUPS", "0"))
MAX_PLANNING_GROUPS = int(os.getenv("MAX_PLANNING_GROUPS", "0"))


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "HSAalenSearchBot/1.2 (+https://www.hs-aalen.de)",
        "Accept": "application/json,text/calendar,text/plain,*/*",
    })
    return session


def bootstrap_starplan_session(session: requests.Session) -> None:
    """Prime Starplan session/cookies before calling JSON endpoints."""
    session.get(
        f"{STARPLAN_BASE_URL}/mobile",
        params={
            "lan": "de",
            "act": "tt",
            "sel": "pg",
            "pu": "50",
            "sd": "true",
            "loc": "1",
            "sa": "false",
            "cb": "o",
        },
        timeout=20,
    )


def _unwrap_json_payload(payload: Any) -> Any:
    """Starplan JSON methods often return wrapped payloads like [[...]]."""
    current = payload
    while isinstance(current, list) and len(current) == 1:
        current = current[0]
    return current


def starplan_json(session: requests.Session, method: str, params: Dict[str, Any]) -> Any:
    """Call Starplan JSON endpoint and return unwrapped payload."""
    response = session.get(
        f"{STARPLAN_BASE_URL}/json",
        params={"m": method, **params},
        timeout=20,
    )
    response.raise_for_status()
    return _unwrap_json_payload(response.json())


def parse_ical_events(ical_content: bytes) -> List[Dict[str, str]]:
    """Parse Starplan iCal content into normalized event dicts."""
    cal = Calendar.from_ical(ical_content)
    events: List[Dict[str, str]] = []

    for component in cal.walk():
        if component.name != "VEVENT":
            continue

        summary = component.get("summary")
        if not summary:
            continue

        dtstart = component.get("dtstart")
        dtend = component.get("dtend")
        location = component.get("location")
        description = component.get("description")

        events.append({
            "title": str(summary),
            "start": str(dtstart.dt) if dtstart else "",
            "end": str(dtend.dt) if dtend else "",
            "location": str(location) if location else "",
            "description": str(description) if description else "",
        })

    return events


def build_timetable_text(org_name: str, pg_name: str, events: List[Dict[str, str]]) -> str:
    lines = [f"Studiengang: {org_name}", f"Gruppe: {pg_name}", "", "Termine:"]
    for event in events:
        lines.append(
            f"- {event.get('title', '')} | {event.get('start', '')} - {event.get('end', '')}"
            f" | Ort: {event.get('location', '')}"
        )
    return "\n".join(lines)


def choose_planning_unit(planning_units: List[Dict[str, Any]]) -> int:
    if PLANNING_UNIT_ID:
        return PLANNING_UNIT_ID

    for pu in planning_units:
        if pu.get("dateasdefault"):
            return int(pu.get("id"))

    if planning_units:
        return int(planning_units[0].get("id"))

    raise RuntimeError("No planning units returned by Starplan")

def scrape_timetables():
    """Scrape timetable data via Starplan iCal export."""
    print(f"📅 Starting timetable scraper at {datetime.now()}")
    print(f"   Starplan base URL: {STARPLAN_BASE_URL}")
    
    results = []
    session = create_session()

    try:
        bootstrap_starplan_session(session)

        planning_units = starplan_json(session, "getpus", {})
        if not isinstance(planning_units, list):
            planning_units = []

        if not planning_units:
            # Retry once after a fresh bootstrap if Starplan returns empty payload.
            bootstrap_starplan_session(session)
            planning_units = starplan_json(session, "getpus", {})
            if not isinstance(planning_units, list):
                planning_units = []

        puid = choose_planning_unit(planning_units)
        print(f"   Planning unit (puid): {puid}")

        org_groups = starplan_json(session, "getogs", {"pu": puid})
        if not isinstance(org_groups, list):
            org_groups = []

        if MAX_ORG_GROUPS > 0:
            org_groups = org_groups[:MAX_ORG_GROUPS]

        print(f"   Org groups: {len(org_groups)}")

        for idx, org in enumerate(org_groups, 1):
            og_id = org.get("id")
            og_name = org.get("name") or org.get("shortname") or str(og_id)
            if not og_id:
                continue

            try:
                planning_groups = starplan_json(session, "getPgsExt", {"pu": puid, "og": og_id})
                if not isinstance(planning_groups, list):
                    planning_groups = []

                if MAX_PLANNING_GROUPS > 0:
                    planning_groups = planning_groups[:MAX_PLANNING_GROUPS]

                print(f"   [{idx}/{len(org_groups)}] {og_name}: {len(planning_groups)} groups")

                for pg in planning_groups:
                    pg_id = pg.get("id")
                    if not pg_id:
                        continue

                    pg_name = pg.get("name") or pg.get("shortname") or str(pg_id)
                    ical_url = (
                        f"{STARPLAN_BASE_URL}/ical?lan=de&puid={puid}&type=pg&pgid={pg_id}"
                    )

                    try:
                        response = session.get(ical_url, timeout=25)
                        response.raise_for_status()

                        content = response.content or b""
                        if b"BEGIN:VCALENDAR" not in content:
                            continue

                        events = parse_ical_events(content)
                        if not events:
                            continue

                        results.append({
                            "url": ical_url,
                            "title": f"{og_name} - {pg_name}",
                            "text": build_timetable_text(og_name, pg_name, events),
                            "scraped_at": datetime.now().isoformat(),
                            "source": "timetable",
                            "program": og_name,
                            "planning_group": pg_name,
                            "event_count": len(events),
                            "events": events,
                        })
                    except Exception as e:
                        print(f"      ⚠️  iCal failed for {pg_name}: {e}")

            except Exception as e:
                print(f"   ⚠️  Error loading groups for {og_name}: {e}")

    except Exception as e:
        print(f"   ❌ Starplan scraping failed: {e}")
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "timetable_data.jsonl")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Scraped {len(results)} timetables")
    print(f"   Saved to: {output_file}")

if __name__ == "__main__":
    scrape_timetables()

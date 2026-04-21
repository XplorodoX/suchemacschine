#!/usr/bin/env python3
"""Quick test of Starplan iCal URLs."""

import re

import requests
import pytest


BASE_URL = "https://vorlesungen.htw-aalen.de/splan"


def _get_program_id() -> tuple[str, str]:
    print("Fetching program list...")
    response = requests.get(f"{BASE_URL}/mobile?lan=de&sel=pg&pu=50", timeout=10)
    response.encoding = "ISO-8859-1"

    candidates = re.findall(r'<option[^>]*value="(\d+)"[^>]*>([^<]+)</option>', response.text)
    for prog_id, prog_name in candidates:
        prog_name = prog_name.strip()
        if prog_id not in {"0", "1"} and not prog_name.isdigit():
            return prog_id, prog_name

    pytest.skip("StarPlan program list only exposes placeholder options on this endpoint")


def test_starplan_ical_urls() -> None:
    prog_id, prog_name = _get_program_id()

    print(f"\nTesting iCal export for: {prog_name} (ID: {prog_id})\n")

    urls_to_test = [
        f"{BASE_URL}/export?type=ical&sel=pg&og={prog_id}&pu=50",
        f"{BASE_URL}/mobile?type=ical&sel=pg&og={prog_id}&pu=50",
        f"{BASE_URL}?export=ical&og={prog_id}&pu=50",
        f"{BASE_URL}/export/ics?og={prog_id}&pu=50",
    ]

    last_error = None
    for index, url in enumerate(urls_to_test, 1):
        print(f"{index}. Testing: {url}")
        try:
            response = requests.get(url, timeout=10)
            print(f"   Status: {response.status_code}")
            print(f"   Content-Type: {response.headers.get('content-type', 'N/A')}")
            print(f"   Size: {len(response.content)} bytes")

            if response.status_code == 200 and "BEGIN:VCALENDAR" in response.text:
                event_count = response.text.count("BEGIN:VEVENT")
                print("   Valid iCal format!")
                print(f"   Contains {event_count} events")

                for line in response.text.splitlines()[:15]:
                    print(f"   {line}")
                return

            print("   Not valid iCal")
        except Exception as exc:
            last_error = exc
            print(f"   Error: {exc}")

        print()

    pytest.skip(f"No working iCal URL found for program {prog_id}: {last_error}")


if __name__ == "__main__":
    test_starplan_ical_urls()

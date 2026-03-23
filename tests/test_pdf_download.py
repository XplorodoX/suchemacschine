#!/usr/bin/env python3
"""Test PDF download and extraction."""

import sys

import requests

sys.path.insert(0, '/home/flo/suchemacschine/scripts')

from pdf_extractor import download_and_extract_pdf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "HSAalenSearchBot/1.1 (+https://www.hs-aalen.de)",
    })
    retry = Retry(
        total=4, connect=4, read=4, backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

session = create_session()
pdf_url = "https://www.hs-aalen.de/fileadmin/content/dokumente/terminpl%C3%A4ne-semesterpl%C3%A4ne/Terminplan_Sommersemester_2026.pdf"

print("Testing PDF download and extraction from:")
print(f"  {pdf_url}\n")

try:
    text = download_and_extract_pdf(pdf_url, session)
    if text:
        print("✓ PDF extracted successfully!")
        print(f"  Content length: {len(text)} chars")
        print(f"  Preview: {text[:200]}...")
    else:
        print("✗ No text extracted from PDF")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

#!/usr/bin/env python3
"""Test PDF download and extraction."""

import sys
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

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

print(f"Testing PDF download and extraction from:")
print(f"  {pdf_url}\n")

try:
    text = download_and_extract_pdf(pdf_url, session)
    if text:
        print(f"✓ PDF extracted successfully!")
        print(f"  Content length: {len(text)} chars")
        print(f"  Preview: {text[:200]}...")
    else:
        print(f"✗ No text extracted from PDF")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

#!/usr/bin/env python3
"""Test searching for content from PDFs."""

import time

import requests

base = 'http://localhost:8000/api/search'

test_queries = [
    'Terminplan Sommersemester',  # From the PDF
    'Sommersemester timetable',   # Variant
    'welche module zweites semester informatik master',  # Our original query
]

for q in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: {q}")
    print('='*60)
    
    try:
        start = time.time()
        r = requests.get(base, params={
            'q': q,
            'page': 1,
            'per_page': 3,
            'include_summary': 'false',
            'include_rerank': 'false',
            'include_expansion': 'false',
            'strict_match': 'false',
            'provider': 'none'
        }, timeout=10)
        elapsed = time.time() - start
        
        data = r.json()
        print(f"Time: {elapsed:.2f}s")
        print(f"Results: {len(data['results'])}")
        
        for i, res in enumerate(data['results'][:3], 1):
            print(f"\n[{i}] {res.get('url', 'unknown')}")
            print(f"    Score: {res.get('score', 0):.4f}")
            text = res.get('text', '')
            if 'Terminplan' in text or 'Sommersemester' in text:
                print("    ✓ Contains PDF content!")
            print(f"    Snippet: {text[:200]}...")
    
    except Exception as e:
        print(f"Error: {e}")

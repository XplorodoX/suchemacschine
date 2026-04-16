#!/usr/bin/env python3
"""Test the new markdown summary display."""

import time

import requests

base = 'http://localhost:8000/api/search'
q = 'welche module zweites semester informatik master'

print(f"Testing Markdown Summary for: '{q}'\n")
print("="*60)

try:
    start = time.time()
    r = requests.get(base, params={
        'q': q,
        'page': 1,
        'per_page': 5,
        'include_summary': 'true',
        'include_rerank': 'false',
        'include_expansion': 'false',
        'strict_match': 'true',
        'provider': 'auto'
    }, timeout=60)
    elapsed = time.time() - start
    data = r.json()
    
    print(f"Time: {elapsed:.2f}s")
    print(f"LLM enabled: {data.get('llm_enabled')}")
    print("\nSummary (for markdown display):\n")
    
    summary = data.get('summary', '')
    if summary:
        print(summary)
        print(f"\n✓ Summary length: {len(summary)} chars")
        print("✓ Ready for Markdown rendering!")
    else:
        print("✗ No summary generated")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

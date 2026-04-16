#!/usr/bin/env python3
"""Quick test of Starplan integration"""
import requests
import json

print("Testing Starplan Search Integration...")
print("=" * 70)

try:
    r = requests.get(
        'http://localhost:8000/api/search',
        params={'q': 'Informatik', 'include_summary': 'false', 'include_expansion': 'false'},
        timeout=15
    )
    
    data = r.json()
    print(f"\nResults for 'Informatik': {data['total_results']}")
    
    # Count timetable vs webpage results
    timetable_count = sum(1 for r in data['results'] if r.get('type') == 'timetable')
    webpage_count = len(data['results']) - timetable_count
    
    print(f"  - Webpages: {webpage_count}")
    print(f"  - Timetables: {timetable_count}")
    
    # Show sample
    if timetable_count > 0:
        print("\n✅ SUCCESS: Timetable results are being returned!")
        for res in data['results']:
            if res.get('type') == 'timetable':
                print(f"\n  [{res['type']}] {res['program']} - {res['day']} {res['time']}")
                print(f"      Score: {res['score']:.3f}")
                break
    else:
        print("\n⚠️  No timetable results found in search")
        print("\nShowing first 3 results:")
        for i, res in enumerate(data['results'][:3], 1):
            print(f"  {i}. [{res.get('type', '?')}] {res['text'][:60]}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

#!/usr/bin/env python3
"""Test multi-collection search"""


import requests

print("=" * 70)
print("Testing Multi-Collection Search with Timetables")
print("=" * 70)

test_queries = [
    "Informatik Montag",
    "AI Stundenplan",
    "Elektrotechnik Vorlesung",
]

for query in test_queries:
    print(f"\nQuery: '{query}'")
    print("-" * 70)
    
    try:
        r = requests.get('http://localhost:8000/api/search', params={
            'q': query,
            'page': 1,
            'include_summary': 'false',
            'include_rerank': 'false',
            'include_expansion': 'false'
        }, timeout=10)
        
        if r.status_code == 200:
            data = r.json()
            print(f"Total results: {data['total_results']}")
            
            print("\nTop 3 results:")
            for i, result in enumerate(data['results'][:3], 1):
                typ = result.get('type', 'webpage')
                text = result.get('text', '')[:70]
                score = result.get('score', 0)
                
                print(f"\n  {i}. [{typ}] Score: {score:.3f}")
                print(f"     {text}")
                
                if result.get('program'):
                    info = f"Program: {result['program']}"
                    if result.get('day'):
                        info += f", Day: {result['day']}"
                    if result.get('time'):
                        info += f", Time: {result['time']}"
                    print(f"     {info}")
        else:
            print(f"Error: {r.status_code}")
    
    except Exception as e:
        print(f"Error: {e}")

print("\n" + "=" * 70)
print("✓ Test complete")

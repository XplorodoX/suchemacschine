import time

import requests

base = 'http://localhost:8000/api/search'
q = 'zweites semester informatik master module'

print(f'Suche: "{q}"\n')

# Primäre Suche
print('=== PRIMÄRE SUCHE ===')
start = time.time()
try:
    r = requests.get(base, params={
        'q': q, 'page': 1, 'per_page': 5,
        'include_summary': 'false',
        'include_rerank': 'false',
        'include_expansion': 'false',
        'strict_match': 'true',
        'provider': 'auto'
    }, timeout=15)
    elapsed = time.time() - start
    data = r.json()
    print(f'✓ Zeit: {elapsed:.2f}s')
    print(f'✓ Ergebnisse: {len(data["results"])} items')
    print(f'✓ Total: {data["total_results"]} matches')
    print(f'✓ LLM enabled: {data["llm_enabled"]}')
    print(f'✓ Provider: {data["provider"]}')
except Exception as e:
    print(f'✗ Fehler: {e}')

# Summary-Suche
print('\n=== SUMMARY-SUCHE (mit Expansion + Reranking) ===')
start = time.time()
try:
    r = requests.get(base, params={
        'q': q, 'page': 1, 'per_page': 5,
        'include_summary': 'true',
        'include_rerank': 'true',
        'include_expansion': 'true',
        'strict_match': 'true',
        'provider': 'auto'
    }, timeout=60)
    elapsed = time.time() - start
    data = r.json()
    print(f'✓ Zeit: {elapsed:.2f}s')
    print(f'✓ Ergebnisse: {len(data["results"])} items')
    print(f'✓ LLM enabled: {data["llm_enabled"]}')
    print(f'✓ Provider: {data["provider"]}')
    if data.get('summary'):
        print(f'✓ Summary gefunden! ({len(data["summary"])} chars)')
        print('\nSummary Preview:')
        print(data['summary'][:300] + '...')
    else:
        print('✗ KEIN Summary generiert!')
except Exception as e:
    print(f'✗ Fehler: {e}')

import time

import requests

base = 'http://localhost:8000/api/search'
q = 'zweites semester informatik master module'

print(f'Suche: "{q}"\n')

# Primäre Suche
print('=== PRIMÄRE SUCHE (pure vector) ===')
start = time.time()
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
print(f'✓ Ergebnisse: {len(data["results"])}')
print('\nScores:')
for i, r in enumerate(data['results'], 1):
    print(f'  [{i}] Score: {r.get("score", 0):.4f}')
    print(f'       URL: {r.get("url", "")}')

top1 = data['results'][0]['score'] if data['results'] else 0
top3_avg = sum(r['score'] for r in data['results'][:3]) / min(3, len(data['results'])) if data['results'] else 0
print(f'\n✓ Top1 Score: {top1:.4f} (need >= 0.5 for summary)')
print(f'✓ Top3 Avg: {top3_avg:.4f} (need >= 0.42 for summary)')
print(f'✓ Has strong evidence: {top1 >= 0.5 and top3_avg >= 0.42}')

# Try with full search (expansion + reranking)
print('\n=== FULL SEARCH (mit expansion + reranking, OHNE summary) ===')
start = time.time()
try:
    r = requests.get(base, params={
        'q': q, 'page': 1, 'per_page': 5,
        'include_summary': 'false',
        'include_rerank': 'true',
        'include_expansion': 'true',
        'strict_match': 'true',
        'provider': 'auto'
    }, timeout=45)
    elapsed = time.time() - start
    data = r.json()
    
    print(f'✓ Zeit: {elapsed:.2f}s')
    print(f'✓ Ergebnisse: {len(data["results"])}')
    print('\nScores (nach reranking):')
    for i, r in enumerate(data['results'], 1):
        print(f'  [{i}] Score: {r.get("score", 0):.4f}')
        print(f'       URL: {r.get("url", "")}')
    
    top1 = data['results'][0]['score'] if data['results'] else 0
    top3_avg = sum(r['score'] for r in data['results'][:3]) / min(3, len(data['results'])) if data['results'] else 0
    print(f'\n✓ Top1 Score: {top1:.4f} (need >= 0.5 for summary)')
    print(f'✓ Top3 Avg: {top3_avg:.4f} (need >= 0.42 for summary)')
    print(f'✓ Has strong evidence: {top1 >= 0.5 and top3_avg >= 0.42}')
except Exception as e:
    print(f'✗ Fehler: {e}')

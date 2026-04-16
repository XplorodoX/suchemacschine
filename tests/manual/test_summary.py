import time

import requests

base = 'http://localhost:8000/api/search'
q = 'zweites semester informatik master module'

print(f'Suche: "{q}"\n')

# Primäre Suche
print('=== PRIMÄRE SUCHE (pure vector, ohne cache) ===')
start = time.time()
r = requests.get(base, params={
    'q': q + ' fresh1', 'page': 1, 'per_page': 5,
    'include_summary': 'false',
    'include_rerank': 'false',
    'include_expansion': 'false',
    'strict_match': 'true',
    'provider': 'auto'
}, timeout=15)
elapsed = time.time() - start
data = r.json()
print(f'✓ Zeit: {elapsed:.2f}s')
print(f'✓ LLM enabled: {data.get("llm_enabled")}')
print(f'✓ Provider: {data.get("provider")}')

# Summary-Anfrage (sollte schnell sein, keine Expansion/Reranking)
print('\n=== SUMMARY-ANFRAGE (mit include_summary=true) ===')
start = time.time()
r = requests.get(base, params={
    'q': q + ' fresh1', 'page': 1, 'per_page': 5,
    'include_summary': 'true',
    'include_rerank': 'false',
    'include_expansion': 'false',
    'strict_match': 'true',
    'provider': 'auto'
}, timeout=30)
elapsed = time.time() - start
data = r.json()
print(f'✓ Zeit: {elapsed:.2f}s')
print(f'✓ LLM enabled: {data.get("llm_enabled")}')
print(f'✓ Provider: {data.get("provider")}')
if data.get('summary'):
    print(f'✓ Summary generiert! ({len(data["summary"])} chars)')
    print(f'\nPreview:\n{data["summary"][:250]}...')
else:
    print('✗ KEINE Summary!')

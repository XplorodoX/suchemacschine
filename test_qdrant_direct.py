#!/usr/bin/env python3
"""Direct Qdrant search test"""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(host='localhost', port=6333)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

test_query = "Informatik Montag"
vector = model.encode(test_query).tolist()

print(f"Query: '{test_query}'")
print("=" * 70)

# Search timetable collection DIRECTLY
print("\nSearching starplan_timetable collection...")
results = client.query_points(
    collection_name="starplan_timetable",
    query=vector,
    limit=5
).points

print(f"Found {len(results)} results:")
for i, res in enumerate(results, 1):
    print(f"\n  {i}. Score: {res.score:.3f}")
    print(f"     Program: {res.payload.get('program')}")
    print(f"     Day: {res.payload.get('day')}")
    print(f"     Time: {res.payload.get('time')}")
    print(f"     Info: {res.payload.get('lecture_info')[:60]}")

# Also search main collection for comparison
print("\n" + "=" * 70)
print("Searching hs_aalen_search collection for comparison...")
results2 = client.query_points(
    collection_name="hs_aalen_search",
    query=vector,
    limit=3
).points

print(f"Found {len(results2)} results:")
for i, res in enumerate(results2, 1):
    text = res.payload.get('text', '')[:60]
    print(f"  {i}. Score: {res.score:.3f}")
    print(f"     {text}")

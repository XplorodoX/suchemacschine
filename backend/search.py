import os
import sys

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Configuration
COLLECTION_NAME = "hs_aalen_search"
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
USE_E5_PREFIX = "e5" in MODEL_NAME.lower()
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


def search(query_text, limit=5):
    # Initialize model and client
    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Generate embedding for the query (e5 models need a "query: " prefix)
    payload = f"query: {query_text}" if USE_E5_PREFIX else query_text
    query_vector = model.encode(payload).tolist()

    # Search in Qdrant using the modern query_points API
    search_result = client.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=limit).points

    return search_result


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Geben Sie Ihren Suchbegriff ein: ")

    print(f"\nSuche nach: '{query}'...\n")

    results = search(query)

    if not results:
        print("Keine Ergebnisse gefunden.")
        return

    for i, res in enumerate(results):
        score = res.score
        url = res.payload.get("url")
        text = res.payload.get("text")
        title = res.payload.get("title", "")
        section = res.payload.get("section_heading", "")

        print(f"{i + 1}. [Score: {score:.4f}] {url}")
        if title:
            print(f"   Titel: {title}")
        if section:
            print(f"   Abschnitt: {section}")
        print(f"   Inhalt: {text[:200]}...")
        print("-" * 40)


if __name__ == "__main__":
    main()

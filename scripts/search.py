import re
import sys

import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Configuration
COLLECTION_NAME = "hs_aalen_search"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# Ollama Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-r1:8b"


def expand_query(query):
    """Verwendet Ollama (DeepSeek), um die Suchanfrage semantisch zu erweitern."""
    print(f"Erweitere Suche mit {OLLAMA_MODEL}...")
    prompt = f"""Du bist ein Suchassistent für die Webseite der Hochschule Aalen. 
Erweitere die folgende Suchanfrage des Nutzers um relevante Schlagworte und eine kurze, präzise Beschreibung, 
um die Vektorsuche in einer Datenbank zu verbessern. 
Antworte NUR mit dem erweiterten Suchtext ohne Einleitung oder zusätzliche Kommentare.

Originalanfrage: {query}"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=30,
        )
        response.raise_for_status()
        expanded = response.json().get("response", "").strip()

        # Bereinige <think>-Tags, falls DeepSeek diese zurückgibt
        expanded = re.sub(r"<think>.*?</think>", "", expanded, flags=re.DOTALL).strip()

        return expanded if expanded else query
    except Exception as e:
        print(f"Ollama Fehler: {e}. Nutze Originalanfrage.")
        return query


def search(query_text, limit=5):
    # Initialize model and client
    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Generate embedding for the query
    query_vector = model.encode(query_text).tolist()

    # Search in Qdrant using the modern query_points API
    search_result = client.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=limit).points

    return search_result


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Geben Sie Ihren Suchbegriff ein: ")

    print(f"\nSuche nach: '{query}'...")

    # Semantische Erweiterung mit Ollama
    semantic_query = expand_query(query)
    if semantic_query != query:
        print(f"Erweiterte Anfrage: '{semantic_query}'\n")
    else:
        print("\n")

    results = search(semantic_query)

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

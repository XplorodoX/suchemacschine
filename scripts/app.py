import re
import requests
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os

# Configuration (Overridden by Environment Variables)
COLLECTION_NAME = "hs_aalen_search"
MODEL_NAME = "all-MiniLM-L6-v2"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")

app = FastAPI(title="HS Aalen AI Search")

# Initialize models and clients once
print("Loading Embedding Model...")
model = SentenceTransformer(MODEL_NAME)
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Serve static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)


def expand_query(query: str) -> str:
    """Expand query semantically using Ollama/DeepSeek."""
    prompt = (
        "Du bist ein Suchassistent für die Webseite der Hochschule Aalen. "
        "Erweitere die folgende Suchanfrage des Nutzers um relevante Schlagworte "
        "und eine kurze, präzise Beschreibung, um die Vektorsuche in einer "
        "Datenbank zu verbessern. "
        "Antworte NUR mit dem erweiterten Suchtext ohne Einleitung oder "
        "zusätzliche Kommentare.\n\n"
        f"Originalanfrage: {query}"
    )
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=30,
        )
        response.raise_for_status()
        expanded = response.json().get("response", "").strip()
        expanded = re.sub(r"<think>.*?</think>", "", expanded, flags=re.DOTALL).strip()
        return expanded if expanded else query
    except Exception:
        return query


def generate_summary(query: str, results: list) -> str:
    """Generate an AI summary of the search results with source references."""
    snippets = ""
    for i, res in enumerate(results[:5], 1):
        text = res.get("text", "")[:300]
        url = res.get("url", "")
        snippets += f"[{i}] {url}\n{text}\n\n"

    prompt = (
        "Du bist ein hilfreicher KI-Assistent der Hochschule Aalen. "
        "Der Nutzer hat nach folgendem gesucht: \"" + query + "\"\n\n"
        "Hier sind die relevantesten Suchergebnisse:\n\n" + snippets +
        "\nErstelle eine kurze, hilfreiche Zusammenfassung (3-5 Sätze) "
        "basierend auf diesen Ergebnissen. Verweise auf die Quellen mit "
        "Nummern in eckigen Klammern wie [1], [2] etc. "
        "Schreibe auf Deutsch. Antworte NUR mit der Zusammenfassung."
    )
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
        summary = response.json().get("response", "").strip()
        summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()
        return summary if summary else ""
    except Exception:
        return ""


@app.get("/api/search")
async def api_search(
    q: str = Query(...),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=50),
):
    """Search endpoint with pagination and AI summary."""
    # 1. Expand
    semantic_query = expand_query(q)

    # 2. Embed
    query_vector = model.encode(semantic_query).tolist()

    # 3. Search (fetch more results for pagination)
    total_needed = page * per_page
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=min(total_needed + per_page, 100),
    ).points

    total_results = len(search_result)

    # 4. Format all results
    all_results = []
    for res in search_result:
        all_results.append({
            "score": res.score,
            "url": res.payload.get("url"),
            "text": res.payload.get("text"),
        })

    # 5. Paginate
    start = (page - 1) * per_page
    end = start + per_page
    page_results = all_results[start:end]

    # 6. AI Summary (only on first page)
    summary = ""
    if page == 1 and all_results:
        summary = generate_summary(q, all_results)

    return {
        "original_query": q,
        "expanded_query": semantic_query,
        "summary": summary,
        "results": page_results,
        "page": page,
        "per_page": per_page,
        "total_results": total_results,
        "has_more": end < total_results,
        "sources": [
            {"index": i + 1, "url": r["url"]}
            for i, r in enumerate(all_results[:5])
        ],
    }


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

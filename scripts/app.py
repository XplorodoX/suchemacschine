import re
import requests
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os
import numpy as np

# Configuration (Overridden by Environment Variables)
COLLECTION_NAME = "hs_aalen_search"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
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
    """Focused query expansion: produces a short, precise search sentence."""
    prompt = (
        "Du bist ein Suchassistent für die Webseite der Hochschule Aalen. "
        "Formuliere die folgende Suchanfrage als einen kurzen, präzisen "
        "Suchsatz (maximal 2 Sätze), der die Absicht des Nutzers klar "
        "beschreibt. Füge KEINE losen Keywords hinzu. "
        "Antworte NUR mit dem Suchsatz.\n\n"
        f"Suchanfrage: {query}"
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
        # Remove quotes if the model wraps it
        expanded = expanded.strip('"').strip("'").strip("„").strip("“")
        return expanded if expanded else query
    except Exception:
        return query


def hybrid_search(query: str, expanded_query: str, total_limit: int = 20):
    """Hybrid search: merge results from original + expanded query vectors."""
    # Encode both queries
    original_vector = model.encode(query).tolist()
    expanded_vector = model.encode(expanded_query).tolist()

    # Search with original query
    original_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=original_vector,
        limit=total_limit,
    ).points

    # Search with expanded query
    expanded_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=expanded_vector,
        limit=total_limit,
    ).points

    # Merge and deduplicate by URL+text, keeping the best score
    seen = {}
    for res in original_results:
        key = (res.payload.get("url", ""), res.payload.get("text", "")[:100])
        if key not in seen or res.score > seen[key].score:
            seen[key] = res

    for res in expanded_results:
        key = (res.payload.get("url", ""), res.payload.get("text", "")[:100])
        # Slight penalty for expanded-only results (original intent matters more)
        if key not in seen or res.score * 0.95 > seen[key].score:
            seen[key] = res

    # Sort by score descending
    merged = sorted(seen.values(), key=lambda x: x.score, reverse=True)
    return merged[:total_limit]


def rerank_results(query: str, results: list) -> list:
    """Use LLM to re-rank results by relevance to the original query."""
    if not results:
        return results

    # Build compact snippets for re-ranking
    snippets = ""
    for i, res in enumerate(results):
        text = res.get("text", "")[:150]
        snippets += f"[{i}] {text}\n"

    prompt = (
        "Bewerte die Relevanz jedes Suchergebnisses zur Anfrage: "
        f'"{query}"\n\n'
        f"Ergebnisse:\n{snippets}\n"
        "Antworte NUR mit einer kommagetrennten Liste der Indexnummern, "
        "sortiert nach Relevanz (relevanteste zuerst). "
        "Beispiel: 3,0,5,1,2,4"
    )
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=30,
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Parse the ranking indices
        indices = []
        for part in raw.replace(" ", "").split(","):
            part = part.strip()
            if part.isdigit():
                idx = int(part)
                if 0 <= idx < len(results) and idx not in indices:
                    indices.append(idx)

        # If we got a valid reranking, use it
        if len(indices) >= len(results) // 2:
            reranked = [results[i] for i in indices]
            # Append any results that weren't ranked
            remaining = [r for i, r in enumerate(results) if i not in indices]
            return reranked + remaining

        return results
    except Exception:
        return results


def generate_summary(query: str, results: list) -> str:
    """Generate an AI summary of the search results with source references."""
    snippets = ""
    for i, res in enumerate(results[:5], 1):
        text = res.get("text", "")[:300]
        url = res.get("url", "")
        snippets += f"[{i}] {url}\n{text}\n\n"

    prompt = (
        "Du bist ein hilfreicher KI-Assistent der Hochschule Aalen. "
        'Der Nutzer hat nach folgendem gesucht: "' + query + '"\n\n'
        "Hier sind die relevantesten Suchergebnisse:\n\n" + snippets
        + "\nErstelle eine kurze, hilfreiche Zusammenfassung (3-5 Sätze) "
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
    """Search endpoint with hybrid search, LLM re-ranking, and AI summary."""
    # 1. Focused query expansion
    semantic_query = expand_query(q)

    # 2. Hybrid search (original + expanded)
    raw_results = hybrid_search(q, semantic_query, total_limit=30)

    # 3. Format results
    all_results = []
    for res in raw_results:
        all_results.append({
            "score": res.score,
            "url": res.payload.get("url"),
            "text": res.payload.get("text"),
        })

    # 4. LLM Re-ranking (only top results)
    all_results = rerank_results(q, all_results)

    total_results = len(all_results)

    # 5. Paginate
    start = (page - 1) * per_page
    end = start + per_page
    page_results = all_results[start:end]

    # 6. AI Summary (only on first page)
    summary = ""
    if page == 1 and page_results:
        summary = generate_summary(q, page_results)

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
            for i, r in enumerate(page_results[:5])
        ],
    }


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

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


def boost_and_rank(query: str, results: list) -> list:
    """Combine vector score with keyword matching and LLM re-ranking."""
    if not results:
        return results

    # 1. Simple Keyword Boosting (BM25-lite)
    query_words = set(query.lower().split())
    for res in results:
        text = res.get("text", "").lower()
        url = res.get("url", "").lower()
        match_count = sum(1 for word in query_words if word in text or word in url)
        # Boost score by 10% for each matching word
        res["score"] = res["score"] * (1 + (match_count * 0.1))

    # Sort after boost
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    # 2. LLM Re-ranking (Top 10 only for speed and focus)
    top_n = 10
    subset = results[:top_n]
    
    snippets = ""
    for i, res in enumerate(subset):
        snippet = res.get("text", "")[:200].replace("\n", " ").strip()
        snippets += f"ID {i}: {snippet}\n"

    prompt = (
        f"Anfrage: \"{query}\"\n\n"
        "Bewerte die Relevanz der folgenden Ergebnisse. "
        "Antworte NUR mit den IDs der relevantesten Ergebnisse in "
        "der besten Reihenfolge, getrennt durch Kommata.\n"
        "Beispiel: 2, 0, 5, 1\n\n"
        f"Ergebnisse:\n{snippets}"
    )
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=40,
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Parse indices
        new_order = []
        for match in re.finditer(r"\d+", raw):
            idx = int(match.group())
            if 0 <= idx < len(subset) and idx not in new_order:
                new_order.append(idx)

        if new_order:
            reranked = [subset[i] for i in new_order]
            remaining_subset = [r for i, r in enumerate(subset) if i not in new_order]
            return reranked + remaining_subset + results[top_n:]
            
        return results
    except Exception:
        return results


def generate_summary(query: str, results: list) -> str:
    """Generate an AI summary of the search results with source references."""
    top_results = results[:5]
    snippets = ""
    for i, res in enumerate(top_results, 1):
        text = res.get("text", "")[:350]
        url = res.get("url", "")
        snippets += f"[{i}] QUELLE: {url}\nTEXT: {text}\n\n"

    prompt = (
        "Du bist der HS Aalen Such-Assistent. "
        f"Beantworte die Anfrage \"{query}\" basierend auf diesen Quellen:\n\n{snippets}\n"
        "Regeln:\n"
        "1. Nutze [1], [2] etc. für Zitate.\n"
        "2. Sei präzise und nenne konkrete Fakten.\n"
        "3. Antworte NUR mit der Zusammenfassung auf Deutsch."
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
    include_summary: bool = Query(True),
):
    """Search endpoint with hybrid search, boosting, and LLM re-ranking."""
    # 1. Expansion
    semantic_query = expand_query(q)

    # 2. Vector Search (increased limit for better re-ranking)
    raw_points = hybrid_search(q, semantic_query, total_limit=50)

    # 3. Initial Formatting
    results = []
    for p in raw_points:
        results.append({
            "score": float(p.score),
            "url": p.payload.get("url"),
            "text": p.payload.get("text"),
        })

    # 4. Boost and Re-rank
    ranked_results = boost_and_rank(q, results)

    # 5. Pagination
    start = (page - 1) * per_page
    end = start + per_page
    page_results = ranked_results[start:end]

    # 6. Summary (only on page 1 and if requested)
    summary = ""
    if page == 1 and page_results and include_summary:
        summary = generate_summary(q, ranked_results)

    return {
        "original_query": q,
        "expanded_query": semantic_query,
        "summary": summary,
        "results": page_results,
        "total_results": len(ranked_results),
        "page": page,
        "per_page": per_page,
        "has_more": end < len(ranked_results),
        "sources": [{"index": i+1, "url": r["url"]} for i, r in enumerate(ranked_results[:5])]
    }


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

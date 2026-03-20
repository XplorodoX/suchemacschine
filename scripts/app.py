import re
import requests
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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
StaticDir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(StaticDir):
    os.makedirs(StaticDir)

def expand_query(query: str):
    prompt = f"""Du bist ein Suchassistent für die Webseite der Hochschule Aalen. 
Erweitere die folgende Suchanfrage des Nutzers um relevante Schlagworte und eine kurze, präzise Beschreibung, um die Vektorsuche in einer Datenbank zu verbessern. 
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
        expanded = re.sub(r"<think>.*?</think>", "", expanded, flags=re.DOTALL).strip()
        return expanded if expanded else query
    except Exception:
        return query

@app.get("/api/search")
async def api_search(q: str = Query(...)):
    # 1. Expand
    semantic_query = expand_query(q)
    
    # 2. Embed
    query_vector = model.encode(semantic_query).tolist()
    
    # 3. Search
    search_result = client.query_points(
        collection_name=COLLECTION_NAME, 
        query=query_vector, 
        limit=6
    ).points
    
    # 4. Format
    results = []
    for res in search_result:
        results.append({
            "score": res.score,
            "url": res.payload.get("url"),
            "text": res.payload.get("text")
        })
    
    return {"original_query": q, "expanded_query": semantic_query, "results": results}

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(StaticDir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import re
import requests
import json
import os
import logging
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, Query, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

try:
    from framework.search.engine import SearchEngine
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    from framework.search.engine import SearchEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

OPENAI_FALLBACK_MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"]

# The new global engine handles caching, LLM calls, embeddings, hybrid search, etc.
# We pass the full path to the config file. Since in Docker it's copied to /app/configs
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "hs-aalen.yaml")
if not os.path.exists(CONFIG_PATH):
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs", "hs-aalen.yaml")

print("Initializing Framework SearchEngine...")
global_engine = SearchEngine(CONFIG_PATH)
print("✓ SearchEngine initialized")

app = FastAPI(title="HS Aalen AI Search")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

def is_ollama_available() -> bool:
    try:
        url = global_engine.ollama_url.replace("/api/generate", "/api/tags")
        response = requests.get(url, timeout=4)
        response.raise_for_status()
        return True
    except Exception:
        return False

def is_openai_available(openai_api_key: str = "") -> bool:
    api_key = (openai_api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not api_key:
        return False
    try:
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=6,
        )
        response.raise_for_status()
        return True
    except Exception:
        return False

def resolve_provider(provider: str, openai_api_key: str = "") -> str:
    requested = (provider or "auto").lower().strip()
    if requested in {"ollama", "openai", "none"}:
        return requested
    if is_ollama_available():
        return "ollama"
    if is_openai_available(openai_api_key):
        return "openai"
    return "none"

@app.get("/api/models")
async def list_models(
    provider: str = Query("auto", pattern="^(auto|none|ollama|openai)$"),
    openai_api_key: str = Query(""),
    x_openai_key: Optional[str] = Header(default=None, alias="X-OpenAI-Key"),
):
    active_openai_key = (openai_api_key or x_openai_key or "").strip()
    requested_provider = (provider or "auto").lower().strip()
    resolved_provider = resolve_provider(requested_provider, active_openai_key)

    if resolved_provider == "none":
        return {
            "models": [],
            "provider": "none",
            "requested_provider": requested_provider,
            "using_fallback": False,
        }

    if resolved_provider == "openai":
        api_key = (active_openai_key or os.getenv("OPENAI_API_KEY", "")).strip()
        if not api_key:
            return {
                "models": OPENAI_FALLBACK_MODELS,
                "provider": "openai",
                "requested_provider": requested_provider,
                "using_fallback": True,
            }
        try:
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=12,
            )
            response.raise_for_status()
            model_ids = [m.get("id", "") for m in response.json().get("data", []) if m.get("id")]
            preferred = [m for m in model_ids if m.startswith("gpt-")]
            preferred_sorted = sorted(set(preferred))[:30]
            return {
                "models": preferred_sorted if preferred_sorted else OPENAI_FALLBACK_MODELS,
                "provider": "openai",
                "requested_provider": requested_provider,
                "using_fallback": not bool(preferred_sorted),
            }
        except Exception:
            return {
                "models": OPENAI_FALLBACK_MODELS,
                "provider": "openai",
                "requested_provider": requested_provider,
                "using_fallback": True,
            }

    try:
        url = global_engine.ollama_url.replace("/api/generate", "/api/tags")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        models_data = response.json().get("models", [])
        return {
            "models": [m["name"] for m in models_data],
            "provider": "ollama",
            "requested_provider": requested_provider,
            "using_fallback": False,
        }
    except Exception as e:
        return {
            "models": [global_engine.ollama_model],
            "provider": "ollama",
            "requested_provider": requested_provider,
            "using_fallback": True,
        }

@app.get("/api/search")
async def api_search(
    q: str = Query(...),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=50),
    include_summary: bool = Query(True),
    include_rerank: bool = Query(True),
    include_expansion: bool = Query(True),
    strict_match: bool = Query(True),
    model_name: str = Query(""),
    provider: str = Query("auto", pattern="^(auto|none|ollama|openai)$"),
    openai_api_key: str = Query(""),
    semester: str = Query("SoSe26"),
    x_openai_key: Optional[str] = Header(default=None, alias="X-OpenAI-Key"),
):
    active_openai_key = (openai_api_key or x_openai_key or "").strip()
    requested_provider = (provider or "auto").lower().strip()
    resolved_provider = resolve_provider(requested_provider, active_openai_key)

    llm_enabled = resolved_provider in {"ollama", "openai"}

    if resolved_provider == "openai":
        model_for_provider = model_name or global_engine.openai_model
    elif resolved_provider == "ollama":
        model_for_provider = model_name or global_engine.ollama_model
    else:
        model_for_provider = ""

    # Call framework SearchEngine
    # It internally handles caching, query expansion, deduplication, and LLM rating
    ranked_results, summary, semantic_query = global_engine.search(
        query=q,
        total_limit=100,  # Grab enough for pagination
        include_expansion=include_expansion and llm_enabled,
        include_rerank=include_rerank and llm_enabled,
        include_summary=include_summary and llm_enabled and (page == 1),
        strict_match=strict_match,
        provider=resolved_provider,
        model_name=model_for_provider,
        openai_api_key=active_openai_key,
        semester=semester,
    )

    # Pagination
    start = (page - 1) * per_page
    end = start + per_page
    page_results = ranked_results[start:end]

    return {
        "original_query": q,
        "expanded_query": semantic_query,
        "summary": summary if (page == 1 and include_summary) else "",
        "results": page_results,
        "total_results": len(ranked_results),
        "page": page,
        "per_page": per_page,
        "has_more": end < len(ranked_results),
        "sources": [{"index": i+1, "url": r["url"]} for i, r in enumerate(ranked_results[:5])],
        "model": model_for_provider,
        "provider": resolved_provider,
        "requested_provider": requested_provider,
        "llm_enabled": llm_enabled,
        "semester": semester,
    }

class FeedbackRequest(BaseModel):
    query: str
    summary: str
    rating: int  
    model: str

@app.post("/api/feedback")
async def save_feedback(req: FeedbackRequest):
    try:
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "query": req.query,
            "summary": req.summary,
            "rating": req.rating,
            "model": req.model
        }
        with open("feedback.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_data, ensure_ascii=False) + "\n")
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not save feedback")

@app.get("/api/status")
async def get_status():
    try:
        client = global_engine._qdrant
        if not client:
            return {
                "status": "error",
                "message": "Qdrant not available",
                "indexing": True,
                "collections": []
            }
        
        collections_response = client.get_collections()
        collections = collections_response.collections if collections_response else []
        
        collection_status = {}
        total_points = 0
        ready_collections = 0
        
        for col in collections:
            col_name = col.name
            try:
                info = client.get_collection(col_name)
                point_count = info.points_count if info else 0
                collection_status[col_name] = {
                    "points": point_count,
                    "empty": point_count == 0
                }
                total_points += point_count
                if point_count > 0:
                    ready_collections += 1
            except Exception as e:
                collection_status[col_name] = {"error": str(e), "empty": True}
        
        total_collections = len(collections)

        def points_for(names: list[str]) -> int:
            return sum(collection_status.get(name, {}).get("points", 0) for name in names)

        website_points = points_for(["hs_aalen_website", "asta_content"])
        timetable_points = points_for(["timetable_data", "starplan_timetable", "starplan_SoSe26"])
        search_points = points_for(["hs_aalen_search"])

        if total_points == 0:
            current_stage = "Warte auf Scraper-Daten"
            stage_details = "Crawler startet und sammelt Webseiten, PDFs und Stundenpläne."
        elif website_points == 0:
            current_stage = "Scraping: Webseiten"
            stage_details = "Website-Inhalte werden gerade erfasst und vorbereitet."
        elif timetable_points == 0:
            current_stage = "Scraping: Stundenpläne"
            stage_details = "Stundenplan-Daten werden gerade gesammelt und normalisiert."
        elif search_points == 0:
            current_stage = "Indexing: Vektoren"
            stage_details = "Embeddings werden erstellt und in Qdrant indexiert."
        else:
            current_stage = "Finalisierung"
            stage_details = "Qualitätscheck und Abschluss der initialen Indizierung."

        is_indexing = total_points < 10

        point_progress = min(total_points / 10.0, 1.0)
        if total_collections > 0:
            collection_progress = ready_collections / total_collections
            progress = (collection_progress * 0.8) + (point_progress * 0.2)
        else:
            progress = point_progress * 0.8

        indexing_progress_percent = round((100.0 if not is_indexing else min(progress * 100.0, 99.0)), 1)
        
        return {
            "status": "ok",
            "indexing": is_indexing,
            "total_points": total_points,
            "ready_collections": ready_collections,
            "total_collections": total_collections,
            "indexing_progress_percent": indexing_progress_percent,
            "current_stage": current_stage,
            "stage_details": stage_details,
            "collections": collection_status,
            "reranker": {
                "model": global_engine.config.get("search", {}).get("cross_encoder_model", "") if global_engine._cross_encoder is not None else None,
                "enabled": global_engine._cross_encoder is not None,
            },
            "message": "Datenbank wird gerade indexiert... Bitte haben Sie Geduld!" if is_indexing else "Bereit zur Suche"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "indexing": True,
            "collections": []
        }

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

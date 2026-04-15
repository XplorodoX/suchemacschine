"""
Generic FastAPI application factory for the search framework.

Usage:
    # In code
    from framework.app import create_app
    app = create_app("configs/hs-aalen.yaml")

    # Via CLI
    python cli.py serve configs/hs-aalen.yaml
    python cli.py serve configs/meine-firma.yaml --port 8080
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import yaml
from fastapi import FastAPI, Query, HTTPException, Header
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .search.engine import SearchEngine

log = logging.getLogger(__name__)

# Per-project cache: {config_path: SearchEngine}
_engine_cache: dict[str, SearchEngine] = {}

# Per-project search result cache: {(project, cache_key): result}
_search_cache: dict[tuple, tuple] = {}
MAX_CACHE_SIZE = 200


def _get_engine(config_path: str) -> SearchEngine:
    """Load or return cached SearchEngine for a config."""
    if config_path not in _engine_cache:
        _engine_cache[config_path] = SearchEngine(config_path)
    return _engine_cache[config_path]


def create_app(config_path: str | Path) -> FastAPI:
    """
    Create a FastAPI application for the given config.

    The app exposes:
      GET /api/search          — main search endpoint
      GET /api/models          — list available LLM models
      POST /api/feedback       — save user feedback
      GET /                    — serve static frontend (if static/ exists)

    Args:
        config_path: path to a YAML config file (e.g. "configs/hs-aalen.yaml")
    """
    config_path = str(Path(config_path).resolve())

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    project_name = config.get("name", "search")
    description = config.get("description", f"Search engine for {project_name}")

    app = FastAPI(
        title=f"Search: {project_name}",
        description=description,
        version="2.0.0",
    )

    # Serve static frontend if it exists (looks in scripts/static/ or ./static/)
    for static_candidate in [
        Path(config_path).parent.parent / "scripts" / "static",
        Path(config_path).parent / "static",
        Path("static"),
        Path("scripts/static"),
    ]:
        if static_candidate.is_dir():
            app.mount("/static", StaticFiles(directory=str(static_candidate)), name="static")
            log.info("Serving static files from %s", static_candidate)
            _static_dir = str(static_candidate)
            break
    else:
        _static_dir = None

    # Lazy engine load (deferred so startup is fast; loads on first request)
    _engine: SearchEngine | None = None

    def get_engine() -> SearchEngine:
        nonlocal _engine
        if _engine is None:
            log.info("Initializing SearchEngine for '%s'...", project_name)
            _engine = SearchEngine(config_path)
        return _engine

    # ---------------------------------------------------------------
    # /api/search
    # ---------------------------------------------------------------

    @app.get("/api/search")
    async def api_search(
        q: str = Query(..., description="Search query"),
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
        active_key = (openai_api_key or x_openai_key or "").strip()
        engine = get_engine()

        # Cache key
        cache_key = (
            project_name, q, provider, model_name, include_rerank,
            include_expansion, strict_match, bool(active_key), semester,
        )

        if cache_key in _search_cache:
            ranked, summary, expanded_q = _search_cache[cache_key]
            log.debug("Cache hit for '%s'", q)
        else:
            ranked, summary, expanded_q = engine.search(
                query=q,
                total_limit=50,
                include_expansion=include_expansion,
                include_rerank=include_rerank,
                include_summary=include_summary and page == 1,
                strict_match=strict_match,
                provider=provider,
                model_name=model_name,
                openai_api_key=active_key,
            )
            # Evict oldest entry if cache full
            if len(_search_cache) >= MAX_CACHE_SIZE:
                _search_cache.pop(next(iter(_search_cache)))
            _search_cache[cache_key] = (ranked, summary, expanded_q)

        # Paginate
        start = (page - 1) * per_page
        end = start + per_page
        page_results = ranked[start:end]

        return {
            "project": project_name,
            "original_query": q,
            "expanded_query": expanded_q,
            "summary": summary if page == 1 and include_summary else "",
            "results": page_results,
            "total_results": len(ranked),
            "page": page,
            "per_page": per_page,
            "has_more": end < len(ranked),
            "sources": [
                {
                    "index": i + 1,
                    "url": r.get("url"),
                    "pdfs": [
                        pdf.get("url")
                        for pdf in r.get("pdf_sources", [])
                        if isinstance(pdf, dict) and pdf.get("url")
                    ],
                }
                for i, r in enumerate(ranked[:5])
            ],
            "semester": semester,
        }

    # ---------------------------------------------------------------
    # /api/models
    # ---------------------------------------------------------------

    @app.get("/api/models")
    async def list_models(
        provider: str = Query("auto", pattern="^(auto|none|ollama|openai)$"),
        openai_api_key: str = Query(""),
        x_openai_key: Optional[str] = Header(default=None, alias="X-OpenAI-Key"),
    ):
        import requests as req
        active_key = (openai_api_key or x_openai_key or "").strip()
        engine = get_engine()
        resolved = engine._resolve_provider(provider, active_key)

        if resolved == "none":
            return {"models": [], "provider": "none"}

        if resolved == "openai":
            api_key = active_key or os.getenv("OPENAI_API_KEY", "")
            try:
                r = req.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=10,
                )
                r.raise_for_status()
                ids = sorted({m["id"] for m in r.json().get("data", []) if m["id"].startswith("gpt-")})
                return {"models": ids[:30] or ["gpt-4o-mini"], "provider": "openai"}
            except Exception:
                return {"models": ["gpt-4o-mini", "gpt-4.1-mini"], "provider": "openai", "fallback": True}

        # Ollama
        try:
            base = engine.ollama_url.replace("/api/generate", "/api/tags")
            r = req.get(base, timeout=8)
            r.raise_for_status()
            return {"models": [m["name"] for m in r.json().get("models", [])], "provider": "ollama"}
        except Exception:
            return {"models": [engine.ollama_model], "provider": "ollama", "fallback": True}

    # ---------------------------------------------------------------
    # /api/feedback
    # ---------------------------------------------------------------

    class FeedbackRequest(BaseModel):
        query: str
        summary: str
        rating: int
        model: str

    @app.post("/api/feedback")
    async def save_feedback(req: FeedbackRequest):
        import json
        try:
            feedback_file = Path(f"feedback_{project_name}.jsonl")
            entry = {
                "timestamp": datetime.now().isoformat(),
                "project": project_name,
                "query": req.query,
                "summary": req.summary,
                "rating": req.rating,
                "model": req.model,
            }
            with feedback_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            return {"status": "ok"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ---------------------------------------------------------------
    # /api/config  (for debugging)
    # ---------------------------------------------------------------

    @app.get("/api/config")
    async def get_config():
        """Return the current project config (minus sensitive values)."""
        safe = {k: v for k, v in config.items() if k not in ("qdrant",)}
        return {"project": project_name, "config": safe}

    # ---------------------------------------------------------------
    # / — Serve frontend
    # ---------------------------------------------------------------

    @app.get("/")
    async def read_index():
        if _static_dir:
            index = Path(_static_dir) / "index.html"
            if index.exists():
                return FileResponse(str(index))
        return JSONResponse(
            {"project": project_name, "status": "running", "docs": "/docs"},
            status_code=200,
        )

    return app

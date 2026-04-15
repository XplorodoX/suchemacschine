"""
SearchEngine — parametrized search engine driven by a YAML config.

Handles:
  - Multi-collection hybrid search (one collection per source)
  - Dense vector search + optional Qdrant RRF sparse search
  - Query expansion (synonym-based, no LLM required)
  - LLM-based query expansion (Ollama / OpenAI, optional)
  - Cross-Encoder reranking
  - LLM reranking fallback
  - Summary generation
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from .ranking import (
    boost_and_rank,
    expand_program_terms,
    has_strong_evidence,
    normalize_text,
)

log = logging.getLogger(__name__)


class SearchEngine:
    """
    Parametrized search engine. One instance per config/project.

    Usage:
        engine = SearchEngine("configs/hs-aalen.yaml")
        results, summary, expanded_q = engine.search("Rektor HS Aalen")
    """

    def __init__(self, config_path: str | Path):
        config_path = Path(config_path)
        with config_path.open(encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.project_name = self.config["name"]
        model_name = self.config.get("embedding_model", "intfloat/multilingual-e5-base")
        self._use_e5_prefix = "e5" in model_name.lower()

        search_cfg = self.config.get("search", {})
        self.relevance_min_score = float(search_cfg.get("relevance_min_score", 0.34))
        self.program_synonyms: dict = search_cfg.get("program_synonyms") or {}
        self.module_synonyms: dict = search_cfg.get("module_synonyms") or {}

        llm_cfg = self.config.get("llm", {})
        self.ollama_url = os.getenv("OLLAMA_URL", llm_cfg.get("ollama_url", "http://localhost:11434/api/generate"))
        self.ollama_model = os.getenv("OLLAMA_MODEL", llm_cfg.get("ollama_model", "deepseek-r1:8b"))
        self.openai_model = os.getenv("OPENAI_MODEL", llm_cfg.get("openai_model", "gpt-4o-mini"))
        self.openai_url = os.getenv("OPENAI_URL", "https://api.openai.com/v1/chat/completions")

        qdrant_cfg = self.config.get("qdrant", {})
        qdrant_host = os.getenv("QDRANT_HOST", qdrant_cfg.get("host", "localhost"))
        qdrant_port = int(os.getenv("QDRANT_PORT", qdrant_cfg.get("port", 6333)))

        # Sources: each source has a collection + search_weight
        self.sources = [s for s in self.config.get("sources", []) if s.get("search_weight", 0) > 0]

        # Load models
        log.info("Loading embedding model: %s", model_name)
        from sentence_transformers import SentenceTransformer
        self._embed_model = SentenceTransformer(model_name)
        log.info("✓ Embedding model loaded")

        # Cross-Encoder
        ce_model = search_cfg.get("cross_encoder_model", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        self._cross_encoder = None
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            self._cross_encoder = CrossEncoder(ce_model)
            log.info("✓ Cross-Encoder loaded: %s", ce_model)
        except Exception as e:
            log.warning("Cross-Encoder not available: %s", e)

        # Sparse encoder
        use_sparse = self.config.get("sparse_vectors", False) or os.getenv("USE_SPARSE_VECTORS", "false").lower() == "true"
        self._sparse_encoder = None
        if use_sparse:
            sparse_model_name = self.config.get("sparse_model", "Qdrant/bm42-all-minilm-l6-v2-attentions")
            try:
                from fastembed import SparseTextEmbedding
                self._sparse_encoder = SparseTextEmbedding(model_name=sparse_model_name)
                log.info("✓ Sparse encoder loaded: %s", sparse_model_name)
            except Exception as e:
                log.warning("Sparse encoder not available: %s", e)

        # Qdrant client
        from qdrant_client import QdrantClient
        from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector, NamedSparseVector
        self._qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        self._Prefetch = Prefetch
        self._FusionQuery = FusionQuery
        self._Fusion = Fusion
        self._SparseVector = SparseVector
        self._NamedSparseVector = NamedSparseVector

        log.info("SearchEngine ready for project '%s' (%d sources)", self.project_name, len(self.sources))

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> list[float]:
        if self._use_e5_prefix:
            text = f"query: {text}"
        return self._embed_model.encode(text).tolist()

    def _sparse_vector(self, text: str):
        if self._sparse_encoder is None:
            return None
        try:
            result = next(self._sparse_encoder.embed([text]))
            return self._SparseVector(indices=result.indices.tolist(), values=result.values.tolist())
        except Exception as e:
            log.debug("Sparse encoding failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Collection search
    # ------------------------------------------------------------------

    def _query_collection(self, collection: str, dense_vec: list[float], query: str, limit: int) -> list:
        sparse_vec = self._sparse_vector(query)
        if sparse_vec is not None:
            try:
                return self._qdrant.query_points(
                    collection_name=collection,
                    prefetch=[
                        self._Prefetch(query=dense_vec, using="dense", limit=limit * 2),
                        self._Prefetch(
                            query=self._NamedSparseVector(name="sparse", vector=sparse_vec),
                            using="sparse",
                            limit=limit * 2,
                        ),
                    ],
                    query=self._FusionQuery(fusion=self._Fusion.RRF),
                    limit=limit,
                ).points
            except Exception:
                pass  # fall through to dense-only

        return self._qdrant.query_points(
            collection_name=collection,
            query=dense_vec,
            limit=limit,
        ).points

    # ------------------------------------------------------------------
    # Multi-source hybrid search
    # ------------------------------------------------------------------

    def _hybrid_search(self, query: str, expanded_query: str, total_limit: int = 50, **kwargs) -> list[dict]:
        """Search all configured collections and merge results."""
        orig_vec = self._encode(query)
        exp_vec = self._encode(expanded_query)

        all_points = []
        total_weight = sum(s.get("search_weight", 0) for s in self.sources)

        for source in self.sources:
            collection = source.get("collection")
            weight = source.get("search_weight", 0.1)
            result_type = source.get("result_type", "webpage")
            limit = max(1, int(total_limit * weight / total_weight))

            try:
                # Main query + expanded query for non-timetable sources
                if result_type == "timetable":
                    points = self._query_collection(collection, orig_vec, query, limit)
                else:
                    orig_pts = self._query_collection(collection, orig_vec, query, limit)
                    exp_pts = self._query_collection(collection, exp_vec, expanded_query, limit)
                    # Merge, prefer higher score
                    seen: dict[str, Any] = {}
                    for p in orig_pts:
                        key = (p.payload.get("url", ""), p.payload.get("text", "")[:80])
                        if key not in seen or p.score > seen[key].score:
                            seen[key] = p
                    for p in exp_pts:
                        key = (p.payload.get("url", ""), p.payload.get("text", "")[:80])
                        if key not in seen or p.score * 0.95 > seen[key].score:
                            seen[key] = p
                    points = list(seen.values())

                # Convert to result dicts
                for p in points:
                    result = self._format_point(p, result_type, source)
                    all_points.append(result)

                log.debug("  %s → %d results (weight=%.0f%%)", collection, len(points), weight * 100)
            except Exception as e:
                log.warning("Collection '%s' unavailable: %s", collection, e)

        return sorted(all_points, key=lambda x: x["score"], reverse=True)[:total_limit]

    def _format_point(self, point, result_type: str, source: dict) -> dict:
        payload = point.payload or {}

        if result_type == "timetable":
            text = (
                f"{payload.get('program', '')} - {payload.get('day', '')} "
                f"{payload.get('time', '')} - {payload.get('lecture_info', '')[:100]}"
            ) or payload.get("content", "") or payload.get("text", "")
        elif result_type in ("website", "asta"):
            text = f"{payload.get('title', '')}: {payload.get('content', '')[:200]}"
        else:
            text = payload.get("text", "")

        return {
            "score": float(point.score),
            "url": payload.get("url"),
            "text": text,
            "title": payload.get("title", text[:50]),
            "type": result_type,
            "program": payload.get("program"),
            "day": payload.get("day"),
            "time": payload.get("time"),
            "room": payload.get("room"),
            "semester": payload.get("semester"),
            "pdf_sources": payload.get("pdf_sources", []),
            "metadata": payload.get("metadata", {}),
        }

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, provider: str, model: str, api_key: str = "", timeout: int = 40) -> str:
        import requests as req
        try:
            if provider == "openai":
                r = req.post(
                    self.openai_url,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2},
                    timeout=timeout,
                )
                r.raise_for_status()
                return (r.json().get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
            else:
                r = req.post(
                    self.ollama_url,
                    json={"model": model, "prompt": prompt, "stream": False},
                    timeout=timeout,
                )
                r.raise_for_status()
                return (r.json().get("response", "") or "").strip()
        except Exception as e:
            log.debug("LLM call failed: %s", e)
            return ""

    def _llm_expand_query(self, query: str, provider: str, model: str, api_key: str = "") -> str:
        prompt = (
            f"Du bist ein Suchassistent für '{self.project_name}'. "
            "Formuliere die folgende Suchanfrage als einen kurzen, präzisen "
            "Suchsatz (maximal 2 Sätze). Füge KEINE losen Keywords hinzu. "
            "Antworte NUR mit dem Suchsatz.\n\n"
            f"Suchanfrage: {query}"
        )
        result = self._call_llm(prompt, provider, model, api_key, timeout=30)
        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
        result = result.strip('"\'„"')
        return result if result else query

    def _llm_rerank(self, query: str, results: list[dict], provider: str, model: str, api_key: str = "") -> list[dict]:
        top_n = 10
        subset = results[:top_n]
        snippets = "\n".join(
            f"ID {i}: {r.get('text', '')[:200].replace(chr(10), ' ')}"
            for i, r in enumerate(subset)
        )
        prompt = (
            f"Anfrage: \"{query}\"\n\n"
            "Bewerte die Relevanz der Ergebnisse. Antworte NUR mit einer JSON-Liste von IDs "
            "in der besten Reihenfolge, z.B. [2,0,5,1].\n\n"
            f"Ergebnisse:\n{snippets}"
        )
        raw = self._call_llm(prompt, provider, model, api_key, timeout=40)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        new_order = []
        for m in re.finditer(r"\d+", raw):
            idx = int(m.group())
            if 0 <= idx < len(subset) and idx not in new_order:
                new_order.append(idx)
        if new_order:
            reranked = [subset[i] for i in new_order]
            leftover = [r for i, r in enumerate(subset) if i not in new_order]
            return reranked + leftover + results[top_n:]
        return results

    def _generate_summary(self, query: str, results: list[dict], provider: str, model: str, api_key: str = "") -> str:
        if not has_strong_evidence(results):
            return ""
        snippets = ""
        for i, r in enumerate(results[:5], 1):
            snippets += f"[{i}] QUELLE: {r.get('url', '')}\nTEXT: {r.get('text', '')[:350]}\n\n"
        prompt = (
            f"Du bist ein hilfsbereiter studentischer Assistent von '{self.project_name}'. "
            f"Beantworte die Anfrage \"{query}\" leicht verständlich basierend auf diesen Quellen:\n\n"
            f"{snippets}"
            "Regeln: Nutze [1],[2] als Quellenangaben. Formuliere locker. "
            "Erfinde nichts. Wenn keine Antwort möglich: antworte exakt UNBEKANNT."
        )
        result = self._call_llm(prompt, provider, model, api_key, timeout=60)
        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
        if "UNBEKANNT" in result.upper():
            return ""
        return result

    # ------------------------------------------------------------------
    # Provider resolution
    # ------------------------------------------------------------------

    def _resolve_provider(self, requested: str, api_key: str = "") -> str:
        import requests as req
        if requested in {"ollama", "openai", "none"}:
            return requested
        # Auto: try Ollama first
        try:
            base = self.ollama_url.replace("/api/generate", "/api/tags")
            req.get(base, timeout=4).raise_for_status()
            return "ollama"
        except Exception:
            pass
        # Then OpenAI
        if api_key or os.getenv("OPENAI_API_KEY"):
            return "openai"
        return "none"

    # ------------------------------------------------------------------
    # Main search method
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        total_limit: int = 50,
        include_expansion: bool = True,
        include_rerank: bool = True,
        include_summary: bool = True,
        strict_match: bool = True,
        provider: str = "auto",
        model_name: str = "",
        openai_api_key: str = "",
        **kwargs,
    ) -> tuple[list[dict], str, str]:
        """
        Execute a search query.

        Returns:
            (ranked_results, summary, expanded_query)
        """
        resolved_provider = self._resolve_provider(provider, openai_api_key)
        llm_enabled = resolved_provider != "none"

        if resolved_provider == "openai":
            active_model = model_name or self.openai_model
            active_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        elif resolved_provider == "ollama":
            active_model = model_name or self.ollama_model
            active_key = ""
        else:
            active_model = ""
            active_key = ""

        # Query expansion
        expanded_query = query
        if include_expansion:
            expanded_query = expand_program_terms(query, self.program_synonyms, self.module_synonyms)
            if llm_enabled and include_expansion:
                expanded_query = self._llm_expand_query(expanded_query, resolved_provider, active_model, active_key)
                expanded_query = expand_program_terms(expanded_query, self.program_synonyms, self.module_synonyms)

        # Search
        raw_results = self._hybrid_search(query, expanded_query, total_limit=total_limit, **kwargs)

        # Rank
        ranked = boost_and_rank(
            query,
            raw_results,
            cross_encoder=self._cross_encoder,
            strict_match=strict_match,
            relevance_min_score=self.relevance_min_score,
        )

        # LLM reranking (only if no cross-encoder)
        if include_rerank and llm_enabled and self._cross_encoder is None:
            ranked = self._llm_rerank(query, ranked, resolved_provider, active_model, active_key)

        # Summary
        summary = ""
        if include_summary and llm_enabled and ranked:
            summary = self._generate_summary(query, ranked, resolved_provider, active_model, active_key)

        return ranked, summary, expanded_query

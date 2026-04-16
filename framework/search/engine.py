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

import hashlib
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from .ranking import (
    boost_and_rank,
    expand_program_terms,
    fuzzy_correct_query,
    has_strong_evidence,
    normalize_text,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TTL Cache
# ---------------------------------------------------------------------------

class _QueryTTLCache:
    """
    Thread-safe in-memory cache with per-entry TTL and a maximum size cap.

    Entries older than *ttl_seconds* are treated as stale and evicted lazily
    (on access) or eagerly when the cache exceeds *max_size* (oldest-first).
    No external dependencies — pure Python.
    """

    def __init__(self, ttl_seconds: float = 300.0, max_size: int = 512):
        self.ttl = ttl_seconds
        self.max_size = max_size
        # {key: (stored_at, value)}
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------

    def _make_key(self, **params) -> str:
        """Stable SHA-256 hash of the given keyword parameters."""
        raw = "|".join(f"{k}={v}" for k, v in sorted(params.items()))
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> tuple[bool, Any]:
        """Return (hit, value). Stale entries are removed on read."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return False, None
            stored_at, value = entry
            if time.monotonic() - stored_at > self.ttl:
                del self._store[key]
                self._misses += 1
                return False, None
            self._hits += 1
            return True, value

    def set(self, key: str, value: Any) -> None:
        """Store *value* under *key*, evicting oldest entries if over capacity."""
        with self._lock:
            if len(self._store) >= self.max_size:
                # Evict the oldest quarter of entries
                sorted_keys = sorted(self._store, key=lambda k: self._store[k][0])
                for old_key in sorted_keys[: max(1, self.max_size // 4)]:
                    del self._store[old_key]
            self._store[key] = (time.monotonic(), value)

    def clear(self) -> int:
        """Remove all entries. Returns number of entries cleared."""
        with self._lock:
            n = len(self._store)
            self._store.clear()
            self._hits = 0
            self._misses = 0
            return n

    def info(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            now = time.monotonic()
            valid = sum(1 for _, (ts, _) in self._store.items() if now - ts <= self.ttl)
            total = self._hits + self._misses
            return {
                "size": len(self._store),
                "valid_entries": valid,
                "ttl_seconds": self.ttl,
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 3) if total else 0.0,
            }


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

        # Build fuzzy-matching vocabulary from all configured synonyms.
        # Keys are canonical terms (e.g. "informatik"), values are known aliases.
        self._fuzzy_vocabulary: set[str] = set()
        for key, synonyms in self.program_synonyms.items():
            self._fuzzy_vocabulary.add(normalize_text(key))
            for s in (synonyms or []):
                self._fuzzy_vocabulary.update(normalize_text(s).split())
        for key, synonyms in self.module_synonyms.items():
            self._fuzzy_vocabulary.add(normalize_text(key))
            for s in (synonyms or []):
                self._fuzzy_vocabulary.update(normalize_text(s).split())

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

        # ColBERT Late Interaction encoder (optional)
        # Uses token-level MaxSim scoring in Qdrant for better rare-term retrieval.
        use_colbert = (
            self.config.get("colbert_vectors", False)
            or os.getenv("USE_COLBERT_VECTORS", "false").lower() == "true"
        )
        self._colbert_encoder = None
        if use_colbert:
            colbert_model_name = self.config.get(
                "colbert_model", "jinaai/jina-colbert-v2"
            )
            try:
                from fastembed import LateInteractionTextEmbedding
                self._colbert_encoder = LateInteractionTextEmbedding(model_name=colbert_model_name)
                log.info("✓ ColBERT encoder loaded: %s", colbert_model_name)
            except ImportError:
                log.warning("fastembed not installed — ColBERT disabled. pip install 'fastembed>=0.3'")
            except Exception as e:
                log.warning("ColBERT encoder not available: %s", e)

        # Qdrant client
        from qdrant_client import QdrantClient
        from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector, NamedSparseVector
        self._qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        self._Prefetch = Prefetch
        self._FusionQuery = FusionQuery
        self._Fusion = Fusion
        self._SparseVector = SparseVector
        self._NamedSparseVector = NamedSparseVector

        # Query result cache
        cache_cfg = search_cfg.get("cache", {})
        cache_ttl = float(
            os.getenv("SEARCH_CACHE_TTL", cache_cfg.get("ttl_seconds", 300))
        )
        cache_max = int(
            os.getenv("SEARCH_CACHE_MAX_SIZE", cache_cfg.get("max_size", 512))
        )
        self._cache = _QueryTTLCache(ttl_seconds=cache_ttl, max_size=cache_max)
        log.info(
            "✓ Query cache enabled (ttl=%ds, max_size=%d)",
            int(cache_ttl), cache_max,
        )

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

    def _colbert_query_vector(self, text: str) -> list[list[float]] | None:
        """
        Encode *text* into ColBERT query token vectors.

        Returns a list of token vectors (shape: n_tokens × colbert_dim) suitable
        for Qdrant MultiVector MaxSim scoring, or None if ColBERT is not loaded.
        """
        if self._colbert_encoder is None:
            return None
        try:
            result = next(self._colbert_encoder.query_embed([text]))
            return result.tolist()
        except Exception as e:
            log.debug("ColBERT query encoding failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Collection search
    # ------------------------------------------------------------------

    def _query_collection(self, collection: str, dense_vec: list[float], query: str, limit: int) -> list:
        sparse_vec = self._sparse_vector(query)
        colbert_vec = self._colbert_query_vector(query)

        # Build Prefetch arms: always Dense, + Sparse and/or ColBERT if available
        has_prefetch = sparse_vec is not None or colbert_vec is not None
        if has_prefetch:
            prefetch = [self._Prefetch(query=dense_vec, using="dense", limit=limit * 2)]
            if sparse_vec is not None:
                prefetch.append(
                    self._Prefetch(
                        query=self._NamedSparseVector(name="sparse", vector=sparse_vec),
                        using="sparse",
                        limit=limit * 2,
                    )
                )
            if colbert_vec is not None:
                # ColBERT query: list[list[float]] (one vector per query token)
                # Qdrant uses MaxSim comparator configured at collection creation time.
                prefetch.append(
                    self._Prefetch(query=colbert_vec, using="colbert", limit=limit * 2)
                )
            try:
                return self._qdrant.query_points(
                    collection_name=collection,
                    prefetch=prefetch,
                    query=self._FusionQuery(fusion=self._Fusion.RRF),
                    limit=limit,
                ).points
            except Exception:
                pass  # fall through to dense-only

        # Dense-only fallback
        return self._qdrant.query_points(
            collection_name=collection,
            query=dense_vec,
            limit=limit,
        ).points

    # ------------------------------------------------------------------
    # Multi-source hybrid search
    # ------------------------------------------------------------------

    def _hybrid_search(self, query: str, expanded_query: str, total_limit: int = 50, **kwargs) -> list[dict]:
        """Search all configured collections and merge results.

        Scores from each collection are independently normalized to [0, 1]
        (min-max) before merging so that larger collections with higher
        absolute RRF/BM25 scores don't unfairly dominate the final ranking.
        """
        orig_vec = self._encode(query)
        exp_vec = self._encode(expanded_query)

        all_results: list[dict] = []
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

                # --- Normalize scores of this collection to [0, 1] ---
                # This prevents large collections (higher absolute RRF/BM25
                # scores) from drowning out smaller ones in the merged list.
                raw_scores = [p.score for p in points]
                if raw_scores:
                    col_min = min(raw_scores)
                    col_max = max(raw_scores)
                    col_denom = (col_max - col_min) if col_max > col_min else 1.0
                else:
                    col_min, col_denom = 0.0, 1.0

                # Convert to result dicts with normalized scores
                collection_results: list[dict] = []
                for p in points:
                    result = self._format_point(p, result_type, source)
                    # Store original (raw) score and replace with normalized one
                    result["raw_score"] = result["score"]
                    result["score"] = (p.score - col_min) / col_denom
                    collection_results.append(result)

                all_results.extend(collection_results)
                log.debug(
                    "  %s → %d results (weight=%.0f%%, score range [%.4f, %.4f])",
                    collection, len(points), weight * 100, col_min, col_min + col_denom,
                )
            except Exception as e:
                log.warning("Collection '%s' unavailable: %s", collection, e)

        return sorted(all_results, key=lambda x: x["score"], reverse=True)[:total_limit]

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
            # Parent-Child retrieval: return the larger parent chunk when available.
            # The child chunk (payload["text"]) was used for retrieval precision;
            # parent_text carries enough surrounding context to be useful to the user.
            text = payload.get("parent_text") or payload.get("text", "")

        result = {
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
        # Expose child text for debugging/logging (None when not in parent-child mode)
        child_text = payload.get("text") if payload.get("parent_text") else None
        if child_text:
            result["child_text"] = child_text
        return result

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

        Results are cached by a TTL cache keyed on all parameters that affect
        the output (query, flags, limits, provider, model).  Cache hits skip
        embedding, Qdrant retrieval, ranking, and any LLM calls entirely.

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

        # Fuzzy correction: fix typos before any expansion
        # (done before cache lookup so the canonical corrected form is cached)
        corrected_query, was_corrected = fuzzy_correct_query(query, self._fuzzy_vocabulary)
        if was_corrected:
            log.info("Fuzzy correction applied: %r → %r", query, corrected_query)
            query = corrected_query

        # --- Cache lookup ---
        cache_key = self._cache._make_key(
            query=query.lower().strip(),
            total_limit=total_limit,
            include_expansion=include_expansion,
            include_rerank=include_rerank,
            include_summary=include_summary,
            strict_match=strict_match,
            provider=resolved_provider,
            model=active_model,
        )
        hit, cached = self._cache.get(cache_key)
        if hit:
            log.debug("Cache HIT for query %r", query)
            return cached

        # --- Cache miss: run full pipeline ---

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

        result = (ranked, summary, expanded_query)
        self._cache.set(cache_key, result)
        log.debug("Cache MISS — stored result for query %r", query)
        return result

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def cache_info(self) -> dict:
        """Return cache statistics (hits, misses, hit_rate, size, …)."""
        return self._cache.info()

    def cache_clear(self) -> int:
        """Invalidate all cached results. Returns number of entries removed."""
        n = self._cache.clear()
        log.info("Cache cleared — %d entries removed", n)
        return n

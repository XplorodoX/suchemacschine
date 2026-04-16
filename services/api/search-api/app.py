import re
import requests
import json
import os
import logging
import numpy as np
import unicodedata
from typing import Optional
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
score_log = logging.getLogger("scores")
from fastapi import FastAPI, Query, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# --- Global Cache ---
# Simple in-memory cache for search results
# format: {(query, model, rerank, expansion): (ranked_results, summary, expanded_query)}
search_cache = {}
MAX_CACHE_SIZE = 100
from pydantic import BaseModel
from qdrant_client import QdrantClient

# Configuration (Overridden by Environment Variables)
COLLECTION_NAME = "hs_aalen_search"
OLLAMA_URL_BASE = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_URL = f"{OLLAMA_URL_BASE}/api/generate"
OLLAMA_EMBEDDINGS_URL = f"{OLLAMA_URL_BASE}/api/embeddings"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
OPENAI_URL = os.getenv("OPENAI_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_FALLBACK_MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"]
RELEVANCE_MIN_SCORE = 0.34

GERMAN_STOPWORDS = {
    "der", "die", "das", "ein", "eine", "einer", "einem", "einen", "und", "oder", "aber", "mit", "ohne",
    "auf", "in", "im", "am", "an", "zu", "zum", "zur", "von", "für", "ist", "sind", "war", "wie", "was",
    "wer", "wo", "wann", "warum", "welche", "welcher", "welches", "ich", "du", "er", "sie", "es", "wir",
    "ihr", "nicht", "kein", "keine", "mehr", "auch", "den", "dem", "des", "bei", "über", "unter", "nach",
}

PROGRAM_QUERY_SYNONYMS = {
    "informatik": ["ai", "csc", "computer science", "kuenstliche intelligenz", "ki"],
    "computer science": ["informatik", "csc", "ai", "ki"],
    "ki": ["kuenstliche intelligenz", "ai", "informatik"],
    "kuenstliche intelligenz": ["ki", "ai", "informatik"],
    "ai": ["ki", "kuenstliche intelligenz", "informatik", "csc"],
    "csc": ["informatik", "computer science", "ai", "ki"],
    "elektrotechnik": ["et", "ee", "electronics", "elektronik"],
    "et": ["elektrotechnik", "electronics", "ee"],
    "maschinenbau": ["mb", "mechanical engineering"],
    "mb": ["maschinenbau", "mechanical engineering"],
    "wirtschaftsinformatik": ["winf", "business informatics"],
    "winf": ["wirtschaftsinformatik", "business informatics"],
}

app = FastAPI(title="HS Aalen AI Search")

# Initialize clients once
print("Connecting to Qdrant...")
try:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print("✓ Qdrant client connected")
except Exception as e:
    print(f"⚠️  Warning: Could not connect to Qdrant: {e}")
    client = None

print("✓ Ready to use Ollama for embeddings and LLM")

# Serve static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)


def call_llm(prompt: str, model_name: str, provider: str = "ollama", openai_api_key: str = "", timeout: int = 40) -> str:
    """Unified LLM call for Ollama and OpenAI."""
    if provider == "openai":
        api_key = (openai_api_key or os.getenv("OPENAI_API_KEY", "")).strip()
        if not api_key:
            return ""

        payload = {
            "model": model_name or OPENAI_MODEL,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        response = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()

    response = requests.post(
        OLLAMA_URL,
        json={"model": model_name or OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    return (response.json().get("response", "") or "").strip()


def get_ollama_embeddings(text: str) -> list[float]:
    """Get embeddings from Ollama via local API."""
    try:
        response = requests.post(
            OLLAMA_EMBEDDINGS_URL,
            json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": text},
            timeout=30,
        )
        response.raise_for_status()
        return response.json().get("embedding", [])
    except Exception as e:
        print(f"⚠️  Warning: Could not get embeddings from Ollama: {e}")
        # Return zero vector as fallback
        return [0.0] * 384  # Default embedding dimension for nomic-embed-text


def is_ollama_available() -> bool:
    try:
        response = requests.get(f"{OLLAMA_URL_BASE}/api/tags", timeout=4)
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

    # Auto mode: prefer local Ollama first, then OpenAI (if key is available), otherwise no LLM.
    if is_ollama_available():
        return "ollama"
    if is_openai_available(openai_api_key):
        return "openai"
    return "none"


def expand_query(query: str, model_name: str, provider: str = "ollama", openai_api_key: str = "") -> str:
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
        expanded = call_llm(
            prompt,
            model_name=model_name,
            provider=provider,
            openai_api_key=openai_api_key,
            timeout=30,
        )
        expanded = re.sub(r"<think>.*?</think>", "", expanded, flags=re.DOTALL).strip()
        # Remove quotes if the model wraps it
        expanded = expanded.strip('"').strip("'").strip("„").strip("“")
        return expanded if expanded else query
    except Exception:
        return query


def expand_program_terms(query: str) -> str:
    """Expand common study program aliases (e.g. Informatik <-> AI/CSC) without LLM."""
    normalized = normalize_text(query)
    expanded_terms = []

    for key, synonyms in PROGRAM_QUERY_SYNONYMS.items():
        if key in normalized:
            expanded_terms.extend(synonyms)

    if not expanded_terms:
        return query

    # Keep order stable while removing duplicates and existing query terms.
    existing_tokens = set(tokenize(query))
    deduped = []
    for term in expanded_terms:
        term_norm = normalize_text(term)
        if term_norm and term_norm not in deduped and term_norm not in existing_tokens:
            deduped.append(term_norm)

    if not deduped:
        return query

    return f"{query} {' '.join(deduped)}"


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    tokens = re.findall(r"[a-zA-Z0-9äöüÄÖÜß]{2,}", normalized)
    return [t for t in tokens if t not in GERMAN_STOPWORDS]


def extract_quoted_phrases(query: str) -> list[str]:
    phrases = re.findall(r'"([^"]+)"', query)
    return [normalize_text(p) for p in phrases if p.strip()]


def strict_match_passes(query: str, text: str, url: str) -> bool:
    haystack = f"{normalize_text(text)} {normalize_text(url)}"
    phrases = extract_quoted_phrases(query)

    # If user provides quoted phrases, enforce exact phrase containment.
    if phrases and not all(p in haystack for p in phrases):
        return False

    # Enforce at least one core token match for precision.
    core_tokens = [t for t in tokenize(query) if len(t) >= 4]
    if not core_tokens:
        return True
    return any(tok in haystack for tok in core_tokens)


def lexical_relevance(query: str, text: str, url: str) -> float:
    q_tokens = tokenize(query)
    if not q_tokens:
        return 0.0

    haystack = f"{normalize_text(text)} {normalize_text(url)}"
    matched = {tok for tok in q_tokens if tok in haystack}
    coverage = len(matched) / len(q_tokens)

    phrase_bonus = 0.0
    normalized_query = normalize_text(query)
    if len(normalized_query) >= 6 and normalized_query in haystack:
        phrase_bonus = 0.2

    long_token_bonus = 0.0
    long_tokens = [t for t in q_tokens if len(t) >= 6]
    if long_tokens:
        long_matches = sum(1 for t in long_tokens if t in haystack)
        long_token_bonus = 0.1 * (long_matches / len(long_tokens))

    score = (0.7 * coverage) + phrase_bonus + long_token_bonus
    return max(0.0, min(1.0, score))


def hybrid_search(query: str, expanded_query: str, total_limit: int = 20, semester: str = "SoSe26"):
    """Hybrid search: search both main content and timetable collections."""
    # Encode both queries using Ollama
    original_vector = get_ollama_embeddings(query)
    expanded_vector = get_ollama_embeddings(expanded_query)
    
    # Check if embeddings are valid (not just fallback zeros)
    has_valid_embeddings = original_vector and any(v != 0.0 for v in original_vector)
    
    if not has_valid_embeddings:
        print(f"⚠️  Warning: No valid embeddings, using text-only search")

    all_results = []
    
    # Search main collection (HTML content)
    try:
        if has_valid_embeddings:
            # Vector search
            original_results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=original_vector,
                limit=int(total_limit * 0.5),
            ).points
            expanded_results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=expanded_vector,
                limit=int(total_limit * 0.5),
            ).points
        else:
            # Fallback to all points if vector search fails
            original_results = client.get_points(
                collection_name=COLLECTION_NAME,
                ids=list(range(0, 100)),
                with_vectors=True,
                with_payload=True,
            ).points
            expanded_results = []

        # Merge main collection results
        seen_main = {}
        for res in original_results:
            key = (res.payload.get("url", ""), res.payload.get("text", "")[:100])
            if key not in seen_main or res.score > seen_main[key].score:
                seen_main[key] = res

        for res in expanded_results:
            key = (res.payload.get("url", ""), res.payload.get("text", "")[:100])
            # Slight penalty for expanded-only results
            if key not in seen_main or res.score * 0.95 > seen_main[key].score:
                seen_main[key] = res

        all_results.extend(seen_main.values())
        print(f"DEBUG: Main collection returned {len(seen_main)} results")
    except Exception as e:
        print(f"Error searching main collection: {e}")
    
    # Search HS Aalen website collection - 25%
    try:
        if has_valid_embeddings:
            hs_aalen_results = client.query_points(
                collection_name="hs_aalen_website",
                query=original_vector,
                limit=int(total_limit * 0.25),
            ).points
        else:
            hs_aalen_results = []
        print(f"DEBUG: HS Aalen website collection returned {len(hs_aalen_results)} results")
        all_results.extend(hs_aalen_results)
    except Exception as e:
        print(f"WARNING: HS Aalen website collection not available: {e}")
    
    # Search semester-specific timetable collection if available
    # Priority: semester-specific collection > fallback to main timetable collection
    semester_collection = f"starplan_{semester}"
    
    try:
        # Try semester-specific collection first
        timetable_results = client.query_points(
            collection_name=semester_collection,
            query=original_vector,
            limit=int(total_limit * 0.15),  # 15% from timetable
        ).points
        print(f"DEBUG: Timetable collection ({semester_collection}) returned {len(timetable_results)} results")
        all_results.extend(timetable_results)
    except Exception as e:
        # Fallback to old main timetable if semester-specific not available
        try:
            print(f"Note: {semester_collection} not available, falling back to starplan_timetable")
            timetable_results = client.query_points(
                collection_name="starplan_timetable",
                query=original_vector,
                limit=int(total_limit * 0.15),
            ).points
            print(f"DEBUG: Main timetable collection returned {len(timetable_results)} results")
            all_results.extend(timetable_results)
        except Exception as e2:
            print(f"WARNING: No timetable collection available: {e2}")
    
    # Search ASTA collection - 10%
    try:
        asta_results = client.query_points(
            collection_name="asta_content",
            query=original_vector,
            limit=int(total_limit * 0.1),  # 10% from ASTA
        ).points
        print(f"DEBUG: ASTA collection returned {len(asta_results)} results")
        all_results.extend(asta_results)
    except Exception as e:
        print(f"WARNING: ASTA collection not available: {e}")
    
    # Final sort by score descending and limit
    merged = sorted(all_results, key=lambda x: x.score, reverse=True)
    print(f"DEBUG: Total merged results: {len(merged)} (limit: {total_limit})")
    return merged[:total_limit]


def boost_and_rank(
    query: str,
    results: list,
    model_name: str,
    include_rerank: bool,
    provider: str = "ollama",
    openai_api_key: str = "",
    strict_match: bool = True,
) -> list:
    """Combine vector score with lexical relevance and optional LLM re-ranking."""
    if not results:
        return results

    # 1. Normalize vector scores for stable fusion with lexical relevance.
    vector_scores = [res.get("score", 0.0) for res in results]
    min_score = min(vector_scores)
    max_score = max(vector_scores)
    denom = (max_score - min_score) if max_score > min_score else 1.0

    for res in results:
        raw_vector = res.get("score", 0.0)
        vector_norm = (raw_vector - min_score) / denom
        lexical = lexical_relevance(query, res.get("text", ""), res.get("url", ""))

        # Weighted rank fusion: semantics dominate, lexical relevance stabilizes precision.
        res["score"] = (0.72 * vector_norm) + (0.28 * lexical)
        res["vector_score"] = float(raw_vector)
        res["lexical_score"] = float(lexical)

    if strict_match:
        # Don't apply strict matching for timetable, website, and ASTA results (they have different data structure)
        results = [
            r
            for r in results
            if r.get("type") in ("timetable", "website", "asta") or strict_match_passes(query, r.get("text", ""), r.get("url", ""))
        ]

    # Filter weak matches so summary generation does not hallucinate from irrelevant sources.
    results = [r for r in results if r["score"] >= RELEVANCE_MIN_SCORE]
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    # Score logging — one line per result for easy grep/analysis
    score_log.info("QUERY: %r  |  %d results after filter", query, len(results))
    for i, r in enumerate(results[:10]):
        score_log.info(
            "  #%d  final=%.3f  vec=%.3f  lex=%.3f  type=%-10s  url=%s",
            i + 1,
            r.get("score", 0.0),
            r.get("vector_score", 0.0),
            r.get("lexical_score", 0.0),
            r.get("type", "?"),
            r.get("url", "")[:80],
        )

    # 2. LLM Re-ranking (Only if requested)
    if not include_rerank:
        return results

    top_n = 10
    subset = results[:top_n]
    
    snippets = ""
    for i, res in enumerate(subset):
        snippet = res.get("text", "")[:200].replace("\n", " ").strip()
        snippets += f"ID {i}: {snippet}\n"

    prompt = (
        f"Anfrage: \"{query}\"\n\n"
        "Bewerte die Relevanz der folgenden Ergebnisse für die Anfrage. "
        "Antworte ausschließlich mit einer JSON-Liste von IDs in der besten Reihenfolge, "
        "z. B. [2,0,5,1]. Keine weiteren Wörter.\n\n"
        f"Ergebnisse:\n{snippets}"
    )
    try:
        raw = call_llm(
            prompt,
            model_name=model_name,
            provider=provider,
            openai_api_key=openai_api_key,
            timeout=40,
        )
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


def dedup_by_url(results: list) -> list:
    """Keep only the highest-scoring chunk per URL.

    Timetable entries share the same iCal URL but represent distinct events
    (different day/time/lecture), so they are deduplicated by (url, text[:80])
    instead of URL alone.
    """
    seen: dict[str, float] = {}   # url -> best score so far
    kept: list = []

    for r in results:  # already sorted by score descending
        url = r.get("url", "")
        score = r.get("score", 0.0)

        if r.get("type") == "timetable":
            # Each timetable event is unique — deduplicate on url+event text
            key = url + "|" + r.get("text", "")[:80]
        else:
            key = url

        if key not in seen:
            seen[key] = score
            kept.append(r)
        # else: same URL already represented by a higher-scored chunk — skip

    return kept


def has_strong_evidence(results: list) -> bool:
    """Check if results are good enough for summary generation.
    
    Lowered thresholds to allow more summaries:
    - If we have at least one result with score >= 0.45, consider it evidence
    - OR if average of top3 results >= 0.40, that's also enough
    """
    if not results:
        return False

    top1 = results[0].get("score", 0.0)
    top3 = results[:3]
    mean_top3 = sum(r.get("score", 0.0) for r in top3) / len(top3)
    
    # More lenient: either top1 >= 0.45 OR mean_top3 >= 0.40
    return (top1 >= 0.45) or (mean_top3 >= 0.40)


def generate_summary(query: str, results: list, model_name: str, provider: str = "ollama", openai_api_key: str = "") -> str:
    """Generate an AI summary using the specified model."""
    if not has_strong_evidence(results):
        return ""

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
        "3. Erfinde keine Informationen, die nicht in den Quellen stehen.\n"
        "4. Wenn die Quellen nicht ausreichen, antworte nur: KEINE_EVIDENZ"
    )
    try:
        summary = call_llm(
            prompt,
            model_name=model_name,
            provider=provider,
            openai_api_key=openai_api_key,
            timeout=60,
        )
        summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()
        if "KEINE_EVIDENZ" in summary.upper():
            return ""
        return summary if summary else ""
    except Exception:
        return ""


@app.get("/api/models")
async def list_models(
    provider: str = Query("auto", pattern="^(auto|none|ollama|openai)$"),
    openai_api_key: str = Query(""),
    x_openai_key: Optional[str] = Header(default=None, alias="X-OpenAI-Key"),
):
    """List available models for selected provider."""
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
        # Use the base URL for Ollama tags
        base_ollama = OLLAMA_URL.replace("/api/generate", "/api/tags")
        response = requests.get(base_ollama, timeout=10)
        response.raise_for_status()
        models_data = response.json().get("models", [])
        return {
            "models": [m["name"] for m in models_data],
            "provider": "ollama",
            "requested_provider": requested_provider,
            "using_fallback": False,
        }
    except Exception as e:
        print(f"Error fetching models: {e}")
        return {
            "models": [OLLAMA_MODEL],
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
    """Search endpoint with caching to improve performance."""
    active_openai_key = (openai_api_key or x_openai_key or "").strip()
    requested_provider = (provider or "auto").lower().strip()
    resolved_provider = resolve_provider(requested_provider, active_openai_key)

    llm_enabled = resolved_provider in {"ollama", "openai"}
    include_llm_expansion = include_expansion and llm_enabled
    include_rerank = include_rerank and llm_enabled
    include_summary = include_summary and llm_enabled

    if resolved_provider == "openai":
        model_for_provider = model_name or OPENAI_MODEL
    elif resolved_provider == "ollama":
        model_for_provider = model_name or OLLAMA_MODEL
    else:
        model_for_provider = ""

    cache_key = (
        q,
        requested_provider,
        resolved_provider,
        model_for_provider,
        include_rerank,
        include_expansion,
        strict_match,
        bool(active_openai_key),
        semester,
    )

    # 1. Check Cache
    if cache_key in search_cache:
        ranked_results, summary, semantic_query = search_cache[cache_key]
        print(f"DEBUG: Cache hit for '{q}'")
    else:
        print(f"DEBUG: Cache miss for '{q}'. Performing search...")
        # A. Expansion
        semantic_query = (
            expand_query(q, model_for_provider, provider=resolved_provider, openai_api_key=active_openai_key)
            if include_llm_expansion
            else q
        )
        if include_expansion:
            semantic_query = expand_program_terms(semantic_query)

        # B. Vector Search
        raw_points = hybrid_search(q, semantic_query, total_limit=50, semester=semester)

        # C. Formatting
        results = []
        for p in raw_points:
            # Handle HTML content, HS Aalen Website, ASTA, and Timetable data
            source = p.payload.get('source', '')
            payload_type = p.payload.get('type', '')
            is_timetable = source == 'starplan_timetable' or payload_type == 'timetable' or bool(p.payload.get('day') and p.payload.get('time'))
            is_hs_website = source == 'hs_aalen_website'
            is_asta = source == 'asta_website'
            
            # For timetable entries, create a formatted text
            if is_timetable:
                formatted_text = f"{p.payload.get('program', '')} - {p.payload.get('day', '')} {p.payload.get('time', '')} - {p.payload.get('lecture_info', '')[:100]}"
                result_type = 'timetable'
            elif is_hs_website:
                # For HS Aalen website, use title + content
                formatted_text = f"{p.payload.get('title', '')}: {p.payload.get('content', '')[:150]}"
                result_type = 'website'
            elif is_asta:
                # For ASTA content, use title + content
                formatted_text = f"{p.payload.get('title', '')}: {p.payload.get('content', '')[:150]}"
                result_type = 'asta'
            else:
                formatted_text = p.payload.get("text", "")
                result_type = 'webpage'
            
            results.append({
                "score": float(p.score),
                "url": p.payload.get("url"),
                "text": formatted_text,
                "title": p.payload.get("title"),  # For website/asta results
                "program": p.payload.get("program"),  # For timetable
                "day": p.payload.get("day"),  # For timetable
                "time": p.payload.get("time"),  # For timetable
                "semester": p.payload.get("semester"),
                "room": p.payload.get("room"),
                "type": result_type,  # "timetable", "website", "asta", or "webpage"
            })

        # D. Boost and Re-rank
        ranked_results = boost_and_rank(
            q,
            results,
            model_for_provider,
            include_rerank,
            provider=resolved_provider,
            openai_api_key=active_openai_key,
            strict_match=strict_match,
        )

        # D2. URL-based deduplication — best chunk per URL wins
        before_dedup = len(ranked_results)
        ranked_results = dedup_by_url(ranked_results)
        score_log.info("Dedup: %d → %d results (-%d duplicate URLs)", before_dedup, len(ranked_results), before_dedup - len(ranked_results))

        # E. Summary (only generated on page 1)
        summary = ""
        if ranked_results and include_summary and page == 1:
            summary = generate_summary(
                q,
                ranked_results,
                model_for_provider,
                provider=resolved_provider,
                openai_api_key=active_openai_key,
            )

        # Save to Cache (Simple size management)
        if len(search_cache) >= MAX_CACHE_SIZE:
            search_cache.pop(next(iter(search_cache)))
        search_cache[cache_key] = (ranked_results, summary, semantic_query)

    # If search was first run without summary, lazily generate it once on page 1.
    if page == 1 and include_summary and ranked_results and not summary:
        summary = generate_summary(
            q,
            ranked_results,
            model_for_provider,
            provider=resolved_provider,
            openai_api_key=active_openai_key,
        )
        search_cache[cache_key] = (ranked_results, summary, semantic_query)

    # 2. Pagination (Applied to cached results)
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
    rating: int  # 1 for up, -1 for down
    model: str


@app.post("/api/feedback")
async def save_feedback(req: FeedbackRequest):
    """Save user feedback to a JSONL file."""
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
        print(f"Error saving feedback: {e}")
        raise HTTPException(status_code=500, detail="Could not save feedback")


@app.get("/api/status")
async def get_status():
    """Check if the database has indexed data."""
    try:
        if not client:
            return {
                "status": "error",
                "message": "Qdrant not available",
                "indexing": True,
                "collections": []
            }
        
        # Get all collections
        collections_response = client.get_collections()
        collections = collections_response.collections if collections_response else []
        
        collection_status = {}
        total_points = 0
        ready_collections = 0
        
        for col in collections:
            col_name = col.name
            try:
                # Try to get collection info
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
                print(f"Error getting info for collection {col_name}: {e}")
                collection_status[col_name] = {"error": str(e), "empty": True}
        
        total_collections = len(collections)

        def points_for(names: list[str]) -> int:
            return sum(collection_status.get(name, {}).get("points", 0) for name in names)

        website_points = points_for(["hs_aalen_website", "asta_content"])
        timetable_points = points_for(["timetable_data", "starplan_timetable"])
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

        # System is "indexing" if dataset is still tiny.
        is_indexing = total_points < 10

        # Progress is based on collection readiness + warm-up points.
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
            "message": "Datenbank wird gerade indexiert... Bitte haben Sie Geduld!" if is_indexing else "Bereit zur Suche"
        }
    except Exception as e:
        print(f"Error getting status: {e}")
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

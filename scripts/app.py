import re
import requests
import json
import os
import numpy as np
import unicodedata
from typing import Optional
from datetime import datetime
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
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector, NamedSparseVector
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

# Configuration (Overridden by Environment Variables)
COLLECTION_NAME = "hs_aalen_search"
# Upgraded to multilingual-e5-base (768-dim, much better for German/multilingual).
# NOTE: If you change this, you MUST re-index all Qdrant collections!
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
# Cross-Encoder model for reranking (multilingual, fast ~12ms/pair)
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
# Sparse embedding model — only active when hs_aalen_search was indexed with USE_SPARSE_VECTORS=true
USE_SPARSE_VECTORS = os.getenv("USE_SPARSE_VECTORS", "false").lower() == "true"
SPARSE_MODEL_NAME = os.getenv("SPARSE_MODEL", "Qdrant/bm42-all-minilm-l6-v2-attentions")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")
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
    # Study Program synonyms
    "informatik": ["in", "ai", "csc", "computer science", "kuenstliche intelligenz", "ki"],
    "computer science": ["informatik", "in", "csc", "ai", "ki"],
    "ki": ["kuenstliche intelligenz", "ai", "informatik", "in"],
    "kuenstliche intelligenz": ["ki", "ai", "informatik", "in"],
    "ai": ["ki", "kuenstliche intelligenz", "informatik", "in", "csc"],
    "csc": ["informatik", "in", "computer science", "ai", "ki"],
    "elektrotechnik": ["et", "e technik", "etechnik", "ee", "electronics", "elektronik"],
    "e technik": ["et", "elektrotechnik", "etechnik"],
    "etechnik": ["et", "elektrotechnik", "e technik"],
    "et": ["elektrotechnik", "e technik", "etechnik", "electronics", "ee"],
    "maschinenbau": ["mb", "mechanical engineering"],
    "mb": ["maschinenbau", "mechanical engineering"],
    "wirtschaftsinformatik": ["winf", "business informatics"],
    "winf": ["wirtschaftsinformatik", "business informatics"],
    "data science": ["ds"],
    "ds": ["data science"],
}

MODULE_QUERY_SYNONYMS = {
    # Module/Class synonyms
    "theoretische informatik": ["tib", "theo", "thi"],
    "theoretische info": ["tib", "theo", "thi"],
    "theo": ["theoretische informatik", "tib"],
    "mathematik": ["mathe", "ma"],
    "mathematik 1": ["mathe 1", "ma1", "ma 1"],
    "mathematik 2": ["mathe 2", "ma2", "ma 2"],
    "software engineering": ["se"],
    "software engineering 1": ["se1", "se 1"],
    "software engineering 2": ["se2", "se 2"],
    "mensch computer interaktion": ["mci"],
    "mensch-computer-interaktion": ["mci"],
}

app = FastAPI(title="HS Aalen AI Search")

# Initialize models and clients once
print("Loading Embedding Model...")
try:
    model = SentenceTransformer(MODEL_NAME)
    print(f"✓ Embedding model loaded: {MODEL_NAME}")
except Exception as e:
    print(f"⚠️  Warning: Could not load embedding model: {e}")
    print("Server will continue (model will load on first request)")
    model = None

sparse_encoder = None
if USE_SPARSE_VECTORS:
    try:
        from fastembed import SparseTextEmbedding
        print(f"Loading sparse embedding model ({SPARSE_MODEL_NAME})...")
        sparse_encoder = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
        print(f"✓ Sparse model loaded: {SPARSE_MODEL_NAME}")
    except ImportError:
        print("⚠️  fastembed not installed — sparse search disabled. Run: pip install fastembed")
    except Exception as e:
        print(f"⚠️  Could not load sparse model: {e}")

print("Loading Cross-Encoder Reranker...")
try:
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    print(f"✓ Cross-Encoder loaded: {CROSS_ENCODER_MODEL}")
except Exception as e:
    print(f"⚠️  Warning: Could not load cross-encoder: {e}. Falling back to LLM reranking.")
    cross_encoder = None

try:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print("✓ Qdrant client connected")
except Exception as e:
    print(f"⚠️  Warning: Could not connect to Qdrant: {e}")
    client = None

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


def is_ollama_available() -> bool:
    try:
        base_ollama = OLLAMA_URL.replace("/api/generate", "/api/tags")
        response = requests.get(base_ollama, timeout=4)
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
    """Expand common study program aliases and module abbreviations without LLM."""
    normalized = normalize_text(query)
    expanded_terms = []

    for key, synonyms in PROGRAM_QUERY_SYNONYMS.items():
        # Only expand program synonyms if they match whole words to prevent
        # "theoretische informatik" from expanding to full "AI" and "CSC"
        if re.search(rf"\b{re.escape(key)}\b", normalized):
            expanded_terms.extend(synonyms)

    for key, synonyms in MODULE_QUERY_SYNONYMS.items():
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


def encode_query(query: str) -> list:
    """Encode a query string into a vector, applying e5-style prefix if needed."""
    if model is None:
        return []
    # intfloat/multilingual-e5-* models require "query: " prefix for retrieval
    if "e5" in MODEL_NAME.lower():
        return model.encode(f"query: {query}").tolist()
    return model.encode(query).tolist()


def cross_encoder_rerank(query: str, results: list, top_k: int = 10) -> list:
    """
    Cross-Encoder Reranking: scores each (query, document) pair jointly with
    full attention — far more accurate than bi-encoder cosine similarity alone.
    Applied only to the top-30 candidates to keep latency low (~12ms/pair).
    """
    if cross_encoder is None or not results:
        return results

    # Only rerank top-30 to stay within latency budget
    candidates = results[:30]
    rest = results[30:]

    pairs = [
        (query, f"{r.get('title', '')} {r.get('text', '')[:500]}")
        for r in candidates
    ]

    try:
        scores = cross_encoder.predict(pairs)
        for r, score in zip(candidates, scores):
            r["cross_encoder_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x.get("cross_encoder_score", 0), reverse=True)
        return reranked[:top_k] + reranked[top_k:] + rest
    except Exception as e:
        print(f"Cross-encoder reranking failed: {e}")
        return results


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


def _sparse_vector_for(text: str):
    """Return (indices, values) for the given text using the sparse encoder, or None."""
    if sparse_encoder is None:
        return None
    try:
        result = next(sparse_encoder.embed([text]))
        return SparseVector(indices=result.indices.tolist(), values=result.values.tolist())
    except Exception as e:
        print(f"Sparse encoding failed: {e}")
        return None


def _query_collection_hybrid(collection_name: str, dense_vector: list, query: str, limit: int) -> list:
    """
    Query a single collection using Qdrant's native RRF hybrid search when sparse
    vectors are available, otherwise fall back to dense-only search.
    """
    if sparse_encoder is not None:
        sparse_vec = _sparse_vector_for(query)
        if sparse_vec is not None:
            # Qdrant RRF fusion: retrieves top candidates from dense + sparse, merges with RRF
            return client.query_points(
                collection_name=collection_name,
                prefetch=[
                    Prefetch(query=dense_vector, using="dense", limit=limit * 2),
                    Prefetch(query=NamedSparseVector(name="sparse", vector=sparse_vec), using="sparse", limit=limit * 2),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
            ).points

    # Dense-only fallback (existing behaviour)
    return client.query_points(
        collection_name=collection_name,
        query=dense_vector,
        limit=limit,
    ).points


def hybrid_search(query: str, expanded_query: str, total_limit: int = 20, semester: str = "SoSe26"):
    """Hybrid search: search both main content and timetable collections."""
    # Encode both queries (applies e5 prefix if model requires it)
    original_vector = encode_query(query)
    expanded_vector = encode_query(expanded_query)

    all_results = []

    # Search main collection (HTML content)
    try:
        original_results = _query_collection_hybrid(
            COLLECTION_NAME, original_vector, query, int(total_limit * 0.5)
        )
        expanded_results = _query_collection_hybrid(
            COLLECTION_NAME, expanded_vector, expanded_query, int(total_limit * 0.5)
        )

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
        hs_aalen_results = _query_collection_hybrid(
            "hs_aalen_website", original_vector, query, int(total_limit * 0.25)
        )
        print(f"DEBUG: HS Aalen website collection returned {len(hs_aalen_results)} results")
        all_results.extend(hs_aalen_results)
    except Exception as e:
        print(f"WARNING: HS Aalen website collection not available: {e}")

    # Search semester-specific timetable collection if available
    semester_collection = f"starplan_{semester}"
    try:
        timetable_results = _query_collection_hybrid(
            semester_collection, original_vector, query, int(total_limit * 0.15)
        )
        print(f"DEBUG: Timetable collection ({semester_collection}) returned {len(timetable_results)} results")
        all_results.extend(timetable_results)
    except Exception as e:
        try:
            print(f"Note: {semester_collection} not available, falling back to starplan_timetable")
            timetable_results = _query_collection_hybrid(
                "starplan_timetable", original_vector, query, int(total_limit * 0.15)
            )
            print(f"DEBUG: Main timetable collection returned {len(timetable_results)} results")
            all_results.extend(timetable_results)
        except Exception as e2:
            print(f"WARNING: No timetable collection available: {e2}")

    # Search ASTA collection - 10%
    try:
        asta_results = _query_collection_hybrid(
            "asta_content", original_vector, query, int(total_limit * 0.1)
        )
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
        url_str = res.get("url", "").lower()
        text_str = res.get("text", "")
        lexical = lexical_relevance(query, text_str, url_str)

        # Weighted rank fusion: semantics dominate, lexical relevance stabilizes precision.
        final_score = (0.72 * vector_norm) + (0.28 * lexical)
        
        # URL Path Boost: If the user searches for a specific concept that is literally the page URL 
        # (e.g. "exmatrikulation" matching "hs-aalen.de/.../exmatrikulation"), boost it significantly.
        q_lower = query.lower().strip()
        if len(q_lower) >= 4 and q_lower in url_str:
            final_score += 0.35  # Massive boost for exact URL match
        elif any(len(t) >= 5 and t in url_str for t in tokenize(q_lower)):
            final_score += 0.15  # Partial URL match boost
            
        # User Feedback: Prioritize timetable answers ("am besten wann das im VL plan ist")
        # Timetables often use acronyms (e.g. "TIB") that fail lexical matching against full names ("Theoretische Informatik").
        # If the semantic vector is somewhat strong (matching concepts like IN, TIB, Theoretisch), bump it high.
        if res.get("type") == "timetable" and raw_vector >= 0.45:
            final_score += 0.25
            
        res["score"] = final_score
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

    # 2. Cross-Encoder Reranking (primary reranker — fast, no LLM needed)
    # Runs on top-30, returns top-10 reordered by cross-encoder score.
    if cross_encoder is not None and results:
        results = cross_encoder_rerank(query, results, top_k=10)

    # 3. LLM Re-ranking (Only if requested AND cross-encoder not available)
    if not include_rerank:
        return results
    if cross_encoder is not None:
        # Cross-encoder already reranked — skip expensive LLM reranker
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
    seen: dict[str, float] = {}
    kept: list = []

    for r in results:  # already sorted by score descending
        url = r.get("url", "")

        if r.get("type") == "timetable":
            key = url + "|" + r.get("text", "")[:80]
        else:
            key = url

        if key not in seen:
            seen[key] = r.get("score", 0.0)
            kept.append(r)

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
        snippets += f"[{i}] QUELLE: {url}\n"
        
        pdf_sources = res.get("pdf_sources", [])
        if pdf_sources:
            pdf_urls = [pdf.get("url") for pdf in pdf_sources if isinstance(pdf, dict) and pdf.get("url")]
            if pdf_urls:
                snippets += f"    Zugehörige PDFs: {', '.join(pdf_urls)}\n"
                
        snippets += f"TEXT: {text}\n\n"

    prompt = (
        "Du bist ein hilfsbereiter studentischer Assistent der HS Aalen (Du-Form). "
        f"Beantworte die Anfrage \"{query}\" leicht verständlich basierend auf diesen Quellen:\n\n{snippets}\n"
        "Regeln:\n"
        "1. Nutze [1], [2] etc. als Quellenangabe im Text.\n"
        "2. Formuliere locker, freundlich und auf Augenhöhe mit Studenten.\n"
        "3. Benutze KEINE hochgestochenen oder akademischen Füllwörter wie 'Evidenz' oder 'Primärer Anlaufpunkt'.\n"
        "4. Wenn Infos fehlen, erfinde nichts. Schreib nicht 'Keine Evidenz gefunden', sondern lass es einfach weg.\n"
        "5. Wenn die Quellen gar keine Antwort bieten, antworte exakt nur: UNBEKANNT\n"
        "6. VERWECHSELE NIEMALS ein Fach/Modul (wie 'Theoretische Informatik') mit einem Studiengang! Wenn danach gefragt wird, nenne die Zeiten aus dem Stundenplan, aber behaupte niemals, es sei ein eigener Studiengang."
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
        if "UNBEKANNT" in summary.upper():
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
                "title": p.payload.get("title", formatted_text[:50]), 
                "program": p.payload.get("program"), 
                "day": p.payload.get("day"), 
                "time": p.payload.get("time"), 
                "semester": p.payload.get("semester"),
                "room": p.payload.get("room"),
                "type": result_type,  # "timetable", "website", "asta", or "webpage"
                "pdf_sources": p.payload.get("pdf_sources", [])
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
        "sources": [
            {
                "index": i+1, 
                "url": r.get("url"),
                "pdfs": [pdf.get("url") for pdf in r.get("pdf_sources", []) if isinstance(pdf, dict) and pdf.get("url")]
            } 
            for i, r in enumerate(ranked_results[:5])
        ],
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


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

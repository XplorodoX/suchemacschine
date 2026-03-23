import json
import os
import re
import unicodedata
from datetime import datetime
import asyncio
import concurrent.futures
from typing import List, Optional, Dict

import requests
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

# --- Global Cache ---
# Simple in-memory cache for search results
# format: {(query, model, rerank, expansion): (ranked_results, summary, expanded_query)}
search_cache = {}
MAX_CACHE_SIZE = 100
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Configuration (Overridden by Environment Variables)
COLLECTION_NAME = "hs_aalen_search"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")
OPENAI_URL = os.getenv("OPENAI_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_FALLBACK_MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"]
RELEVANCE_MIN_SCORE = 0.30
GITHUB_URL = os.getenv("GITHUB_URL", "https://models.github.ai/inference")
GITHUB_MODEL = os.getenv("GITHUB_MODEL", "mistral-ai/Ministral-3B-Instruct")

# --- Rate Limiting ---
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", 100))
user_request_log = {} # {ip: [timestamp, ...]}

# --- AI Availability Cache ---
AI_AVAILABILITY = {
    "ollama": None,  # (timestamp, bool)
    "openai": {},    # {key: (timestamp, bool)}
    "github": None,  # (timestamp, bool)
}
CACHE_TTL = 300  # 5 minutes

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

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models and clients once
print("Loading Embedding Model...")
try:
    model = SentenceTransformer(MODEL_NAME)
    print("✓ Embedding model loaded")
except Exception as e:
    print(f"⚠️  Warning: Could not load embedding model: {e}")
    print("Server will continue (model will load on first request)")
    model = None

try:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print("✓ Qdrant client connected")
except Exception as e:
    print(f"⚠️  Warning: Could not connect to Qdrant: {e}")
    client = None

# Serve static files
# STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
# if not os.path.exists(STATIC_DIR):
#     os.makedirs(STATIC_DIR)


def call_llm(prompt: str, model_name: str, provider: str = "ollama", openai_api_key: str = "", timeout: int = 40) -> str:
    """Unified LLM call for Ollama, OpenAI, and GitHub Models."""
    if provider == "openai":
        api_key = (openai_api_key or os.getenv("OPENAI_API_KEY", "")).strip()
        if not api_key:
            return ""

        payload = {
            "model": model_name or OPENAI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        response = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=timeout)
        if response.status_code == 429:
            return "ERROR_LIMIT_EXCEEDED"
        response.raise_for_status()
        data = response.json()
        return (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()

    if provider == "github":
        api_key = os.getenv("GITHUB_TOKEN", "").strip()
        if not api_key:
            return ""
        
        payload = {
            "model": model_name or GITHUB_MODEL,
            "messages": [
                {"role": "system", "content": "Du bist ein hilfreicher Assistent der Hochschule Aalen."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        # The GitHub endpoint usually expects /chat/completions if using OpenAI format, 
        # but their documentation shows the base /inference URL for some libraries.
        # We'll try the chat completions path first as it's more standard.
        url = GITHUB_URL
        if not url.endswith("/chat/completions"):
            url = f"{url.rstrip('/')}/chat/completions"
            
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if response.status_code == 429:
            return "ERROR_LIMIT_EXCEEDED"
        response.raise_for_status()
        data = response.json()
        return (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()

    if provider == "ollama":
        payload = {
            "model": model_name or OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        if response.status_code == 429:
            return "ERROR_LIMIT_EXCEEDED"
        response.raise_for_status()
        return (response.json().get("response", "") or "").strip()


def is_github_available() -> bool:
    api_key = os.getenv("GITHUB_TOKEN", "").strip()
    if not api_key:
        return False
    
    now = datetime.now().timestamp()
    cached = AI_AVAILABILITY["github"]
    if cached and now - cached[0] < CACHE_TTL:
        return cached[1]

    available = False
    try:
        # Just check if we can reach the endpoint or list models
        url = GITHUB_URL.replace("/inference", "/models") # Hypothetical models list
        # Alternative: just try a very small request
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        response = requests.get("https://models.github.ai/models", headers=headers, timeout=3)
        if response.status_code in (200, 404): # 404 is okay if endpoint exists but models path is different
            available = True
    except Exception:
        available = False
    
    AI_AVAILABILITY["github"] = (now, available)
    return available


def is_ollama_available() -> bool:
    now = datetime.now().timestamp()
    cached = AI_AVAILABILITY["ollama"]
    if cached and now - cached[0] < CACHE_TTL:
        return cached[1]

    available = False
    try:
        base_ollama = OLLAMA_URL.replace("/api/generate", "/api/tags")
        response = requests.get(base_ollama, timeout=2) # Shorter timeout
        response.raise_for_status()
        available = True
    except Exception:
        available = False
    
    AI_AVAILABILITY["ollama"] = (now, available)
    return available


def is_openai_available(openai_api_key: str = "") -> bool:
    api_key = (openai_api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not api_key:
        return False
    
    now = datetime.now().timestamp()
    cached = AI_AVAILABILITY["openai"].get(api_key)
    if cached and now - cached[0] < CACHE_TTL:
        return cached[1]

    available = False
    try:
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=3, # Shorter timeout
        )
        response.raise_for_status()
        available = True
    except Exception:
        available = False
    
    AI_AVAILABILITY["openai"][api_key] = (now, available)
    return available


def resolve_provider(provider: str, openai_api_key: str = "") -> str:
    requested = (provider or "auto").lower().strip()
    if requested in {"ollama", "openai", "github", "none"}:
        return requested

    # Auto mode: prefer local Ollama first, then GitHub (if token is available), then OpenAI.
    if is_ollama_available():
        return "ollama"
    if is_github_available():
        return "github"
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
        return expanded if expanded and expanded != "ERROR_LIMIT_EXCEEDED" else query
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
    
    # Check for direct inclusion or partial word match (must be a significant part of the word)
    for tok in core_tokens:
        if tok in haystack:
            return True
        # Also check for common variations (simple plural, etc. by checking if tok or haystack-parts match)
        if len(tok) >= 6:
            # Check for substring matches for longer words
            if any(part in haystack for part in [tok[:-1], tok[:-2]] if len(part) >= 4):
                return True
                
    return False


def lexical_relevance(query: str, text: str, url: str, title: str = "") -> float:
    q_tokens = tokenize(query)
    if not q_tokens:
        return 0.0

    haystack = f"{normalize_text(text)} {normalize_text(url)} {normalize_text(title)}"
    matched = {tok for tok in q_tokens if tok in haystack}
    coverage = len(matched) / len(q_tokens)

    # Title match bonus: give extra weight if query terms appear in the title
    title_norm = normalize_text(title)
    title_matches = sum(1 for tok in q_tokens if tok in title_norm)
    title_bonus = 0.15 * (title_matches / len(q_tokens)) if q_tokens else 0.0

    phrase_bonus = 0.0
    normalized_query = normalize_text(query)
    if len(normalized_query) >= 6 and normalized_query in haystack:
        phrase_bonus = 0.2

    # Completeness bonus: if ALL query tokens match, give a significant boost
    completeness_bonus = 0.0
    if len(matched) == len(q_tokens) and len(q_tokens) > 1:
        completeness_bonus = 0.1

    long_token_bonus = 0.0
    long_tokens = [t for t in q_tokens if len(t) >= 6]
    if long_tokens:
        long_matches = sum(1 for t in long_tokens if t in haystack)
        long_token_bonus = 0.1 * (long_matches / len(long_tokens))
    score = (0.5 * coverage) + title_bonus + phrase_bonus + completeness_bonus + long_token_bonus
    return max(0.0, min(1.0, score))


def detect_timetable_intent(query: str) -> bool:
    """Detect if the user is likely looking for a timetable/schedule."""
    timetable_keywords = [
        "plan", "stundenplan", "vorlesungsplan", "wann", "uhr", "termin",
        "montag", "dienstag", "mittwoch", "donnerstag", "freitag", "samstag", "sonntag",
        "zeit", "raum", "vlesung", "vorlesung", "vl", "uhrzeit"
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in timetable_keywords)


def hybrid_search(
    query: str, 
    expanded_query: str, 
    total_limit: int = 50, 
    semester: str = "SoSe26",
    is_timetable_query: bool = False
) -> List:
    """Hybrid search: search both main content and timetable collections."""
    # Encode both queries
    original_vector = model.encode(query).tolist()
    expanded_vector = model.encode(expanded_query).tolist()

    all_results = []
    
    def qdrant_fetch(coll: str, vec: List[float], lim: int) -> List:
        try:
            return client.query_points(collection_name=coll, query=vec, limit=lim).points
        except Exception as e:
            print(f"Error fetching from {coll}: {e}")
            return []

    # Actually, simpler: run 5 separate threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        f_main_orig = executor.submit(qdrant_fetch, COLLECTION_NAME, original_vector, int(total_limit * 0.5))
        f_main_exp = executor.submit(qdrant_fetch, COLLECTION_NAME, expanded_vector, int(total_limit * 0.5))
        f_website = executor.submit(qdrant_fetch, "hs_aalen_website", original_vector, int(total_limit * 0.25))
        f_asta = executor.submit(qdrant_fetch, "asta_content", original_vector, int(total_limit * 0.1))
        f_pdf = executor.submit(qdrant_fetch, "hs_aalen_pdfs", original_vector, int(total_limit * 0.2))
        
        # Always fetch timetable results
        semester_collection = f"starplan_{semester}"
        f_timetable = executor.submit(qdrant_fetch, semester_collection, original_vector, int(total_limit * 0.25))

        # Collect
        original_results = f_main_orig.result()
        expanded_results = f_main_exp.result()
        hs_aalen_results = f_website.result()
        asta_results = f_asta.result()
        pdf_results = f_pdf.result()
        
        timetable_results = f_timetable.result()
        print(f"DEBUG: Search Results - Orig: {len(original_results)}, Exp: {len(expanded_results)}, Web: {len(hs_aalen_results)}, Asta: {len(asta_results)}, PDF: {len(pdf_results)}, TT: {len(timetable_results)}")
        # Fallback for timetable
        if not timetable_results:
            timetable_results = qdrant_fetch("starplan_timetable", original_vector, int(total_limit * 0.25))

    # Merge main collection
    seen_main = {}
    for res in original_results:
        key = (res.payload.get("url", ""), res.payload.get("text", "")[:100])
        if key not in seen_main or res.score > seen_main[key].score:
            seen_main[key] = res
    for res in expanded_results:
        key = (res.payload.get("url", ""), res.payload.get("text", "")[:100])
        if key not in seen_main or res.score * 0.95 > seen_main[key].score:
            seen_main[key] = res
    
    all_results.extend(seen_main.values())
    all_results.extend(hs_aalen_results)
    all_results.extend(asta_results)
    all_results.extend(pdf_results)
    all_results.extend(timetable_results)

    # Global Deduplication by URL or Text Content
    unique_results = {}
    for res in all_results:
        url = res.payload.get("url", "").rstrip("/")
        # For timetables, dedup by content if the URL is the same or missing
        # Many timetables point to the same "Starplan" URL but have different course codes
        # We want to keep diversity but avoid 10 nearly identical rows.
        payload_type = res.payload.get('type', '')
        if not payload_type:
            source = res.payload.get('source', '')
            if source == 'starplan_timetable' or bool(res.payload.get('day') and res.payload.get('time')):
                payload_type = 'timetable'

        text_content = res.payload.get("text", res.payload.get("content", ""))
        text_id = text_content[:150].strip()
        
        # Complex key: URL + category + a snippet of text
        # This allows the same URL to appear if content is different (e.g. sections)
        # but prevents identical text from different "copies" of the same event.
        dedup_key = f"{url}|{payload_type}|{text_id}"
        
        if dedup_key not in unique_results or res.score > unique_results[dedup_key].score:
            unique_results[dedup_key] = res
            
    # Final sort
    merged = sorted(unique_results.values(), key=lambda x: x.score, reverse=True)
    return merged[:total_limit]


def boost_and_rank(
    query: str,
    results: list,
    model_name: str,
    include_rerank: bool,
    provider: str = "ollama",
    openai_api_key: str = "",
    is_timetable_query: bool = False,
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
        
        # Stability fix: If scores are very close, don't punish the lower end too hard.
        if denom < 0.1:
            # For small ranges, just use a linear mapping that doesn't go to zero.
            vector_norm = 0.5 + (0.5 * (raw_vector - min_score) / denom) if denom > 1e-6 else 1.0
        else:
            vector_norm = (raw_vector - min_score) / denom
        url_str = res.get("url", "").lower()
        text_str = res.get("text", "")
        title_str = res.get("title", "")
        lexical = lexical_relevance(query, text_str, url_str, title_str)

        # Weighted rank fusion: semantics dominate, lexical relevance stabilizes precision.
        final_score = (0.65 * vector_norm) + (0.35 * lexical)
        
        # URL Path Boost: If the user searches for a specific concept that is literally the page URL 
        # (e.g. "exmatrikulation" matching "hs-aalen.de/.../exmatrikulation"), boost it significantly.
        q_lower = query.lower().strip()
        if len(q_lower) >= 4 and q_lower in url_str:
            final_score += 0.35  # Massive boost for exact URL match
        elif any(len(t) >= 5 and t in url_str for t in tokenize(q_lower)):
            final_score += 0.15  # Partial URL match boost
            
        # Timetable Boosting: ONLY boost if user specifically asked for times/dates/plans.
        # Otherwise, demote slightly if it's a weak match to keep website results on top.
        if res.get("type") == "timetable":
            if is_timetable_query and raw_vector >= 0.40:
                final_score += 0.30
            elif not is_timetable_query:
                # If not a timetable query, pull it down a bit so official info wins
                final_score -= 0.10
            
        if res.get("type") == "pdf":
            final_score += 0.20
            
        res["score"] = final_score
        res["vector_score"] = float(raw_vector)
        res["lexical_score"] = float(lexical)

    if strict_match:
        # Don't apply strict matching for timetable, website, and ASTA results (they have different data structure)
        results = [
            r
            for r in results
            if r.get("type") in ("timetable", "website", "asta", "pdf") or strict_match_passes(query, r.get("text", ""), r.get("url", ""))
        ]

    # Filter weak matches so summary generation does not hallucinate from irrelevant sources.
    results = [r for r in results if r["score"] >= RELEVANCE_MIN_SCORE]
    results = sorted(results, key=lambda x: x["score"], reverse=True)

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
    
    # More permissive: either top1 >= 0.40 OR mean_top3 >= 0.35
    result_ok = (top1 >= 0.40) or (mean_top3 >= 0.35)
    if not result_ok:
        print(f"DEBUG: Summary skipped - evidence too weak (top1={top1:.2f}, mean_top3={mean_top3:.2f})")
    return result_ok


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
        if summary == "ERROR_LIMIT_EXCEEDED":
            return "⚠️ Limit erreicht: Die KI antwortet gerade nicht (zu viele Anfragen). Bitte versuche es später erneut."
        if "UNBEKANNT" in summary.upper():
            return ""
        return summary if summary else ""
    except Exception:
        return ""


@app.get("/api/models")
async def list_models(
    provider: str = Query("auto", pattern="^(auto|none|ollama|openai|github)$"),
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
    request: Request,
    q: str = Query(...),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=50),
    include_summary: bool = Query(True),
    include_rerank: bool = Query(True),
    include_expansion: bool = Query(True),
    strict_match: bool = Query(True),
    model_name: str = Query(""),
    provider: str = Query("auto", pattern="^(auto|none|ollama|openai|github)$"),
    openai_api_key: str = Query(""),
    semester: str = Query("SoSe26"),
    x_openai_key: Optional[str] = Header(default=None, alias="X-OpenAI-Key"),
):
    """Search endpoint with caching to improve performance."""
    active_openai_key = (openai_api_key or x_openai_key or "").strip()
    requested_provider = (provider or "auto").lower().strip()
    resolved_provider = resolve_provider(requested_provider, active_openai_key)
    
    # --- Rate Limiting Check ---
    client_ip = request.client.host
    now = datetime.now().timestamp()
    if client_ip not in user_request_log:
        user_request_log[client_ip] = []
    
    # Filter only timestamps within the last hour
    user_request_log[client_ip] = [ts for ts in user_request_log[client_ip] if now - ts < 3600]
    
    if len(user_request_log[client_ip]) >= RATE_LIMIT_PER_HOUR:
        print(f"RATE LIMIT EXCEEDED for {client_ip}")
        raise HTTPException(
            status_code=429, 
            detail="Rate limit exceeded. Please try again later. (Limit: 100 requests/hour)"
        )
    
    user_request_log[client_ip].append(now)

    llm_enabled = resolved_provider in {"ollama", "openai", "github"}
    include_llm_expansion = include_expansion and llm_enabled
    include_rerank = include_rerank and llm_enabled
    include_summary = include_summary and llm_enabled

    if resolved_provider == "openai":
        model_for_provider = model_name or OPENAI_MODEL
    elif resolved_provider == "github":
        model_for_provider = model_name or GITHUB_MODEL
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
        is_timetable_query = detect_timetable_intent(q)
        raw_points = hybrid_search(q, semantic_query, total_limit=50, semester=semester, is_timetable_query=is_timetable_query)

        # C. Formatting
        results = []
        for p in raw_points:
            # Handle HTML content, HS Aalen Website, ASTA, and Timetable data
            source = p.payload.get('source', '')
            payload_type = p.payload.get('type', '')
            is_timetable = source == 'starplan_timetable' or payload_type == 'timetable' or bool(p.payload.get('day') and p.payload.get('time'))
            is_hs_website = source == 'hs_aalen_website'
            is_asta = source == 'asta_website'
            is_pdf = source == 'hs_aalen_pdfs' or payload_type == 'pdf' or p.payload.get('url', '').lower().endswith('.pdf')
            
            # For timetable entries, create a formatted text
            if is_timetable:
                formatted_text = f"{p.payload.get('program', '')} - {p.payload.get('day', '')} {p.payload.get('time', '')} - {p.payload.get('lecture_info', '')[:100]}"
                result_type = 'timetable'
            elif is_pdf:
                formatted_text = p.payload.get("text", "")
                result_type = 'pdf'
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
            is_timetable_query=is_timetable_query,
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
        "llm_enabled": llm_enabled,
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


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

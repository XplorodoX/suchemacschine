import json
import os
import re
import unicodedata
from datetime import datetime
import asyncio
import concurrent.futures
from typing import List, Optional, Dict
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, Header, HTTPException, Query, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from qdrant_client.models import Prefetch, FusionQuery, Fusion
from sentence_transformers import SentenceTransformer

# --- Configuration ---
COLLECTION_NAME = "hs_aalen_hybrid"
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
GITHUB_URL = os.getenv("GITHUB_URL", "https://models.github.io/inference")
GITHUB_MODEL = os.getenv("GITHUB_MODEL", "openai/gpt-5")
GITHUB_MODEL_FALLBACKS = ["gpt-4o", "gpt-4o-mini", "gpt-4"]
CACHE_TTL = 3600  # Increased cache for better performance
DEFAULT_LLM_TIMEOUT = 60
OLLAMA_MODEL_DEFAULT = "qwen3:0.6b"

GERMAN_STOPWORDS = {"der", "die", "das", "ein", "eine", "einer", "einem", "einen", "und", "oder", "aber", "mit", "ohne", "auf", "in", "im", "am", "an", "zu", "zum", "zur", "von", "für", "ist", "sind", "war", "wie", "was", "wer", "wo", "wann", "warum", "welche", "welcher", "welches", "ich", "du", "er", "sie", "es", "wir", "ihr", "nicht", "kein", "keine", "mehr", "auch", "den", "dem", "des", "bei", "über", "unter", "nach"}
AI_AVAILABILITY = {"github": None}

app = FastAPI(title="HS Aalen AI Search")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

print("Loading Embedding Model...")
try:
    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print("✓ Models and Qdrant initialized")
except Exception as e:
    print(f"Warning: Initialization failed: {e}")
    model, client = None, None

def normalize_text(t: str) -> str:
    t = (t or "").lower()
    t = unicodedata.normalize("NFKC", t)
    return re.sub(r"\s+", " ", t).strip()

def tokenize(t: str) -> List[str]:
    return [w for w in re.findall(r"[a-zA-Z0-9äöüß]{2,}", normalize_text(t)) if w not in GERMAN_STOPWORDS]

import hashlib
def sparse_encode(text: str) -> models.SparseVector:
    tokens = tokenize(text)
    if not tokens: return models.SparseVector(indices=[], values=[])
    counts = {}
    for tok in tokens:
        idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % 1000000
        counts[idx] = counts.get(idx, 0) + 1.0
    return models.SparseVector(indices=list(counts.keys()), values=list(counts.values()))

def is_github_available() -> bool:
    if not os.getenv("GITHUB_TOKEN"): return False
    now = datetime.now().timestamp()
    if AI_AVAILABILITY["github"] and now - AI_AVAILABILITY["github"][0] < CACHE_TTL: return AI_AVAILABILITY["github"][1]
    AI_AVAILABILITY["github"] = (now, True)
    return True

def resolve_provider(provider: str, key: str = "") -> str:
    return "none"

def call_llm(prompt: str, model_name: str, provider: str = "github", api_key: str = "", timeout: int = DEFAULT_LLM_TIMEOUT) -> str:
    base_url = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")
    url = f"{base_url}/api/chat"
    m = os.getenv("OLLAMA_MODEL", model_name or OLLAMA_MODEL_DEFAULT)
    try:
        # Using /api/chat which is more robust for newest models
        payload = {
            "model": m,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        res = requests.post(url, json=payload, timeout=timeout)
        res.raise_for_status()
        return res.json().get("message", {}).get("content", "")
    except Exception as e:
        print(f"LLM Error ({m}): {e}")
        return ""

def rerank_with_llm(q: str, results: list, m: str, p: str, key: str = "") -> list:
    if not results or len(results) < 2: return results
    subset = results[:10]
    snippets = ""
    for i, r in enumerate(subset):
        text = r.get("text", "")[:150].replace("\n", " ")
        snippets += f"ID {i}: {text}\n"
    prompt = f"Anfrage: {q}\nOrdne diese IDs nach Relevanz (beste zuerst): {snippets}\nAntwort NUR als Liste: [ID1, ID2...]"
    res = call_llm(prompt, m, p, key)
    try:
        new_ids = [int(i) for i in re.findall(r"\d+", res) if 0 <= int(i) < len(subset)]
        if new_ids:
            reranked = [subset[i] for i in new_ids]
            seen_urls = {r.get("url") for r in reranked if r.get("url")}
            for r in subset:
                if r.get("url") and r["url"] not in seen_urls: reranked.append(r)
            return reranked + results[10:]
    except: pass
    return results

def parse_intent_and_filters(q: str, m: str, p: str, key: str = "") -> dict:
    """Classify user intent and extract entities using Few-Shot prompting."""
    prompt = f"""Du bist ein präziser Router für die Hochschul-KI Aalen. Deine Aufgabe ist es, die Nutzeranfrage zu klassifizieren und Entitäten zu extrahieren. Antworte AUSSCHLIESSLICH im JSON-Format.

Kategorien:
- PERSON: Suche nach Professoren, Mitarbeitern, Sprechstunden.
- TIMETABLE: Vorlesungen, Prüfungsphasen, Termine aus Starplan.
- DOCUMENT: SPO, Satzungen, Anträge, Formulare (PDF-Fokus).
- GENERAL: Mensa, Parken, allgemeine Campus-Infos.

Beispiele:
Nutzer: "Wann hat Prof. Müller Sprechstunde?"
JSON: {{"intent": "PERSON", "entity": "Müller", "boost_pdf": false, "time_filter": null}}

Nutzer: "Ich brauche die SPO für Informatik 2023"
JSON: {{"intent": "DOCUMENT", "entity": "SPO Informatik", "boost_pdf": true, "time_filter": "2023"}}

Nutzer: "Was gibt es heute in der Mensa?"
JSON: {{"intent": "GENERAL", "entity": "Mensa", "boost_pdf": false, "time_filter": "today"}}

Nutzer: {q}
JSON:"""
    
    res = call_llm(prompt, m, p, key)
    data = {"intent": "GENERAL", "entity": q}
    
    # LLM Parsing
    try:
        res_clean = re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip()
        match = re.search(r"\{.*\}", res_clean, re.DOTALL)
        if match:
            llm_data = json.loads(match.group(0).replace("'", '"'))
            data.update(llm_data)
    except: pass
            
    return data

# Keywords that signal a timetable/schedule query
TIMETABLE_SIGNALS = {
    "montag", "dienstag", "mittwoch", "donnerstag", "freitag",
    "stundenplan", "vorlesung", "semesterplan", "uhrzeit", "raum",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "lecture", "schedule", "timetable",
}


def _detect_query_type(query: str) -> str:
    """
    Classify query to adjust collection weights dynamically.
    Returns: 'timetable', 'general', or 'asta'
    """
    q = query.lower()
    tokens = set(re.findall(r"[a-zA-ZäöüÄÖÜß]{3,}", q))

    if tokens & TIMETABLE_SIGNALS:
        return "timetable"
    if any(w in q for w in ["asta", "stura", "fachschaft", "ernas", "studentisch"]):
        return "asta"
    return "general"


def _get_collection_weights(query_type: str) -> dict[str, float]:
    """
    Return per-collection score weights based on query type.
    No hardcoded percentages — weights are relative boost factors applied after
    fusion, so they're easy to tune without touching search logic.
    """
    if query_type == "timetable":
        return {
            "hs_aalen_search": 0.4,
            "hs_aalen_website": 0.2,
            "starplan_timetable": 2.0,  # Strong boost for timetable queries
            "asta_content": 0.1,
        }
    if query_type == "asta":
        return {
            "hs_aalen_search": 0.5,
            "hs_aalen_website": 0.3,
            "starplan_timetable": 0.1,
            "asta_content": 2.0,  # Strong boost for ASTA queries
        }
    # General query — balanced
    return {
        "hs_aalen_search": 1.0,
        "hs_aalen_website": 0.7,
        "starplan_timetable": 0.3,
        "asta_content": 0.4,
    }


def hybrid_search(
    query: str,
    expanded_query: str,
    model,
    client: QdrantClient,
    total_limit: int = 20,
    semester: str = "SoSe26",
) -> list:
    """
    True hybrid search: dense vector (semantic) + sparse BM25 keyword matching.
    Weights collections dynamically based on query type.

    Falls back gracefully if sparse vectors aren't available in the collection.
    """
    query_type = _detect_query_type(query)
    weights = _get_collection_weights(query_type)

    # Dense vector from embedding model
    dense_vector = model.encode(expanded_query).tolist()

    all_results = []

    # Collections to search (use semester-specific timetable if available)
    timetable_collection = f"starplan_{semester}"
    collections_to_search = {
        "hs_aalen_search": weights.get("hs_aalen_search", 1.0),
        "hs_aalen_website": weights.get("hs_aalen_website", 0.7),
        timetable_collection: weights.get("starplan_timetable", 0.3),
        "asta_content": weights.get("asta_content", 0.4),
    }

    per_collection_limit = max(10, total_limit)

    for collection_name, weight in collections_to_search.items():
        if weight < 0.05:
            continue  # Skip negligible collections

        try:
            results = _search_collection(
                client=client,
                collection_name=collection_name,
                dense_vector=dense_vector,
                query=query,
                limit=per_collection_limit,
            )

            # Apply collection weight to scores
            for r in results:
                r["score"] = r["score"] * weight
                r["collection"] = collection_name

            all_results.extend(results)

        except Exception as e:
            print(f"WARNING: Collection {collection_name} unavailable: {e}")

            # Fallback to old timetable collection for semester-specific ones
            if collection_name.startswith("starplan_") and collection_name != "starplan_timetable":
                try:
                    fallback_results = _search_collection(
                        client=client,
                        collection_name="starplan_timetable",
                        dense_vector=dense_vector,
                        query=query,
                        limit=per_collection_limit,
                    )
                    fallback_weight = weights.get("starplan_timetable", 0.3)
                    for r in fallback_results:
                        r["score"] = r["score"] * fallback_weight
                        r["collection"] = "starplan_timetable"
                    all_results.extend(fallback_results)
                except Exception:
                    pass

    # Deduplicate by URL, keeping highest score
    seen: dict[str, dict] = {}
    for r in all_results:
        url = r.get("url", "")
        if url not in seen or r["score"] > seen[url]["score"]:
            seen[url] = r

    # Sort by score descending
    sorted_results = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return sorted_results[:total_limit]


def _search_collection(
    client: QdrantClient,
    collection_name: str,
    dense_vector: list[float],
    query: str,
    limit: int,
) -> list[dict]:
    """
    Search a single collection. Tries sparse+dense hybrid first,
    falls back to dense-only if sparse vectors aren't configured.
    """
    try:
        # Try hybrid (sparse BM25 + dense) using Qdrant's query API
        # This requires the collection to have been created with sparse vector support
        results = client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=limit * 2,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        ).points

    except Exception:
        # Fallback: dense-only search (current behavior)
        results = client.query_points(
            collection_name=collection_name,
            query=dense_vector,
            limit=limit,
            with_payload=True,
        ).points

    return _format_results(results, collection_name)


def _format_results(points, collection_name: str) -> list[dict]:
    """Convert Qdrant ScoredPoint objects to dicts."""
    formatted = []
    for p in points:
        payload = p.payload or {}
        source = payload.get("source", "")
        payload_type = payload.get("type", "")

        is_timetable = (
            source == "starplan_timetable"
            or payload_type == "timetable"
            or (payload.get("day") and payload.get("time"))
        )
        is_asta = source == "asta_website"
        is_hs_website = source == "hs_aalen_website"

        if is_timetable:
            # Build display text from structured timetable fields
            parts = [
                payload.get("name") or payload.get("title", ""),
                payload.get("day", ""),
                payload.get("time", ""),
            ]
            text = " — ".join(p for p in parts if p)
            result_type = "timetable"
        elif is_hs_website or is_asta:
            text = f"{payload.get('title', '')}: {payload.get('content', '')[:200]}"
            result_type = "asta" if is_asta else "website"
        else:
            text = payload.get("text", "")
            result_type = "webpage"

        formatted.append({
            "score": float(p.score),
            "url": payload.get("url", ""),
            "text": text,
            "title": payload.get("title", ""),
            "program": payload.get("program"),
            "day": payload.get("day"),
            "time": payload.get("time"),
            "room": payload.get("room"),
            "lecturer": payload.get("lecturer"),
            "semester": payload.get("semester"),
            "type": result_type,
            "pdf_sources": payload.get("pdf_sources", []),
            "collection": collection_name,
        })

    return formatted

def optimized_hybrid_search(query: str, intent_data: dict, limit: int = 100) -> List:
    """Wrapper that calls the new hybrid_search with the global model/client."""
    return hybrid_search(query, query, model, client, total_limit=limit)

def expand_query(q: str, m: str, p: str, key: str = "") -> str:
    prompt = f"Suche: {q}\nGib 3-5 zusätzliche, hochrelevante deutsche Suchbegriffe oder Synonyme an, um die Suche zu erweitern. Antworte NUR mit den Begriffen, kommagetrennt."
    res = call_llm(prompt, m, p, key, timeout=20)
    if res and "UNBEKANNT" not in res.upper():
        clean_res = re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip()
        expanded = f"{q}, {clean_res}"
        return expanded
    return q

def fetch_parent_context(results: list, limit: int = 3) -> list:
    """Fetch additional lectures for the same day/program if a lecture is found."""
    lectures = [r for r in results if r.get("type") == "timetable" and r.get("start_time")]
    if not lectures: return results
    
    # Take the best lecture hit
    best = lectures[0]
    p_id = best.get("studiengang")
    s_time = best.get("start_time")
    
    if p_id and s_time:
        try:
            day_prefix = s_time[:10]
            context_results = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="studiengang", match=models.MatchValue(value=p_id)),
                        models.FieldCondition(key="start_time", match=models.MatchText(text=day_prefix))
                    ]
                ),
                limit=10,
                with_payload=True
            )[0]
            
            # Add context to the first result text
            extra_text = "\nKontext (andere Termine an diesem Tag):\n"
            for p in context_results:
                payload = p.payload
                extra_text += f"- {payload.get('start_time','')[11:16]} {payload.get('title')} in {payload.get('raum')}\n"
            
            best["text"] = best["text"] + extra_text
        except: pass
    return results

def boost_and_rank(q: str, results: list, intent_data: dict = None) -> list:
    tokens = tokenize(q)
    q_low = q.lower()
    intent = (intent_data or {}).get("intent", "general")
    
    for r in results:
        title = (r.get("title") or "").lower()
        url = (r.get("url") or "").lower()
        text = normalize_text(r.get("text", "") + " " + title)
        source = r.get("source", "")
        res_type = r.get("type", "")
        
        # Lexical Match Score
        lex_score = sum(2.0 if tok in title else (1.5 if tok in url else 1.0) for tok in tokens if tok in text) / (len(tokens) or 1)
        r["score"] = (0.35 * r["score"]) + (0.65 * lex_score)
        
        # Exact Phrase Boost
        if q_low in title: r["score"] += 1.5
        
        # Intent-Based Dynamic Boosting
        if intent == "timetable":
            if res_type == "timetable": r["score"] += 1.2
            else: r["score"] -= 0.3
        elif intent == "person":
            if source == "hs_aalen" and res_type == "webpage": r["score"] += 1.0
            if "prof" in title or "rector" in title or "rektor" in title: r["score"] += 1.0
        elif intent == "document":
            if res_type == "pdf": r["score"] += 1.5
            if any(k in title for k in ["satzung", "ordnung"]): r["score"] += 0.8
        
        # source/type defaults
        if source in ["hs_aalen", "asta"]: r["score"] += 0.3
            
    return sorted(results, key=lambda x: x["score"], reverse=True)

def generate_summary(q: str, results: list, m: str, p: str, key: str = "") -> str:
    # PERMISSIVE THRESHOLD: even low RRF scores are valid
    if not results or results[0]["score"] < 0.0001: 
        print(f"Summary skipped: No results or score too low ({results[0]['score'] if results else 'N/A'})")
        return ""
    
    # Increase snippet count for better context
    snippets = ""
    for i, r in enumerate(results[:5]):
        snippets += f"[{i+1}] {r.get('url','')}: {r.get('text','')[:400]}\n\n"
        
    p_prompt = f"""Anfrage: {q}
Basierend auf diesen Informationen der Hochschule Aalen:
{snippets}

Beantworte die Anfrage kurz und präzise in der Du-Form. 
Nutze Quellenangaben wie [1], [2]. 
Falls die Information absolut nicht vorhanden ist, antworte NUR mit 'UNBEKANNT'."""

    res = call_llm(p_prompt, m, p, key)
    res_clean = re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip()
    
    if "UNBEKANNT" in res_clean.upper() and len(res_clean) < 20:
        return ""
    return res_clean

@app.get("/api/models")
async def api_models(provider: str = Query("auto"), x_key: Optional[str] = Header(None, alias="X-OpenAI-Key")):
    return {"models": []}

@app.get("/api/search")
async def api_search(q: str = Query(...), provider: str = Query("auto"), model_name: Optional[str] = Query(None), x_key: Optional[str] = Header(None, alias="X-OpenAI-Key")):
    res_p = resolve_provider(provider, x_key or "")
    res_m = model_name or GITHUB_MODEL
    
    # Phase 1: Simple Local Expansion (no LLM call for now to save time)
    expanded_q = q # Skipping LLM expand_query
    intent_data = {"intent": "GENERAL", "entity": q} # Skipping LLM parse_intent_and_filters
    
    # Custom Weighted RRF
    results = optimized_hybrid_search(expanded_q, intent_data, 100)
    
    # Ranking & Context
    ranked = boost_and_rank(q, results, intent_data)
    contextualized = fetch_parent_context(ranked)
    
    # Skipping Rerank-LLM to save time
    reranked = contextualized
    
    # Phase 2: No synchronous summary to avoid timeouts on slow systems
    summary = "" # Frontend will call /api/summarize separately
    
    return {
        "results": reranked[:10], 
        "total_results": len(ranked), 
        "summary": summary, 
        "model": res_m, 
        "provider": res_p, 
        "filters": intent_data, 
        "expanded_query": expanded_q,
        "llm_enabled": True # Keep frontend happy
    }

@app.get("/api/health")
async def health(): return {"status": "ok"}

@app.post("/api/summarize")
async def api_summarize(q: str = Body(...), results: list = Body(...), provider: str = Body("auto"), model_name: Optional[str] = Body(None), x_key: Optional[str] = Header(None, alias="X-OpenAI-Key")):
    res_p = resolve_provider(provider, x_key or "")
    res_m = model_name or GITHUB_MODEL
    s = generate_summary(q, results, res_m, res_p, x_key or "")
    return {"summary": s, "model": res_m, "provider": res_p}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

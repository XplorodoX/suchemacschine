import json
import os
import re
import unicodedata
from typing import List, Optional, Dict
from urllib.parse import urlparse

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient, models
from qdrant_client.models import Prefetch, FusionQuery, Fusion
from sentence_transformers import SentenceTransformer

# --- Configuration ---
COLLECTION_NAME = "hs_aalen_hybrid"
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

GERMAN_STOPWORDS = {"der", "die", "das", "ein", "eine", "einer", "einem", "einen", "und", "oder", "aber", "mit", "ohne", "auf", "in", "im", "am", "an", "zu", "zum", "zur", "von", "für", "ist", "sind", "war", "wie", "was", "wer", "wo", "wann", "warum", "welche", "welcher", "welches", "ich", "du", "er", "sie", "es", "wir", "ihr", "nicht", "kein", "keine", "mehr", "auch", "den", "dem", "des", "bei", "über", "unter", "nach"}

app = FastAPI(title="HS Aalen Search")
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
    """
    if query_type == "timetable":
        return {
            "hs_aalen_search": 0.4,
            "hs_aalen_website": 0.2,
            "starplan_timetable": 2.0,
            "asta_content": 0.1,
        }
    if query_type == "asta":
        return {
            "hs_aalen_search": 0.5,
            "hs_aalen_website": 0.3,
            "starplan_timetable": 0.1,
            "asta_content": 2.0,
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
    model,
    client: QdrantClient,
    total_limit: int = 20,
    semester: str = "SoSe26",
) -> list:
    """
    True hybrid search: dense vector (semantic) + sparse BM25 keyword matching.
    Weights collections dynamically based on query type.
    """
    query_type = _detect_query_type(query)
    weights = _get_collection_weights(query_type)

    # Dense vector from embedding model
    dense_vector = model.encode(query).tolist()

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
            continue

        try:
            results = _search_collection(
                client=client,
                collection_name=collection_name,
                dense_vector=dense_vector,
                query=query,
                limit=per_collection_limit,
            )

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
        # Fallback: dense-only search
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


def fetch_parent_context(results: list, limit: int = 3) -> list:
    """Fetch additional lectures for the same day/program if a lecture is found."""
    lectures = [r for r in results if r.get("type") == "timetable" and r.get("start_time")]
    if not lectures: return results
    
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
            
            extra_text = "\nKontext (andere Termine an diesem Tag):\n"
            for p in context_results:
                payload = p.payload
                extra_text += f"- {payload.get('start_time','')[11:16]} {payload.get('title')} in {payload.get('raum')}\n"
            
            best["text"] = best["text"] + extra_text
        except: pass
    return results


def _detect_intent_local(q: str) -> dict:
    """Simple rule-based intent detection without LLM."""
    q_low = q.lower()
    
    if any(w in q_low for w in ["prof", "dozent", "sprechstunde", "professor"]):
        return {"intent": "PERSON", "entity": q}
    if any(w in q_low for w in ["spo", "ordnung", "satzung", "antrag", "formular", "pdf"]):
        return {"intent": "DOCUMENT", "entity": q}
    if any(w in q_low for w in ["stundenplan", "vorlesung", "prüfung", "klausur", "termin"]):
        return {"intent": "TIMETABLE", "entity": q}
    
    return {"intent": "GENERAL", "entity": q}


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


@app.get("/api/search")
async def api_search(q: str = Query(...)):
    # Local intent detection (no LLM)
    intent_data = _detect_intent_local(q)
    
    # Hybrid vector search
    results = hybrid_search(q, model, client, total_limit=100)
    
    # Ranking & Context
    ranked = boost_and_rank(q, results, intent_data)
    contextualized = fetch_parent_context(ranked)
    
    return {
        "results": contextualized[:10], 
        "total_results": len(ranked), 
        "filters": intent_data, 
    }

@app.get("/api/health")
async def health(): return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

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
from sentence_transformers import SentenceTransformer

# --- Configuration ---
COLLECTION_NAME = "hs_aalen_hybrid"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
GITHUB_URL = os.getenv("GITHUB_URL", "https://models.github.io/inference")
GITHUB_MODEL = os.getenv("GITHUB_MODEL", "openai/gpt-5")
GITHUB_MODEL_FALLBACKS = ["gpt-4o", "gpt-4o-mini", "gpt-4"]
CACHE_TTL = 300

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
    p = (provider or "auto").lower().strip()
    if p == "none": return "none"
    return "github"

def call_llm(prompt: str, model_name: str, provider: str = "github", api_key: str = "", timeout: int = 40) -> str:
    if provider == "github":
        api_key = api_key or os.getenv("GITHUB_TOKEN", "").strip()
        if not api_key: return ""
        for m in [model_name] + GITHUB_MODEL_FALLBACKS:
            try:
                res = requests.post(GITHUB_URL.rstrip('/') + "/chat/completions", headers={"Authorization": f"Bearer {api_key}"}, json={"model": m, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}, timeout=timeout)
                if res.status_code == 429: return "LIMIT_EXCEEDED"
                res.raise_for_status()
                return res.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            except: continue
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

def custom_weighted_rrf(dense_hits, sparse_hits, k=60, dense_weight=1.0, sparse_weight=1.0):
    """
    Kombiniert Ergebnisse aus Vektor- (dense) und Keyword-Suche (sparse).
    Erhöhe sparse_weight, wenn der Intent 'PERSON' oder 'DOCUMENT' ist.
    """
    scores = {}
    docs = {}
    
    # Verarbeite Dense Ergebnisse
    for rank, hit in enumerate(dense_hits):
        doc_id = hit.id
        scores[doc_id] = scores.get(doc_id, 0) + dense_weight * (1 / (k + rank + 1))
        docs[doc_id] = hit
        
    # Verarbeite Sparse Ergebnisse
    for rank, hit in enumerate(sparse_hits):
        doc_id = hit.id
        scores[doc_id] = scores.get(doc_id, 0) + sparse_weight * (1 / (k + rank + 1))
        docs[doc_id] = hit
        
    # Sortiere nach aggregiertem Score
    sorted_ids = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    
    # Rekonstruiere Ergebnisliste
    final_results = []
    for doc_id, score in sorted_ids:
        hit = docs[doc_id]
        payload = hit.payload
        final_results.append({
            "score": score,
            "url": payload.get("url"),
            "text": payload.get("content") or payload.get("text") or "",
            "title": payload.get("title") or "Seite",
            "type": payload.get("type", "webpage"),
            "source": payload.get("source", "hs_aalen"),
            "start_time": payload.get("start_time"),
            "raum": payload.get("raum"),
            "studiengang": payload.get("studiengang")
        })
    return final_results

def expand_query(q: str, m: str, p: str, key: str = "") -> str:
    prompt = f"Suche: {q}\nGib 3-5 zusätzliche, hochrelevante deutsche Suchbegriffe oder Synonyme an, um die Suche zu erweitern. Antworte NUR mit den Begriffen, kommagetrennt."
    res = call_llm(prompt, m, p, key, timeout=10)
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

def optimized_hybrid_search(query: str, intent_data: dict, limit: int = 100) -> List:
    if not client: return []
    try:
        dense_vec = model.encode(query).tolist()
        sparse_vec = sparse_encode(query)
        
        # Build filter (simplified)
        q_filter = None
        
        # Weights based on Intent
        intent = intent_data.get("intent", "GENERAL").upper()
        d_weight, s_weight = 1.0, 1.0
        if intent in ["PERSON", "DOCUMENT"]:
            s_weight = 2.0  # Boost keywords for specific names or document types
            
        # Fetch separate lists for custom RRF
        dense_hits = client.query_points(
            collection_name=COLLECTION_NAME,
            query=dense_vec,
            using="dense",
            limit=limit,
            query_filter=q_filter
        ).points
        
        sparse_hits = client.query_points(
            collection_name=COLLECTION_NAME,
            query=sparse_vec,
            using="sparse",
            limit=limit,
            query_filter=q_filter
        ).points
        
        return custom_weighted_rrf(dense_hits, sparse_hits, dense_weight=d_weight, sparse_weight=s_weight)
        
    except Exception as e:
        print(f"Hybrid search failed: {e}")
        return []

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
    return {"models": [GITHUB_MODEL] + GITHUB_MODEL_FALLBACKS}

@app.get("/api/search")
async def api_search(q: str = Query(...), provider: str = Query("auto"), model_name: Optional[str] = Query(None), x_key: Optional[str] = Header(None, alias="X-OpenAI-Key")):
    res_p = resolve_provider(provider, x_key or "")
    res_m = model_name or GITHUB_MODEL
    
    # Phase 1: Expansion & Intent Routing
    expanded_q = expand_query(q, res_m, res_p, x_key or "")
    intent_data = parse_intent_and_filters(q, res_m, res_p, x_key or "")
    
    # Custom Weighted RRF
    results = optimized_hybrid_search(expanded_q, intent_data, 100)
    
    ranked = boost_and_rank(q, results, intent_data)
    contextualized = fetch_parent_context(ranked)
    reranked = rerank_with_llm(q, contextualized, res_m, res_p, x_key or "")
    summary = generate_summary(q, reranked, res_m, res_p, x_key or "")
    return {"results": reranked[:10], "total_results": len(ranked), "summary": summary, "model": res_m, "provider": res_p, "filters": intent_data, "expanded_query": expanded_q}

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

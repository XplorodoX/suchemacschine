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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")
OPENAI_URL = os.getenv("OPENAI_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GITHUB_URL = os.getenv("GITHUB_URL", "https://models.github.io/inference")
GITHUB_MODEL = os.getenv("GITHUB_MODEL", "openai/gpt-5")
GITHUB_MODEL_FALLBACKS = ["gpt-4o", "gpt-4"]
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

def sparse_encode(text: str) -> models.SparseVector:
    tokens = tokenize(text)
    if not tokens: return models.SparseVector(indices=[], values=[])
    counts = {}
    for tok in tokens:
        idx = hash(tok) % 1000000
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
    if p in {"ollama", "openai", "github"}: return p
    if is_github_available(): return "github"
    if os.getenv("OPENAI_API_KEY") or key: return "openai"
    return "ollama"

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
    if provider == "openai":
        api_key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
        try:
            res = requests.post(OPENAI_URL, headers={"Authorization": f"Bearer {api_key}"}, json={"model": OPENAI_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}, timeout=timeout)
            res.raise_for_status()
            return res.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        except: pass
    if provider == "ollama":
        try:
            res = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=timeout)
            return res.json().get("response", "").strip()
        except: pass
    return ""

def rerank_with_llm(q: str, results: list, m: str, p: str, key: str = "") -> list:
    if not results or len(results) < 2: return results
    subset = results[:5]
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
            seen_urls = {r["url"] for r in reranked}
            for r in subset:
                if r["url"] not in seen_urls: reranked.append(r)
            return reranked + results[5:]
    except: pass
    return results

def parse_filters(q: str, m: str, p: str, key: str = "") -> dict:
    prompt = f"Suche: {q}\nFilter extrahieren. source (asta, website, pdf, starplan), semester. ZEIT: morgen/heute? NUR JSON: {{'source': '...', 'type': '...', 'start_range': 'ISO-DATE-START', 'end_range': 'ISO-DATE-END'}}"
    res = call_llm(prompt, m, p, key)
    filters = {}
    
    # Keyword Fallback
    ql = q.lower()
    if "asta" in ql: filters["source"] = "asta"
    if "pdf" in ql: filters["type"] = "pdf"
    if "starplan" in ql: filters["source"] = "starplan"
    if "heute" in ql or "jetzt" in ql:
        now = datetime.now()
        filters["start_range"] = int(now.replace(hour=0, minute=0, second=0).timestamp())
        filters["end_range"] = int(now.replace(hour=23, minute=59, second=59).timestamp())
    if "morgen" in ql:
        import datetime as dt
        tomorrow = datetime.now() + dt.timedelta(days=1)
        filters["start_range"] = int(tomorrow.replace(hour=0, minute=0, second=0).timestamp())
        filters["end_range"] = int(tomorrow.replace(hour=23, minute=59, second=59).timestamp())
    
    # LLM Parsing
    try:
        res_clean = re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip()
        match = re.search(r"\{.*\}", res_clean, re.DOTALL)
        if match:
            llm_f = json.loads(match.group(0).replace("'", '"'))
            # Convert LLM ISO dates to timestamps if they exist
            for k in ["start_range", "end_range"]:
                if k in llm_f and isinstance(llm_f[k], str):
                    try: 
                        dt_str = llm_f[k].replace('Z', '+00:00')
                        llm_f[k] = int(datetime.fromisoformat(dt_str).timestamp())
                    except: pass
            filters.update(llm_f)
    except: pass
    return filters

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

def hybrid_search(query: str, filters: dict = None, limit: int = 50) -> List:
    if not client: return []
    try:
        dense_vec = model.encode(query).tolist()
        sparse_vec = sparse_encode(query)
        
        # Build filter
        q_filter = None
        if filters:
            must = []
            if "source" in filters: must.append(models.FieldCondition(key="source", match=models.MatchValue(value=filters["source"])))
            if "type" in filters: must.append(models.FieldCondition(key="type", match=models.MatchValue(value=filters["type"])))
            if "start_range" in filters or "end_range" in filters:
                must.append(models.FieldCondition(
                    key="start_time", 
                    range=models.Range(
                        gte=filters.get("start_range"),
                        lte=filters.get("end_range")
                    )
                ))
            if must: q_filter = models.Filter(must=must)

        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=sparse_vec, using="sparse", limit=limit),
                models.Prefetch(query=dense_vec, using="dense", limit=limit),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=q_filter,
            limit=limit
        ).points
    except Exception as e:
        print(f"Hybrid search failed: {e}")
        dense_vec = model.encode(query).tolist()
        try: return client.query_points(collection_name="hs_aalen_search", query=dense_vec, limit=limit).points
        except: return []

def boost_and_rank(q: str, results: list) -> list:
    tokens = tokenize(q)
    for r in results:
        text = normalize_text(r.get("text", "") + " " + r.get("title", ""))
        lex_score = sum(1 for tok in tokens if tok in text) / (len(tokens) or 1)
        r["score"] = (0.7 * r["score"]) + (0.3 * lex_score)
        if q.lower() in r.get("url", "").lower(): r["score"] += 0.2
    return sorted(results, key=lambda x: x["score"], reverse=True)

def generate_summary(q: str, results: list, m: str, p: str, key: str = "") -> str:
    # PERMISSIVE THRESHOLD FOR RRF
    if not results or results[0]["score"] < 0.01: return ""
    snippets = ""
    for i, r in enumerate(results[:3]):
        snippets += f"[{i+1}] {r.get('url','')}: {r.get('text','')[:300]}\n\n"
    p_prompt = f"Beantworte {q} basierend auf:\n{snippets}\nRegeln: Kurz, Du-Form, Quellen [1], [2]. Falls keine Info: UNBEKANNT"
    res = call_llm(p_prompt, m, p, key)
    res = re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip()
    return "" if "UNBEKANNT" in res.upper() else res

@app.get("/api/search")
async def api_search(q: str = Query(...), provider: str = Query("auto"), x_key: Optional[str] = Header(None, alias="X-OpenAI-Key")):
    res_p = resolve_provider(provider, x_key or "")
    res_m = GITHUB_MODEL if res_p == "github" else (OPENAI_MODEL if res_p == "openai" else OLLAMA_MODEL)
    
    # Phase 4: Self-Querying
    filters = parse_filters(q, res_m, res_p, x_key or "")
    
    raw = hybrid_search(q, filters, 50)
    results = []
    for p in raw:
        payload = p.payload
        results.append({
            "score": float(p.score),
            "url": payload.get("url"),
            "text": payload.get("content") or payload.get("text") or "",
            "title": payload.get("title") or "Seite",
            "type": payload.get("type", "webpage"),
            "source": payload.get("source", "hs_aalen"),
            "start_time": payload.get("start_time"),
            "raum": payload.get("raum"),
            "studiengang": payload.get("studiengang")
        })
    ranked = boost_and_rank(q, results)
    contextualized = fetch_parent_context(ranked)
    reranked = rerank_with_llm(q, contextualized, res_m, res_p, x_key or "")
    summary = generate_summary(q, reranked, res_m, res_p, x_key or "")
    return {"results": reranked[:10], "total_results": len(ranked), "summary": summary, "model": res_m, "provider": res_p, "filters": filters}

@app.get("/api/health")
async def health(): return {"status": "ok"}

@app.post("/api/summarize")
async def api_summarize(q: str = Body(...), results: list = Body(...)):
    s = generate_summary(q, results, GITHUB_MODEL, "github")
    return {"summary": s}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import json
import os
import re
import unicodedata
import math
from typing import List, Optional, Dict
from urllib.parse import urlparse

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient, models
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from pydantic import BaseModel
import sqlite3

# --- Configuration ---
COLLECTION_NAME = "hs_aalen_hybrid"
# Dense embedding model. multilingual-e5-base (768-dim) gives noticeably better
# German retrieval than the old MiniLM (384-dim). NOTE: e5 models require a task
# prefix ("query: " for queries, "passage: " for documents) and the collections
# must be re-indexed with the *same* model/dimension.
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
USE_E5_PREFIX = "e5" in MODEL_NAME.lower()
# Cross-Encoder used to re-rank the top candidates jointly (query, document).
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
# How many top candidates to feed into the (relatively expensive) cross-encoder.
RERANK_POOL = int(os.getenv("RERANK_POOL", 30))
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

GERMAN_STOPWORDS = {"der", "die", "das", "ein", "eine", "einer", "einem", "einen", "und", "oder", "aber", "mit", "ohne", "auf", "in", "im", "am", "an", "zu", "zum", "zur", "von", "für", "ist", "sind", "war", "wie", "was", "wer", "wo", "wann", "warum", "welche", "welcher", "welches", "ich", "du", "er", "sie", "es", "wir", "ihr", "nicht", "kein", "keine", "mehr", "auch", "den", "dem", "des", "bei", "über", "unter", "nach"}

app = FastAPI(title="HS Aalen Search")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Database for NavBoost ---
DB_FILE = os.path.join(os.path.dirname(__file__), "navboost.db")

def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS navboost_stats (
                query TEXT,
                url TEXT,
                short_clicks INTEGER DEFAULT 0,
                long_clicks INTEGER DEFAULT 0,
                PRIMARY KEY (query, url)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_stats (
                query TEXT PRIMARY KEY,
                search_count INTEGER DEFAULT 1
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error initializing DB: {e}")

init_db()

class FeedbackRequest(BaseModel):
    query: str
    url: str
    type: str # "short_click" or "long_click"

print("Loading models...")
try:
    model = SentenceTransformer(MODEL_NAME)
    print(f"✓ Dense embedding model loaded ({MODEL_NAME})")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    print("✓ BM25 sparse model loaded")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print("✓ Qdrant connected")
except Exception as e:
    print(f"Warning: Initialization failed: {e}")
    model, sparse_model, client = None, None, None

# Cross-Encoder for re-ranking. Loaded separately so a failure here doesn't take
# down the whole API — search still works without it, just with weaker ordering.
cross_encoder = None
try:
    from sentence_transformers.cross_encoder import CrossEncoder
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    print(f"✓ Cross-Encoder loaded ({CROSS_ENCODER_MODEL})")
except Exception as e:
    print(f"Warning: Cross-Encoder unavailable, ranking will skip rerank step: {e}")


def encode_query(text: str) -> list:
    """Encode a search query into a dense vector, applying the e5 task prefix."""
    payload = f"query: {text}" if USE_E5_PREFIX else text
    return model.encode(payload).tolist()


def sparse_encode(text: str) -> SparseVector:
    """Encode text into a BM25 sparse vector using fastembed."""
    result = list(sparse_model.embed([text]))[0]
    return SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )


def normalize_text(t: str) -> str:
    t = (t or "").lower()
    t = unicodedata.normalize("NFKC", t)
    return re.sub(r"\s+", " ", t).strip()

def tokenize(t: str) -> List[str]:
    return [w for w in re.findall(r"[a-zA-Z0-9äöüß]{2,}", normalize_text(t)) if w not in GERMAN_STOPWORDS]

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
            "hs_aalen_pdfs": 0.1,   # NEU
        }
    if query_type == "asta":
        return {
            "hs_aalen_search": 0.5,
            "hs_aalen_website": 0.3,
            "starplan_timetable": 0.1,
            "asta_content": 2.0,
            "hs_aalen_pdfs": 0.3,   # NEU
        }
    # General query — balanced
    return {
        "hs_aalen_search": 1.0,
        "hs_aalen_website": 0.7,
        "starplan_timetable": 0.3,
        "asta_content": 0.6,        # War 0.4, erhöht für ASTA-Nischeninhalte
        "hs_aalen_pdfs": 0.8,       # NEU – PDFs werden jetzt durchsucht
    }
# Common university term synonyms for query expansion
PROGRAM_SYNONYMS = {
    "info": "informatik",
    "informatik": "computer science informationstechnologie",
    "wi": "wirtschaftsinformatik",
    "bwl": "betriebswirtschaftslehre",
    "mathe": "mathematik",
    "e-technik": "elektrotechnik",
    "et": "elektrotechnik",
    "mb": "maschinenbau",
    "mensa": "speiseplan essen cafeteria",
    "bib": "bibliothek",
    "spo": "studien prüfungsordnung",
    "bafög": "bafög studienfinanzierung amt",
    "prüfung": "klausur exam prüfungsleistung",
    "bewerbung": "zulassung immatrikulation einschreibung",
    "praktikum": "praxissemester praxisphase",
    "thesis": "bachelorarbeit masterarbeit abschlussarbeit",
}


def expand_program_terms(query: str) -> str:
    """Expand known abbreviations and synonyms in the query."""
    q_low = query.lower()
    additions = []
    for term, synonyms in PROGRAM_SYNONYMS.items():
        if term in q_low:
            additions.append(synonyms)
    if additions:
        return f"{query} {' '.join(additions)}"
    return query


def generate_query_variants(query: str) -> list[str]:
    """Generate rule-based query variants for multi-query retrieval."""
    variants = [query]

    # Reversed token order (often helps with German word order)
    tokens = query.split()
    if len(tokens) > 1:
        variants.append(" ".join(reversed(tokens)))

    # Synonym-expanded variant
    expanded = expand_program_terms(query)
    if expanded != query:
        variants.append(expanded)

    # Deduplicate while preserving order
    return list(dict.fromkeys(variants))


def hybrid_search(
    query: str,
    model,
    client: QdrantClient,
    total_limit: int = 20,
    semester: str = "SoSe26",
) -> list:
    """
    Multi-query hybrid search: generates query variants, searches each with
    dense + BM25 sparse, then merges all results via Reciprocal Rank Fusion.
    """
    query_type = _detect_query_type(query)
    weights = _get_collection_weights(query_type)
    variants = generate_query_variants(query)

    # Collections to search
    timetable_collection = f"starplan_{semester}"
    collections_to_search = {
        "hs_aalen_search": weights.get("hs_aalen_search", 1.0),
        "hs_aalen_website": weights.get("hs_aalen_website", 0.7),
        timetable_collection: weights.get("starplan_timetable", 0.3),
        "asta_content": weights.get("asta_content", 0.4),
        "hs_aalen_pdfs": weights.get("hs_aalen_pdfs", 0.8),   # NEU
    }

    per_collection_limit = max(10, total_limit)

    # RRF accumulator: url -> {result_dict, rrf_score}
    rrf_results: dict[str, dict] = {}
    RRF_K = 60  # Standard RRF constant

    for variant in variants:
        # Encode this variant (e5 query prefix applied inside encode_query)
        dense_vector = encode_query(variant)
        sparse_vector = sparse_encode(variant)

        for collection_name, weight in collections_to_search.items():
            if weight < 0.05:
                continue

            try:
                results = _search_collection(
                    client=client,
                    collection_name=collection_name,
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    query=variant,
                    limit=per_collection_limit,
                )
            except Exception as e:
                print(f"WARNING: Collection {collection_name} unavailable: {e}")

                # Fallback for semester-specific timetable
                if collection_name.startswith("starplan_") and collection_name != "starplan_timetable":
                    try:
                        results = _search_collection(
                            client=client,
                            collection_name="starplan_timetable",
                            dense_vector=dense_vector,
                            sparse_vector=sparse_vector,
                            query=variant,
                            limit=per_collection_limit,
                        )
                        weight = weights.get("starplan_timetable", 0.3)
                        collection_name = "starplan_timetable"
                    except Exception:
                        continue
                else:
                    continue

            # Accumulate RRF scores across variants and collections
            for rank, r in enumerate(results):
                url = r.get("url", f"_no_url_{rank}")
                rrf_score = weight / (RRF_K + rank + 1)

                if url not in rrf_results:
                    r["collection"] = collection_name
                    rrf_results[url] = {"result": r, "rrf": 0.0}
                rrf_results[url]["rrf"] += rrf_score

    # Assign merged RRF scores and sort
    merged = []
    for entry in rrf_results.values():
        r = entry["result"]
        r["score"] = entry["rrf"]
        merged.append(r)

    sorted_results = sorted(merged, key=lambda x: x["score"], reverse=True)
    return sorted_results[:total_limit]


def _search_collection(
    client: QdrantClient,
    collection_name: str,
    dense_vector: list[float],
    sparse_vector: SparseVector,
    query: str,
    limit: int,
) -> list[dict]:
    """
    Search a single collection using true hybrid retrieval:
    1. Dense prefetch (semantic similarity)
    2. Sparse BM25 prefetch (keyword/phrase matching)
    3. Reciprocal Rank Fusion (RRF) to merge both rankings

    Falls back to dense-only if the collection doesn't have sparse vectors.
    """
    try:
        # True hybrid: dense + sparse BM25 with RRF fusion
        results = client.query_points(
            collection_name=collection_name,
            prefetch=[
                # Dense (semantic)
                Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=limit * 2,
                ),
                # Sparse = BM25 (keyword/phrase matching)
                Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=limit * 2,
                ),
            ],
            # Reciprocal Rank Fusion: merges both rankings fairly
            # Documents that appear in both rankings get boosted
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        ).points

    except Exception as e:
        print(f"  Hybrid search failed for {collection_name}, trying dense-only: {e}")
        # Fallback: dense-only search (for collections without sparse vectors)
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
            # Final fallback: unnamed vector search (legacy collections)
            results = client.query_points(
                collection_name=collection_name,
                query=dense_vector,
                limit=limit,
                with_payload=True,
            ).points

    return _format_results(results, collection_name)


def _format_results(points, collection_name: str) -> list[dict]:
    """Convert Qdrant ScoredPoint objects to dicts.

    FIX 3: Use full content (up to 2000 chars) in the `text` field so that:
      - BM25 sparse vectors have enough signal to match multi-word queries
      - lexical_relevance() can actually find bigrams/trigrams
      - PDF content embedded during scraping is searchable
    """
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
        is_asta = source in ("asta_website", "asta")
        is_hs_website = source in ("hs_aalen_website", "hs_aalen")
        is_pdf = payload_type == "pdf" or source == "hs_aalen_pdfs"  # NEU

        if is_timetable:
            parts = [
                payload.get("name") or payload.get("title", ""),
                payload.get("day", ""),
                payload.get("time", ""),
            ]
            text = " — ".join(pt for pt in parts if pt)
            result_type = "timetable"

        elif is_pdf:                                                   # NEU – vor dem asta-check
            text = payload.get("text", "")
            result_type = "pdf"

        elif is_hs_website or is_asta:
            # FIX 3 — was: content[:200]. Now: up to 2000 chars so ranking has signal.
            # The frontend snippet is already limited in ResultItem.tsx via line-clamp.
            title = payload.get("title", "")
            content = payload.get("content", "")[:2000]

            # Include PDF text if the page had embedded PDFs
            pdf_extra = ""
            pdf_sources = payload.get("pdf_sources", [])
            if pdf_sources:
                # Pull pdf text from sections payload if available
                sections = payload.get("sections", [])
                pdf_section = next((s for s in sections if s.get("heading") == "PDF-Dokumente"), None)
                if pdf_section:
                    pdf_extra = " " + pdf_section.get("text", "")[:500]

            text = f"{title}: {content}{pdf_extra}".strip()
            result_type = "asta" if is_asta else "website"

        else:
            # hs_aalen_search chunks already have a `text` field
            # Also include section heading for better context
            raw_text = payload.get("text", "")
            section_heading = payload.get("section_heading", "")
            if section_heading and section_heading not in raw_text[:50]:
                text = f"{section_heading}: {raw_text}"
            else:
                text = raw_text
            result_type = "webpage"

        # Resolve title (prefer payload.title, fall back to first line of text)
        title_field = payload.get("title", "")
        if not title_field and text:
            title_field = text[:60].split("\n")[0]

        formatted.append({
            "score": float(p.score),
            "url": payload.get("url", ""),
            "text": text,
            "title": title_field,
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
        return {"intent": "person", "entity": q}
    if any(w in q_low for w in ["spo", "ordnung", "satzung", "antrag", "formular", "pdf"]):
        return {"intent": "document", "entity": q}
    if any(w in q_low for w in ["stundenplan", "vorlesung", "prüfung", "klausur", "termin"]):
        return {"intent": "timetable", "entity": q}
    
    return {"intent": "general", "entity": q}


def build_ngrams(tokens: list[str], n: int) -> set[str]:
    """Build n-grams from a token list."""
    return {" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


def lexical_relevance(query: str, text: str, url: str) -> float:
    """
    FIX 3 improvement: also search inside PDF content that is now part of `text`.
    The function itself is unchanged in logic — it automatically benefits because
    `text` now contains up to 2000 chars including PDF snippets.
    """
    q_tokens = tokenize(query)
    if not q_tokens:
        return 0.0

    haystack = f"{normalize_text(text)} {normalize_text(url)}"

    # Unigram coverage
    matched_unigrams = {tok for tok in q_tokens if tok in haystack}
    unigram_coverage = len(matched_unigrams) / len(q_tokens)

    # Bigram bonus
    bigram_bonus = 0.0
    if len(q_tokens) >= 2:
        q_bigrams = build_ngrams(q_tokens, 2)
        matched_bigrams = {bg for bg in q_bigrams if bg in haystack}
        if q_bigrams:
            bigram_bonus = 0.25 * (len(matched_bigrams) / len(q_bigrams))

    # Trigram bonus
    trigram_bonus = 0.0
    if len(q_tokens) >= 3:
        q_trigrams = build_ngrams(q_tokens, 3)
        matched_trigrams = {tg for tg in q_trigrams if tg in haystack}
        if q_trigrams:
            trigram_bonus = 0.15 * (len(matched_trigrams) / len(q_trigrams))

    # Exact phrase bonus
    phrase_bonus = 0.0
    normalized_query = normalize_text(query)
    if len(normalized_query) >= 6 and normalized_query in haystack:
        phrase_bonus = 0.3

    # Long token bonus (German compound words like "Prüfungsordnung")
    long_token_bonus = 0.0
    long_tokens = [t for t in q_tokens if len(t) >= 6]
    if long_tokens:
        long_matches = sum(1 for t in long_tokens if t in haystack)
        long_token_bonus = 0.1 * (long_matches / len(long_tokens))

    score = (
        0.55 * unigram_coverage
        + bigram_bonus
        + trigram_bonus
        + phrase_bonus
        + long_token_bonus
    )
    return max(0.0, min(1.0, score))


def get_navboost_stats(q: str) -> dict:
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT url, short_clicks, long_clicks FROM navboost_stats WHERE query = ?", (q.lower().strip(),))
        rows = cursor.fetchall()
        conn.close()
        return {row[0]: {"short": row[1], "long": row[2]} for row in rows}
    except Exception:
        return {}


def cross_encoder_rerank(query: str, results: list, pool: int = RERANK_POOL) -> list:
    """
    Re-rank the top `pool` candidates with a cross-encoder that scores the
    (query, document) pair jointly. This is the single biggest quality lever:
    the bi-encoder retrieval only gives a rough ordering, while the cross-encoder
    reads query and document together.

    The cross-encoder score is squashed to [0, 1] and blended 50/50 with the
    existing pipeline score, so retrieval signal, lexical match and boosts still
    carry weight (and ties are broken sensibly).
    """
    if cross_encoder is None or not results:
        return results

    candidates = results[:pool]
    rest = results[pool:]
    pairs = [(query, f"{r.get('title', '')} {r.get('text', '')[:500]}") for r in candidates]

    try:
        ce_scores = cross_encoder.predict(pairs)
    except Exception as e:
        print(f"Cross-encoder rerank failed, keeping retrieval order: {e}")
        return results

    for r, raw in zip(candidates, ce_scores):
        ce_norm = 1.0 / (1.0 + math.exp(-float(raw)))  # sigmoid → [0, 1]
        r["cross_encoder_score"] = ce_norm
        r["score"] = 0.5 * r["score"] + 0.5 * ce_norm

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates + rest


def boost_and_rank(q: str, results: list, intent_data: dict = None) -> list:
    if not results:
        return results

    q_low = q.lower().strip()
    intent = (intent_data or {}).get("intent", "general")
    navboost_stats = get_navboost_stats(q_low)

    # --- Normalize retrieval (RRF) scores to [0, 1] ---
    # The hybrid_search step produces RRF scores that live on a tiny scale
    # (~0.01–0.1). Blending them raw with the lexical score (0–1) effectively
    # discarded the semantic signal. Min-max normalisation puts both on the same
    # [0, 1] scale so the fusion below is meaningful.
    retrieval_scores = [r.get("score", 0.0) for r in results]
    min_s, max_s = min(retrieval_scores), max(retrieval_scores)
    denom = (max_s - min_s) or 1.0

    for r in results:
        title = (r.get("title") or "")
        url = (r.get("url") or "")
        text = r.get("text", "") + " " + title
        source = r.get("source", "")
        res_type = r.get("type", "")

        vector_norm = (r.get("score", 0.0) - min_s) / denom
        lex_score = lexical_relevance(q, text, url)

        # Semantic-leaning fusion (was inverted: 35% vector / 65% lexical).
        base = (0.70 * vector_norm) + (0.30 * lex_score)
        r["vector_score"] = vector_norm
        r["lexical_score"] = lex_score

        # --- Boosts as bounded multipliers, so they nudge rather than dominate ---
        boost = 1.0
        url_norm = normalize_text(url)
        if len(q_low) >= 4 and q_low in url_norm:
            boost *= 1.4
        elif any(len(t) >= 5 and t in url_norm for t in tokenize(q_low)):
            boost *= 1.15

        if intent == "timetable":
            boost *= 1.5 if res_type == "timetable" else 0.85
        elif intent == "person":
            if source == "hs_aalen" and res_type == "webpage":
                boost *= 1.3
            if any(k in title.lower() for k in ("prof", "rector", "rektor")):
                boost *= 1.3
        elif intent == "document":
            if res_type == "pdf":
                boost *= 1.5
            if any(k in title.lower() for k in ("satzung", "ordnung")):
                boost *= 1.2

        if source in ("hs_aalen", "asta"):
            boost *= 1.1

        score = base * boost

        # --- NavBoost: small additive nudge on the same [0, 1] scale ---
        if url in navboost_stats:
            stats = navboost_stats[url]
            nb = (stats["short"] * 0.02) + (stats["long"] * 0.08)
            score += min(nb, 0.2)
            r["is_navboosted"] = True

        r["score"] = score

    ranked = sorted(results, key=lambda x: x["score"], reverse=True)
    # Cross-Encoder reranking on the top pool (no-op if the model failed to load)
    ranked = cross_encoder_rerank(q, ranked)
    return ranked


@app.get("/api/search")
async def api_search(q: str = Query(...), page: int = Query(1)):
    # Local intent detection (no LLM)
    intent_data = _detect_intent_local(q)
    
    q_clean = q.lower().strip()
    if len(q_clean) >= 3:
        try:
            conn = sqlite3.connect(DB_FILE)
            cur = conn.cursor()
            cur.execute("SELECT search_count FROM query_stats WHERE query = ?", (q_clean,))
            row = cur.fetchone()
            if row:
                cur.execute("UPDATE query_stats SET search_count = search_count + 1 WHERE query = ?", (q_clean,))
            else:
                cur.execute("INSERT INTO query_stats (query, search_count) VALUES (?, 1)", (q_clean,))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error logging query: {e}")
    
    # Hybrid vector search (dense + BM25 sparse with RRF)
    # Get plenty of candidates for deep pagination
    results = hybrid_search(q, model, client, total_limit=150)
    
    # Ranking & Context
    ranked = boost_and_rank(q, results, intent_data)
    
    total_results = len(ranked)
    per_page = 10
    total_pages = max(1, math.ceil(total_results / per_page))
    
    page = max(1, min(page, total_pages))
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    contextualized = fetch_parent_context(ranked[start_idx:end_idx])
    
    return {
        "results": contextualized, 
        "total_results": total_results, 
        "page": page,
        "total_pages": total_pages,
        "per_page": per_page,
        "filters": intent_data, 
    }

@app.post("/api/feedback/click")
async def register_click(data: FeedbackRequest):
    q_low = data.query.lower().strip()
    if not q_low or not data.url:
        return {"status": "ignored"}
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute("SELECT short_clicks, long_clicks FROM navboost_stats WHERE query = ? AND url = ?", (q_low, data.url))
        row = cursor.fetchone()
        
        if row:
            if data.type == "long_click":
                cursor.execute("UPDATE navboost_stats SET long_clicks = long_clicks + 1 WHERE query = ? AND url = ?", (q_low, data.url))
            else:
                cursor.execute("UPDATE navboost_stats SET short_clicks = short_clicks + 1 WHERE query = ? AND url = ?", (q_low, data.url))
        else:
            if data.type == "long_click":
                cursor.execute("INSERT INTO navboost_stats (query, url, short_clicks, long_clicks) VALUES (?, ?, 0, 1)", (q_low, data.url))
            else:
                cursor.execute("INSERT INTO navboost_stats (query, url, short_clicks, long_clicks) VALUES (?, ?, 1, 0)", (q_low, data.url))
                
        conn.commit()
        conn.close()
        return {"status": "success"}
    except Exception as e:
        print(f"DB Error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/suggestions")
async def api_suggestions():
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("SELECT query FROM query_stats ORDER BY search_count DESC LIMIT 15")
        rows = cur.fetchall()
        conn.close()
        return {"suggestions": [row[0].title() for row in rows]}
    except Exception as e:
        print(f"Suggestions Error: {e}")
        return {"suggestions": []}

@app.get("/api/health")
async def health(): return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

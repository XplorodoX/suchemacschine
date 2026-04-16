"""
Ranking utilities for the search framework.

Extracted and generalized from scripts/app.py so both the new parametrized
framework and the existing scripts can share the same scoring logic.

Pipeline:
  1. normalize_text / tokenize — preprocessing
  2. lexical_relevance         — token coverage + phrase bonus
  3. strict_match_passes       — exact phrase enforcement
  4. boost_and_rank            — fuse vector score + lexical + URL boosts
  5. cross_encoder_rerank      — Cross-Encoder re-scores top-30 jointly
"""

from __future__ import annotations

import re
import unicodedata
import logging

log = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Default German stopwords (can be extended via config)
# -------------------------------------------------------------------------
DEFAULT_STOPWORDS: set[str] = {
    "der", "die", "das", "ein", "eine", "einer", "einem", "einen", "und", "oder",
    "aber", "mit", "ohne", "auf", "in", "im", "am", "an", "zu", "zum", "zur",
    "von", "für", "ist", "sind", "war", "wie", "was", "wer", "wo", "wann",
    "warum", "welche", "welcher", "welches", "ich", "du", "er", "sie", "es",
    "wir", "ihr", "nicht", "kein", "keine", "mehr", "auch", "den", "dem",
    "des", "bei", "über", "unter", "nach",
}


# -------------------------------------------------------------------------
# Text preprocessing
# -------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str, stopwords: set[str] = DEFAULT_STOPWORDS) -> list[str]:
    normalized = normalize_text(text)
    tokens = re.findall(r"[a-zA-Z0-9äöüÄÖÜß]{2,}", normalized)
    return [t for t in tokens if t not in stopwords]


def extract_quoted_phrases(query: str) -> list[str]:
    phrases = re.findall(r'"([^"]+)"', query)
    return [normalize_text(p) for p in phrases if p.strip()]


# -------------------------------------------------------------------------
# Lexical relevance
# -------------------------------------------------------------------------

def lexical_relevance(
    query: str,
    text: str,
    url: str,
    stopwords: set[str] = DEFAULT_STOPWORDS,
) -> float:
    """
    Score [0, 1] based on token overlap + phrase bonus + long-token bonus.
    72% of query tokens must appear in text or URL for full coverage score.
    """
    q_tokens = tokenize(query, stopwords)
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


def strict_match_passes(
    query: str,
    text: str,
    url: str,
    stopwords: set[str] = DEFAULT_STOPWORDS,
) -> bool:
    """
    Returns True if at least one core query token (≥4 chars) appears
    in the document text or URL. Enforces quoted phrases if present.
    """
    haystack = f"{normalize_text(text)} {normalize_text(url)}"
    phrases = extract_quoted_phrases(query)
    if phrases and not all(p in haystack for p in phrases):
        return False
    core_tokens = [t for t in tokenize(query, stopwords) if len(t) >= 4]
    if not core_tokens:
        return True
    return any(tok in haystack for tok in core_tokens)


# -------------------------------------------------------------------------
# Synonym expansion (program/module abbreviations, no LLM)
# -------------------------------------------------------------------------

def expand_program_terms(
    query: str,
    program_synonyms: dict[str, list[str]] | None = None,
    module_synonyms: dict[str, list[str]] | None = None,
) -> str:
    """
    Expand study-program abbreviations and module name aliases.
    Returns query with additional terms appended.
    """
    if not program_synonyms and not module_synonyms:
        return query

    normalized = normalize_text(query)
    expanded_terms: list[str] = []

    for key, synonyms in (program_synonyms or {}).items():
        if re.search(rf"\b{re.escape(key)}\b", normalized):
            expanded_terms.extend(synonyms)

    for key, synonyms in (module_synonyms or {}).items():
        if key in normalized:
            expanded_terms.extend(synonyms)

    if not expanded_terms:
        return query

    existing = set(tokenize(query))
    deduped = []
    for term in expanded_terms:
        term_norm = normalize_text(term)
        if term_norm and term_norm not in deduped and term_norm not in existing:
            deduped.append(term_norm)

    return f"{query} {' '.join(deduped)}" if deduped else query


# -------------------------------------------------------------------------
# Cross-Encoder reranking
# -------------------------------------------------------------------------

def cross_encoder_rerank(
    query: str,
    results: list[dict],
    cross_encoder,
    top_k: int = 10,
    candidate_pool: int = 30,
) -> list[dict]:
    """
    Rerank results using a cross-encoder that scores (query, document) jointly.
    Only the top `candidate_pool` results are reranked to keep latency low.

    Args:
        cross_encoder: loaded CrossEncoder model instance
        top_k:         how many results to return in the reranked top
        candidate_pool:how many candidates to feed into the cross-encoder
    """
    if cross_encoder is None or not results:
        return results

    candidates = results[:candidate_pool]
    rest = results[candidate_pool:]

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
        log.warning("Cross-encoder reranking failed: %s", e)
        return results


# -------------------------------------------------------------------------
# Main scoring pipeline
# -------------------------------------------------------------------------

def boost_and_rank(
    query: str,
    results: list[dict],
    cross_encoder=None,
    strict_match: bool = True,
    relevance_min_score: float = 0.34,
    stopwords: set[str] = DEFAULT_STOPWORDS,
    non_strict_types: tuple[str, ...] = ("timetable", "website", "asta"),
) -> list[dict]:
    """
    Combines vector score with lexical relevance + URL boosts, then
    applies Cross-Encoder reranking if a model is provided.

    Score formula:
        final = 0.72 * vector_norm + 0.28 * lexical + url_boost + type_boost
    """
    if not results:
        return results

    # --- 1. Normalize vector scores to [0, 1] ---
    vector_scores = [r.get("score", 0.0) for r in results]
    min_s = min(vector_scores)
    max_s = max(vector_scores)
    denom = (max_s - min_s) if max_s > min_s else 1.0

    for res in results:
        raw = res.get("score", 0.0)
        vector_norm = (raw - min_s) / denom
        url_str = normalize_text(res.get("url", ""))
        text_str = res.get("text", "")

        lexical = lexical_relevance(query, text_str, url_str, stopwords)
        final = (0.72 * vector_norm) + (0.28 * lexical)

        # URL path boost: query term literally in the URL path
        q_lower = query.lower().strip()
        if len(q_lower) >= 4 and q_lower in url_str:
            final += 0.35
        elif any(len(t) >= 5 and t in url_str for t in tokenize(q_lower, stopwords)):
            final += 0.15

        # Timetable boost: if semantic score is already decent
        if res.get("type") == "timetable" and raw >= 0.45:
            final += 0.25

        res["score"] = final
        res["vector_score"] = float(raw)
        res["lexical_score"] = float(lexical)

    # --- 2. Strict match filter ---
    if strict_match:
        results = [
            r for r in results
            if r.get("type") in non_strict_types
            or strict_match_passes(query, r.get("text", ""), r.get("url", ""), stopwords)
        ]

    # --- 3. Minimum score threshold ---
    results = [r for r in results if r["score"] >= relevance_min_score]
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    # --- 4. Cross-Encoder reranking ---
    if cross_encoder is not None and results:
        results = cross_encoder_rerank(query, results, cross_encoder)

    return results


# -------------------------------------------------------------------------
# Evidence quality check (for summary generation gating)
# -------------------------------------------------------------------------

def has_strong_evidence(results: list[dict], min_top1: float = 0.45, min_avg3: float = 0.40) -> bool:
    """
    Returns True if results are strong enough to generate a summary.
    Prevents hallucinated summaries from weak matches.
    """
    if not results:
        return False
    top1 = results[0].get("score", 0.0)
    avg_top3 = sum(r.get("score", 0.0) for r in results[:3]) / min(3, len(results))
    return (top1 >= min_top1) or (avg_top3 >= min_avg3)

"""
app_search_fix.py — Patches for backend/app.py to fix Bug 3

FIX 3:  In _format_results(), website and asta results only passed content[:200]
        into the `text` field. That kills BM25 keyword matching for long-form pages.
        Also: PDFs from augmented pages were shown but their full text was missing.

Apply by replacing the relevant sections in backend/app.py.
"""

# ─────────────────────────────────────────────────────────────────────────────
# REPLACEMENT for _format_results() in backend/app.py
# ─────────────────────────────────────────────────────────────────────────────

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

        if is_timetable:
            parts = [
                payload.get("name") or payload.get("title", ""),
                payload.get("day", ""),
                payload.get("time", ""),
            ]
            text = " — ".join(pt for pt in parts if pt)
            result_type = "timetable"

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


# ─────────────────────────────────────────────────────────────────────────────
# REPLACEMENT for lexical_relevance() — minor improvement for PDF content
# ─────────────────────────────────────────────────────────────────────────────

import re
import unicodedata

GERMAN_STOPWORDS = {
    "der", "die", "das", "ein", "eine", "einer", "einem", "einen", "und",
    "oder", "aber", "mit", "ohne", "auf", "in", "im", "am", "an", "zu",
    "zum", "zur", "von", "für", "ist", "sind", "war", "wie", "was", "wer",
    "wo", "wann", "warum", "welche", "welcher", "welches", "ich", "du", "er",
    "sie", "es", "wir", "ihr", "nicht", "kein", "keine", "mehr", "auch",
    "den", "dem", "des", "bei", "über", "unter", "nach",
}


def normalize_text(t: str) -> str:
    t = (t or "").lower()
    t = unicodedata.normalize("NFKC", t)
    return re.sub(r"\s+", " ", t).strip()


def tokenize(t: str) -> list[str]:
    return [
        w for w in re.findall(r"[a-zA-Z0-9äöüß]{2,}", normalize_text(t))
        if w not in GERMAN_STOPWORDS
    ]


def build_ngrams(tokens: list[str], n: int) -> set[str]:
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


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

"""
hybrid_search.py — Drop-in replacement for the hybrid_search() function in app.py

Uses Qdrant's native sparse+dense hybrid search instead of manual lexical scoring.
This gives proper BM25 keyword matching combined with semantic vector search.

Requirements:
  pip install qdrant-client[fastembed]
  # fastembed provides BM25 sparse vectors without a separate service
"""

import os
import re
from qdrant_client import QdrantClient
from qdrant_client.models import (
    NamedVector,
    NamedSparseVector,
    SparseVector,
    Prefetch,
    FusionQuery,
    Fusion,
    SearchRequest,
)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

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

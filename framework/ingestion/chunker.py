"""
ContextualChunker — splits documents into chunks and embeds them.

Each chunk gets a context header so it is self-contained when read in isolation:

  Seite: HS Aalen – Informatik
  Abschnitt: Prüfungsordnung

  Der Antrag muss bis zum 15. des Monats eingereicht werden ...

This is the "Contextual Retrieval" approach from Anthropic (2024): prepend
parent-document context to every chunk before embedding, which significantly
improves retrieval quality for dense vector search.

Output format (list of Chunk dicts):
  {
    "url": str,
    "text": str,             # context header + chunk body
    "title": str,
    "h1": str,
    "section_heading": str,
    "section_index": int,
    "chunk_index": int,
    "type": str,
    "metadata": dict,
    "embedding": list[float],
    # optional sparse fields:
    "sparse_indices": list[int],
    "sparse_values": list[float],
  }
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)


def _build_context_header(title: str, h1: str, heading: str) -> str:
    """Return a short human-readable context prefix for a chunk."""
    parts = []
    if title:
        parts.append(f"Seite: {title}")
    if h1 and h1.lower() != title.lower():
        parts.append(f"Überschrift: {h1}")
    if heading and heading not in ("Allgemein", title, h1):
        parts.append(f"Abschnitt: {heading}")
    return "\n".join(parts)


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    RecursiveCharacterTextSplitter-style splitting using LangChain.
    Falls back to simple stride-based splitting if LangChain is unavailable.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        return splitter.split_text(text)
    except ImportError:
        # Simple stride-based fallback
        stride = chunk_size - chunk_overlap
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start : start + chunk_size])
            start += stride
        return chunks


class ContextualChunker:
    """
    Chunks documents and embeds them using a SentenceTransformer model.

    Args:
        model_name:   HuggingFace model name (e.g. "intfloat/multilingual-e5-base")
        chunk_size:   max characters per chunk (default 800)
        chunk_overlap:overlap between consecutive chunks (default 120)
        sparse_model_name: optional BM42/SPLADE model name for sparse vectors
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        sparse_model_name: str | None = None,
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._use_e5_prefix = "e5" in model_name.lower()
        self._model = None
        self._sparse_model = None

        # Load dense model
        log.info("Loading embedding model: %s", model_name)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            log.info("✓ Embedding model loaded")
        except Exception as e:
            raise RuntimeError(f"Could not load embedding model '{model_name}': {e}")

        # Load sparse model (optional)
        if sparse_model_name:
            log.info("Loading sparse model: %s", sparse_model_name)
            try:
                from fastembed import SparseTextEmbedding
                self._sparse_model = SparseTextEmbedding(model_name=sparse_model_name)
                log.info("✓ Sparse model loaded")
            except ImportError:
                log.warning("fastembed not installed — sparse vectors disabled. pip install fastembed")
            except Exception as e:
                log.warning("Could not load sparse model '%s': %s", sparse_model_name, e)

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, applying e5 prefix if needed."""
        if self._use_e5_prefix:
            prefixed = [f"passage: {t}" for t in texts]
        else:
            prefixed = texts
        return self._model.encode(prefixed, show_progress_bar=False).tolist()

    def _sparse_embed(self, texts: list[str]) -> list[tuple[list[int], list[float]] | None]:
        """Return list of (indices, values) tuples, or None if sparse unavailable."""
        if self._sparse_model is None:
            return [None] * len(texts)
        try:
            results = list(self._sparse_model.embed(texts))
            return [(r.indices.tolist(), r.values.tolist()) for r in results]
        except Exception as e:
            log.warning("Sparse embedding failed: %s", e)
            return [None] * len(texts)

    def chunk_document(self, doc: dict) -> list[dict]:
        """
        Split a single document into contextual chunks.
        No embeddings yet — call chunk_all() for batched embedding.
        """
        url = doc.get("url", "")
        title = (doc.get("title") or "").strip()
        h1 = (doc.get("h1") or "").strip()
        sections = doc.get("sections") or []
        doc_type = doc.get("type", "webpage")
        metadata = doc.get("metadata") or {}

        raw_chunks: list[dict] = []

        # Special handling for timetable entries — no chunking needed
        if doc_type == "timetable":
            content = doc.get("content", "")
            if content:
                raw_chunks.append({
                    "url": url,
                    "text": content,
                    "title": title,
                    "h1": h1,
                    "section_heading": "",
                    "section_index": 0,
                    "chunk_index": 0,
                    "type": doc_type,
                    "metadata": metadata,
                })
            return raw_chunks

        # Section-aware chunking
        if sections:
            for s_idx, section in enumerate(sections):
                heading = (section.get("heading") or "Allgemein").strip()
                section_text = (section.get("text") or "").strip()
                if not section_text:
                    continue
                ctx = _build_context_header(title, h1, heading)
                pieces = _split_text(section_text, self.chunk_size, self.chunk_overlap)
                for c_idx, piece in enumerate(pieces):
                    text = f"{ctx}\n\n{piece}" if ctx else piece
                    raw_chunks.append({
                        "url": url,
                        "text": text,
                        "title": title,
                        "h1": h1,
                        "section_heading": heading,
                        "section_index": s_idx,
                        "chunk_index": c_idx,
                        "type": doc_type,
                        "metadata": metadata,
                    })

        # Fallback: chunk the full content field
        if not raw_chunks:
            content = (doc.get("content") or "").strip()
            if not content:
                return []
            ctx = _build_context_header(title, h1, "")
            pieces = _split_text(content, self.chunk_size + 100, self.chunk_overlap + 30)
            for c_idx, piece in enumerate(pieces):
                text = f"{ctx}\n\n{piece}" if ctx else piece
                raw_chunks.append({
                    "url": url,
                    "text": text,
                    "title": title,
                    "h1": h1,
                    "section_heading": "Allgemein",
                    "section_index": 0,
                    "chunk_index": c_idx,
                    "type": doc_type,
                    "metadata": metadata,
                })

        return raw_chunks

    def chunk_all(self, documents: list[dict], batch_size: int = 64) -> list[dict]:
        """
        Chunk all documents and embed them in batches.
        Returns list of Chunk dicts ready for indexing.
        """
        # 1. Chunk everything (no embedding yet)
        all_raw: list[dict] = []
        for doc in documents:
            all_raw.extend(self.chunk_document(doc))

        log.info("Chunked %d documents → %d chunks", len(documents), len(all_raw))

        # 2. Embed in batches
        texts = [c["text"] for c in all_raw]
        embedded_chunks: list[dict] = []

        for batch_start in range(0, len(texts), batch_size):
            batch_texts = texts[batch_start : batch_start + batch_size]
            batch_chunks = all_raw[batch_start : batch_start + batch_size]

            embeddings = self._embed_texts(batch_texts)
            sparse = self._sparse_embed(batch_texts)

            for chunk, emb, sp in zip(batch_chunks, embeddings, sparse):
                result = {**chunk, "embedding": emb}
                if sp is not None:
                    result["sparse_indices"] = sp[0]
                    result["sparse_values"] = sp[1]
                embedded_chunks.append(result)

            if (batch_start // batch_size + 1) % 10 == 0:
                log.info(
                    "  Embedded %d/%d chunks...",
                    min(batch_start + batch_size, len(texts)),
                    len(texts),
                )

        log.info("Embedding complete — %d chunks ready for indexing", len(embedded_chunks))
        return embedded_chunks

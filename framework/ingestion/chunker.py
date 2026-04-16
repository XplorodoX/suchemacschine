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
    "text": str,             # context header + child chunk body (embedded)
    "parent_text": str,      # context header + parent chunk body (returned in results)
                             #   only present when parent_chunk_size is configured;
                             #   equals text when parent-child mode is off.
    "title": str,
    "h1": str,
    "section_heading": str,
    "section_index": int,
    "chunk_index": int,
    "parent_index": int,     # index of the parent chunk within its section (0-based)
    "type": str,
    "metadata": dict,
    "embedding": list[float],         # dense bi-encoder vector
    "colbert_vectors": list[list[float]],  # token-level ColBERT vectors (optional)
    # optional sparse fields:
    "sparse_indices": list[int],
    "sparse_values": list[float],
  }

Parent-Child (Small-to-Big) Retrieval
--------------------------------------
When *parent_chunk_size* is set, each section is first split into large
*parent* chunks (e.g. 1000 chars), then each parent is further split into
small *child* chunks (e.g. 200 chars).  Only the child text is embedded;
the parent text rides along in the payload and is returned to the user.
This gives precise retrieval (small chunks match queries well) with rich,
contextual results (the parent contains enough surrounding text to be
readable and useful).  Without parent_chunk_size the chunker behaves
exactly as before.
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
        model_name:        HuggingFace model name (e.g. "intfloat/multilingual-e5-base")
        chunk_size:        child chunk size in characters (embedded, default 800).
                           When parent_chunk_size is set this should be a small value
                           like 200 so child chunks are precise retrieval targets.
        chunk_overlap:     overlap between consecutive child chunks (default 120)
        parent_chunk_size: if given, enables Parent-Child (Small-to-Big) retrieval.
                           Sections are first split into parent chunks of this size;
                           each parent is then split further into child chunks of
                           *chunk_size*.  Parent text is stored in payload as
                           ``parent_text`` and returned to users instead of the
                           short child text.  Set to None (default) for the classic
                           single-level chunking behaviour.
        colbert_model_name: optional ColBERT model for Late Interaction token-level
                           embeddings (e.g. "jinaai/jina-colbert-v2").  When set,
                           each chunk also gets a ``colbert_vectors`` field containing
                           one 128-dim vector per token.  Qdrant stores these as a
                           MultiVector and uses MaxSim scoring at query time.
                           Requires fastembed ≥ 0.3:  pip install fastembed
        sparse_model_name: optional BM42/SPLADE model name for sparse vectors
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        parent_chunk_size: int | None = None,
        colbert_model_name: str | None = None,
        sparse_model_name: str | None = None,
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Parent-child mode: parent_chunk_size must be larger than chunk_size.
        if parent_chunk_size is not None and parent_chunk_size <= chunk_size:
            raise ValueError(
                f"parent_chunk_size ({parent_chunk_size}) must be larger than "
                f"chunk_size ({chunk_size})"
            )
        self.parent_chunk_size = parent_chunk_size
        # Child overlap: keep small relative to child size to avoid redundancy
        self._child_overlap = min(chunk_overlap, max(20, chunk_size // 6))
        self._use_e5_prefix = "e5" in model_name.lower()
        self._model = None
        self._sparse_model = None
        self._colbert_model = None

        # Load dense model
        log.info("Loading embedding model: %s", model_name)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            log.info("✓ Embedding model loaded")
        except Exception as e:
            raise RuntimeError(f"Could not load embedding model '{model_name}': {e}")

        # Load ColBERT model (optional) — Late Interaction token-level embeddings
        if colbert_model_name:
            log.info("Loading ColBERT model: %s", colbert_model_name)
            try:
                from fastembed import LateInteractionTextEmbedding
                self._colbert_model = LateInteractionTextEmbedding(model_name=colbert_model_name)
                log.info("✓ ColBERT model loaded: %s", colbert_model_name)
            except ImportError:
                log.warning("fastembed not installed — ColBERT disabled. pip install 'fastembed>=0.3'")
            except Exception as e:
                log.warning("Could not load ColBERT model '%s': %s", colbert_model_name, e)

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

    def _colbert_embed(self, texts: list[str]) -> list[list[list[float]] | None]:
        """
        Return token-level ColBERT embeddings for each text.

        Each element is a list of token vectors (shape: n_tokens × colbert_dim),
        or None if the ColBERT model is not loaded.
        Uses 'passage:' prefix for ColBERT passage encoding when the model
        is jina-colbert (which follows the same convention as e5).
        """
        if self._colbert_model is None:
            return [None] * len(texts)
        try:
            results = list(self._colbert_model.embed(texts))
            # Each result is an ndarray of shape (n_tokens, colbert_dim)
            return [r.tolist() for r in results]
        except Exception as e:
            log.warning("ColBERT embedding failed: %s", e)
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
                raw_chunks.extend(
                    self._make_chunks(
                        section_text, ctx, url, title, h1, heading, s_idx, doc_type, metadata
                    )
                )

        # Fallback: chunk the full content field
        if not raw_chunks:
            content = (doc.get("content") or "").strip()
            if not content:
                return []
            ctx = _build_context_header(title, h1, "")
            raw_chunks.extend(
                self._make_chunks(
                    content, ctx, url, title, h1, "Allgemein", 0, doc_type, metadata,
                    fallback=True,
                )
            )

        return raw_chunks

    # ------------------------------------------------------------------

    def _make_chunks(
        self,
        text: str,
        ctx: str,
        url: str,
        title: str,
        h1: str,
        heading: str,
        s_idx: int,
        doc_type: str,
        metadata: dict,
        fallback: bool = False,
    ) -> list[dict]:
        """
        Turn a block of section text into chunk dicts.

        In Parent-Child mode (``self.parent_chunk_size`` is set):
          1. Split *text* into large parent pieces.
          2. Split each parent further into small child pieces.
          3. Each child dict carries ``parent_text`` (with context header)
             so the retrieval layer can return the richer parent document.
        In classic mode (``self.parent_chunk_size`` is None):
          Behaves exactly as before — one level of splitting.
        """
        chunks: list[dict] = []

        if self.parent_chunk_size is not None:
            # ── Parent-Child mode ──────────────────────────────────────
            parent_size = self.parent_chunk_size
            parent_overlap = self.chunk_overlap  # coarser overlap for parents
            if fallback:
                parent_size += 200  # slightly bigger parents for full-content fallback

            parents = _split_text(text, parent_size, parent_overlap)
            for p_idx, parent_piece in enumerate(parents):
                parent_text = f"{ctx}\n\n{parent_piece}" if ctx else parent_piece
                children = _split_text(parent_piece, self.chunk_size, self._child_overlap)
                for c_idx, child_piece in enumerate(children):
                    child_text = f"{ctx}\n\n{child_piece}" if ctx else child_piece
                    chunks.append({
                        "url": url,
                        "text": child_text,       # embedded — small, precise
                        "parent_text": parent_text,  # returned — large, contextual
                        "title": title,
                        "h1": h1,
                        "section_heading": heading,
                        "section_index": s_idx,
                        "parent_index": p_idx,
                        "chunk_index": c_idx,
                        "type": doc_type,
                        "metadata": metadata,
                    })
        else:
            # ── Classic single-level mode ──────────────────────────────
            size = self.chunk_size + (100 if fallback else 0)
            overlap = self.chunk_overlap + (30 if fallback else 0)
            for c_idx, piece in enumerate(_split_text(text, size, overlap)):
                chunk_text = f"{ctx}\n\n{piece}" if ctx else piece
                chunks.append({
                    "url": url,
                    "text": chunk_text,
                    "title": title,
                    "h1": h1,
                    "section_heading": heading,
                    "section_index": s_idx,
                    "parent_index": 0,
                    "chunk_index": c_idx,
                    "type": doc_type,
                    "metadata": metadata,
                })

        return chunks

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
            colbert = self._colbert_embed(batch_texts)

            for chunk, emb, sp, cb in zip(batch_chunks, embeddings, sparse, colbert):
                result = {**chunk, "embedding": emb}
                if sp is not None:
                    result["sparse_indices"] = sp[0]
                    result["sparse_values"] = sp[1]
                if cb is not None:
                    # list[list[float]]: one vector per token
                    result["colbert_vectors"] = cb
                embedded_chunks.append(result)

            if (batch_start // batch_size + 1) % 10 == 0:
                log.info(
                    "  Embedded %d/%d chunks...",
                    min(batch_start + batch_size, len(texts)),
                    len(texts),
                )

        log.info("Embedding complete — %d chunks ready for indexing", len(embedded_chunks))
        return embedded_chunks

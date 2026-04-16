"""
SearchPipeline — orchestrates the full ingestion process for one config.

Usage:
    pipeline = SearchPipeline("configs/hs-aalen.yaml")
    pipeline.run()

What it does for each source defined in the config:
  1. Load raw documents (WebsiteLoader / StarplanLoader / FolderLoader / UrlListLoader)
  2. Chunk documents and embed them (ContextualChunker)
  3. Index chunks into Qdrant (QdrantIndexer)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from .loaders import get_loader
from .chunker import ContextualChunker
from .indexer import QdrantIndexer

log = logging.getLogger(__name__)


class SearchPipeline:
    """
    Generic ingestion pipeline driven entirely by a YAML config file.

    One pipeline instance = one config = one project (e.g. "hs-aalen").
    Multiple sources in the same config are processed sequentially.
    Each source maps to its own Qdrant collection.
    """

    def __init__(self, config_path: str | Path):
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with config_path.open(encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.project_name = self.config["name"]
        self.embedding_model = self.config.get("embedding_model", "intfloat/multilingual-e5-base")
        self.vector_size = int(self.config.get("vector_size", 768))

        use_sparse = (
            self.config.get("sparse_vectors", False)
            or os.getenv("USE_SPARSE_VECTORS", "false").lower() == "true"
        )
        self.sparse_model = self.config.get("sparse_model") if use_sparse else None

        qdrant_cfg = self.config.get("qdrant", {})
        self.qdrant_host = os.getenv("QDRANT_HOST", qdrant_cfg.get("host", "localhost"))
        self.qdrant_port = int(os.getenv("QDRANT_PORT", qdrant_cfg.get("port", 6333)))

        log.info(
            "SearchPipeline initialized for project '%s' (model=%s, qdrant=%s:%d)",
            self.project_name, self.embedding_model, self.qdrant_host, self.qdrant_port,
        )

    def run(self, source_names: list[str] | None = None):
        """
        Run the full ingestion pipeline.

        Args:
            source_names: if given, only process sources with these names.
                          Default: process all sources.
        """
        sources = self.config.get("sources", [])
        if not sources:
            log.warning("No sources defined in config — nothing to do.")
            return

        if source_names:
            sources = [s for s in sources if s.get("name") in source_names]
            if not sources:
                log.warning("No matching sources for filter: %s", source_names)
                return

        # Build chunker once (model loads once for all sources)
        chunker = ContextualChunker(
            model_name=self.embedding_model,
            chunk_size=int(sources[0].get("chunk_size", self.config.get("chunk_size", 800))),
            chunk_overlap=int(sources[0].get("chunk_overlap", self.config.get("chunk_overlap", 120))),
            sparse_model_name=self.sparse_model,
        )

        total_chunks = 0

        for source in sources:
            source_name = source.get("name", source.get("type", "unnamed"))
            collection = source.get("collection", f"{self.project_name}_{source_name}")

            log.info("=" * 60)
            log.info("Processing source: %s → collection: %s", source_name, collection)

            # Use source-specific chunk size if provided
            src_chunk_size = source.get("chunk_size")
            src_chunk_overlap = source.get("chunk_overlap")
            if src_chunk_size or src_chunk_overlap:
                # Rebuild chunker with source-specific settings
                source_chunker = ContextualChunker(
                    model_name=self.embedding_model,
                    chunk_size=int(src_chunk_size or 800),
                    chunk_overlap=int(src_chunk_overlap or 120),
                    sparse_model_name=self.sparse_model,
                )
            else:
                source_chunker = chunker

            # Step 1: Load
            try:
                loader = get_loader(source)
                documents = loader.load()
                log.info("  Loaded %d documents", len(documents))
            except Exception as e:
                log.error("  Failed to load source '%s': %s", source_name, e)
                continue

            if not documents:
                log.warning("  No documents loaded from '%s' — skipping", source_name)
                continue

            # Step 2: Chunk + Embed
            try:
                chunks = source_chunker.chunk_all(documents)
                log.info("  Produced %d chunks", len(chunks))
            except Exception as e:
                log.error("  Failed to chunk source '%s': %s", source_name, e)
                continue

            # Step 3: Index
            try:
                indexer = QdrantIndexer(
                    collection=collection,
                    vector_size=self.vector_size,
                    host=self.qdrant_host,
                    port=self.qdrant_port,
                )
                n = indexer.index(chunks)
                total_chunks += n
            except Exception as e:
                log.error("  Failed to index source '%s': %s", source_name, e)
                continue

            log.info("  ✓ Done: %s", source_name)

        log.info("=" * 60)
        log.info("Pipeline complete — %d total chunks indexed across %d sources", total_chunks, len(sources))

import json
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Path to the scraped data
INPUT_FILE = "/Users/merluee/Desktop/suchemacschine/data.jsonl"
OUTPUT_FILE = "/Users/merluee/Desktop/suchemacschine/processed_data.jsonl"

# Upgraded to multilingual-e5-base: much better multilingual quality (768-dim).
# IMPORTANT: After changing this model you must re-index all Qdrant collections!
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
# e5 models require a task prefix: "passage: " for documents, "query: " for queries
USE_E5_PREFIX = "e5" in MODEL_NAME.lower()

# SPLADE sparse embeddings (opt-in).
# Set USE_SPARSE_VECTORS=true to also compute sparse vectors.
# Requires: pip install fastembed
# When enabled, also update index_to_qdrant.py (USE_SPARSE_VECTORS=true).
USE_SPARSE_VECTORS = os.getenv("USE_SPARSE_VECTORS", "false").lower() == "true"

sparse_model = None
if USE_SPARSE_VECTORS:
    try:
        from fastembed import SparseTextEmbedding
        # BM42: attention-weighted BM25, works for German/multilingual text.
        # Alternative for better term expansion: prithivida/Splade_PP_en_v1 (English only).
        SPARSE_MODEL_NAME = os.getenv("SPARSE_MODEL", "Qdrant/bm42-all-minilm-l6-v2-attentions")
        print(f"Loading sparse embedding model ({SPARSE_MODEL_NAME})...")
        sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
        print("✓ Sparse model loaded")
    except ImportError:
        print("⚠️  fastembed not installed — skipping sparse vectors. Run: pip install fastembed")


def load_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)


def contextual_prefix(title: str, h1: str, heading: str) -> str:
    """
    Contextual Chunking: prepend a short context header to every chunk so it
    remains self-contained when read in isolation by the embedding model.
    This prevents the "chunk without context" problem where a chunk like
    "Der Antrag muss bis zum 15. eingereicht werden" gives no hint what it's about.
    """
    parts = []
    if title:
        parts.append(f"Seite: {title}")
    # Only add h1 if it differs meaningfully from the title
    if h1 and h1.lower() != title.lower():
        parts.append(f"Überschrift: {h1}")
    if heading and heading not in ("Allgemein", title, h1):
        parts.append(f"Abschnitt: {heading}")
    return "\n".join(parts)


def build_section_chunks(record):
    """Create chunks aligned with page sections whenever available."""
    title = (record.get("title") or "").strip()
    h1 = (record.get("h1") or "").strip()
    sections = record.get("sections") or []
    url = record.get("url")

    chunk_records = []

    if sections:
        for idx, section in enumerate(sections):
            heading = (section.get("heading") or "Allgemein").strip()
            section_text = (section.get("text") or "").strip()
            if not section_text:
                continue

            # Build contextual prefix so every chunk knows what page/section it's from
            ctx = contextual_prefix(title, h1, heading)

            section_chunks = chunk_text(section_text, chunk_size=800, chunk_overlap=120)
            for cidx, chunk in enumerate(section_chunks):
                # Context header + chunk body (same approach as Anthropic Contextual Retrieval)
                enriched_text = f"{ctx}\n\n{chunk}" if ctx else chunk
                chunk_records.append(
                    {
                        "url": url,
                        "text": enriched_text,
                        "section_heading": heading,
                        "title": title,
                        "h1": h1,
                        "section_index": idx,
                        "chunk_index": cidx,
                        "used_js_render": bool(record.get("used_js_render", False)),
                    }
                )

    if not chunk_records:
        content = (record.get("content") or "").strip()
        if not content:
            return []

        ctx = contextual_prefix(title, h1, "")

        fallback_chunks = chunk_text(content, chunk_size=900, chunk_overlap=150)
        for cidx, chunk in enumerate(fallback_chunks):
            enriched_text = f"{ctx}\n\n{chunk}" if ctx else chunk
            chunk_records.append(
                {
                    "url": url,
                    "text": enriched_text,
                    "section_heading": "Allgemein",
                    "title": title,
                    "h1": h1,
                    "section_index": 0,
                    "chunk_index": cidx,
                    "used_js_render": bool(record.get("used_js_render", False)),
                }
            )

    return chunk_records


def main():
    print(f"Loading data from {INPUT_FILE}...")
    raw_data = load_data(INPUT_FILE)

    print(f"Loading embedding model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)

    processed_records = []

    print("Processing and embedding chunks...")
    for i, record in enumerate(raw_data):
        chunk_records = build_section_chunks(record)
        if not chunk_records:
            continue

        raw_texts = [item["text"] for item in chunk_records]
        # e5 models need "passage: " prefix on documents at index time
        chunk_texts = [f"passage: {t}" for t in raw_texts] if USE_E5_PREFIX else raw_texts
        embeddings = model.encode(chunk_texts, show_progress_bar=False)

        # Compute sparse vectors if enabled
        sparse_embeddings = None
        if sparse_model is not None:
            sparse_embeddings = list(sparse_model.embed(raw_texts))

        for j, (item, embedding) in enumerate(zip(chunk_records, embeddings)):
            record = {**item, "embedding": embedding.tolist()}
            if sparse_embeddings is not None:
                sp = sparse_embeddings[j]
                # Store as {indices: [...], values: [...]} for Qdrant SparseVector
                record["sparse_indices"] = sp.indices.tolist()
                record["sparse_values"] = sp.values.tolist()
            processed_records.append(record)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(raw_data)} documents...")

    print(f"Saving {len(processed_records)} chunks to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in processed_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Data preparation complete!")


if __name__ == "__main__":
    main()

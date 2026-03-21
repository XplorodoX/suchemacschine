import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Path to the scraped data
INPUT_FILE = "/Users/merluee/Desktop/suchemacschine/data.jsonl"
OUTPUT_FILE = "/Users/merluee/Desktop/suchemacschine/processed_data.jsonl"

# Choice of model: multilingual model for proper German text support.
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


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

            section_chunks = chunk_text(section_text, chunk_size=800, chunk_overlap=120)
            for cidx, chunk in enumerate(section_chunks):
                enriched_text = "\n".join(part for part in [title, h1, heading, chunk] if part)
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

        fallback_chunks = chunk_text(content, chunk_size=900, chunk_overlap=150)
        for cidx, chunk in enumerate(fallback_chunks):
            enriched_text = "\n".join(part for part in [title, h1, chunk] if part)
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

        chunk_texts = [item["text"] for item in chunk_records]
        embeddings = model.encode(chunk_texts)

        for item, embedding in zip(chunk_records, embeddings):
            processed_records.append({**item, "embedding": embedding.tolist()})

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(raw_data)} documents...")

    print(f"Saving {len(processed_records)} chunks to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in processed_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Data preparation complete!")


if __name__ == "__main__":
    main()

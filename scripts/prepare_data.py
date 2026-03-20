import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Path to the scraped data
INPUT_FILE = "/Users/merluee/Desktop/suchemacschine/data.jsonl"
OUTPUT_FILE = "/Users/merluee/Desktop/suchemacschine/processed_data.jsonl"

# Choice of model: 'all-MiniLM-L6-v2' is small, fast, and good for general purpose.
MODEL_NAME = "all-MiniLM-L6-v2"


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


def main():
    print(f"Loading data from {INPUT_FILE}...")
    raw_data = load_data(INPUT_FILE)

    print(f"Loading embedding model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)

    processed_records = []

    print("Processing and embedding chunks...")
    for i, record in enumerate(raw_data):
        url = record["url"]
        content = record["content"]

        # Split content into chunks
        chunks = chunk_text(content)

        if not chunks:
            continue

        # Generate embeddings for all chunks in a batch for efficiency
        embeddings = model.encode(chunks)

        for chunk, embedding in zip(chunks, embeddings):
            processed_records.append(
                {
                    "url": url,
                    "text": chunk,
                    "embedding": embedding.tolist(),  # Convert numpy array to list for JSON serialization
                }
            )

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(raw_data)} documents...")

    print(f"Saving {len(processed_records)} chunks to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in processed_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Data preparation complete!")


if __name__ == "__main__":
    main()

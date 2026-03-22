#!/usr/bin/env python3
"""
Text Processor Service
Tokenisiert und chunked Texte für Embeddings
"""
import os
import json
import glob
from datetime import datetime
from typing import List, Dict

INPUT_DIR = os.getenv("INPUT_DIR", "/data/raw")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/data/processed")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
OVERLAP = int(os.getenv("OVERLAP", "50"))

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    stride = chunk_size - overlap
    
    for i in range(0, len(text), stride):
        chunk = text[i:i + chunk_size]
        if len(chunk.strip()) > 20:  # Only keep meaningful chunks
            chunks.append(chunk)
        
        # Stop if we've reached the end
        if i + chunk_size >= len(text):
            break
    
    return chunks

def process_jsonl_file(input_file: str) -> List[Dict]:
    """Process a JSONL file and return chunked records."""
    results = []
    
    print(f"   Processing: {os.path.basename(input_file)}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                text = record.get('text', '')
                
                if not text:
                    continue
                
                # Create chunks
                chunks = chunk_text(text)
                
                for chunk_id, chunk in enumerate(chunks):
                    processed_record = {
                        "id": f"{record.get('url', 'unknown')}_chunk_{chunk_id}",
                        "url": record.get('url', ''),
                        "title": record.get('title', ''),
                        "text": chunk,
                        "chunk_id": chunk_id,
                        "total_chunks": len(chunks),
                        "source": record.get('source', 'unknown'),
                        "processed_at": datetime.now().isoformat(),
                        "original_scraped_at": record.get('scraped_at', '')
                    }
                    results.append(processed_record)
            
            except json.JSONDecodeError as e:
                print(f"   ⚠️  Error on line {line_num}: {e}")
            except Exception as e:
                print(f"   ⚠️  Error processing line {line_num}: {e}")
    
    return results

def main():
    """Main processing function."""
    print(f"🔄 Starting text processor at {datetime.now()}")
    print(f"   Input dir: {INPUT_DIR}")
    print(f"   Chunk size: {CHUNK_SIZE}, Overlap: {OVERLAP}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find all JSONL files
    input_files = glob.glob(os.path.join(INPUT_DIR, "*.jsonl"))
    
    if not input_files:
        print(f"⚠️  No JSONL files found in {INPUT_DIR}")
        return
    
    all_results = []
    
    for input_file in input_files:
        processed = process_jsonl_file(input_file)
        all_results.extend(processed)
    
    # Save processed data
    output_file = os.path.join(OUTPUT_DIR, "processed_data.jsonl")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in all_results:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Processed {len(all_results)} chunks")
    print(f"   Saved to: {output_file}")

if __name__ == "__main__":
    main()

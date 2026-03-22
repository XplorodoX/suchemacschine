#!/usr/bin/env python3
"""
Embeddings Service
Nutzt Ollama für Vektorgenerierung
"""
import os
import json
import glob
from datetime import datetime
from typing import List

import requests
import numpy as np

INPUT_DIR = os.getenv("INPUT_DIR", "/data/processed")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/data/embeddings")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text")

def get_embedding(text: str) -> List[float]:
    """Get embedding vector from Ollama."""
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": OLLAMA_MODEL, "prompt": text},
            timeout=30
        )
        response.raise_for_status()
        return response.json().get("embedding", [])
    except Exception as e:
        print(f"   ⚠️  Error getting embedding: {e}")
        # Return zero vector as fallback
        return [0.0] * 384

def process_embeddings():
    """Main embedding function."""
    print(f"📊 Starting embeddings service at {datetime.now()}")
    print(f"   Ollama host: {OLLAMA_HOST}")
    print(f"   Model: {OLLAMA_MODEL}")
    
    # Check Ollama connection
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        models = response.json().get("models", [])
        print(f"   Available models: {[m.get('name') for m in models]}")
    except Exception as e:
        print(f"   ⚠️  Warning: Could not connect to Ollama: {e}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find processed data file
    input_files = glob.glob(os.path.join(INPUT_DIR, "*.jsonl"))
    
    if not input_files:
        print(f"⚠️  No input files found in {INPUT_DIR}")
        return
    
    input_file = input_files[0]  # Process first file
    print(f"   Input file: {os.path.basename(input_file)}")
    
    results = []
    processed_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                text = record.get('text', '')
                
                if not text or len(text.strip()) < 10:
                    continue
                
                # Get embedding
                embedding = get_embedding(text)
                
                if not embedding:
                    print(f"   ⚠️  Line {line_num}: Failed to get embedding")
                    continue
                
                # Add embedding to record
                record['embedding'] = embedding
                results.append(record)
                processed_count += 1
                
                if line_num % 50 == 0:
                    print(f"   Progress: {line_num}/{total_lines} processed ({processed_count} with embeddings)")
            
            except Exception as e:
                print(f"   ⚠️  Error on line {line_num}: {e}")
    
    # Save results
    output_file = os.path.join(OUTPUT_DIR, "embeddings_data.jsonl")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Generated {processed_count} embeddings")
    print(f"   Saved to: {output_file}")

if __name__ == "__main__":
    process_embeddings()

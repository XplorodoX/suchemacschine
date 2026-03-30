#!/usr/bin/env python3
"""
Prepare Starplan Semester Data for Embeddings
Generiert Embeddings für jedes Semester separat
"""

import json
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load embedding model
logger.info("Loading SentenceTransformer model...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
logger.info(f"Model loaded: {model.get_sentence_embedding_dimension()} dimensions")


def prepare_semester_data(semester_code: str, input_file: str):
    """Prepare embeddings für ein einzelnes Semester"""
    
    logger.info(f"\n📚 Processing {semester_code}...")
    
    # Load raw data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    semester_data = data[semester_code] if isinstance(data, dict) and semester_code in data else data
    
    # Process lectures
    indexed_records = []
    lectures = semester_data.get('lectures', [])
    
    logger.info(f"  Preparing {len(lectures)} lectures for embedding...")
    
    for i, lecture in enumerate(lectures, 1):
        if i % 50 == 0 or i == 1:
            logger.info(f"    [{i}/{len(lectures)}] Processing...")
        
        # Create embedding text
        text = f"{lecture.get('name', '')} {lecture.get('lecturer', '')} {lecture.get('day', '')} {lecture.get('time', '')} {lecture.get('room', '')}"
        text = text.strip()
        
        if not text:
            continue
        
        # Generate embedding
        embedding = model.encode(text, convert_to_tensor=False).tolist()
        
        # Create record
        record = {
            'id': f"{semester_code}_{i}",
            'semester': semester_code,
            'content': text,
            'metadata': {
                'title': lecture.get('name', ''),
                'lecturer': lecture.get('lecturer', ''),
                'day': lecture.get('day', ''),
                'time': lecture.get('time', ''),
                'room': lecture.get('room', ''),
                'semester': semester_code,
                'type': 'timetable'
            },
            'embedding': embedding
        }
        indexed_records.append(record)
    
    # Save indexed records
    output_file = f"starplan_{semester_code}_indexed_data.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in indexed_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    logger.info(f"  ✅ Generated {len(indexed_records)} embeddings for {semester_code}")
    logger.info(f"  ✓ Saved to {output_file}")
    
    return indexed_records


def main():
    # Check if we have multi-semester data
    if Path("starplan_all_semesters_data.json").exists():
        logger.info("📖 Processing multi-semester data...")
        
        with open("starplan_all_semesters_data.json", 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        for semester_code in all_data.keys():
            prepare_semester_data(semester_code, "starplan_all_semesters_data.json")
        
        logger.info("\n✅ All semesters processed!")
    else:
        logger.warning("⚠️  starplan_all_semesters_data.json not found")
        logger.info("Run starplan_multi_semester_scraper.py first")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Prepare Starplan timetable data for indexing in Qdrant

Converts starplan_data.json to indexed records with embeddings
"""

import json
import logging

from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def prepare_starplan_data():
    """Load Starplan data and prepare for indexing"""
    
    logger.info("Loading starplan_data.json...")
    with open('starplan_data.json', 'r', encoding='utf-8') as f:
        starplan_data = json.load(f)
    
    # Create records for each lecture
    records = []
    
    logger.info("Creating lecture records...")
    
    for prog_id, timetable in starplan_data['timetables'].items():
        program_name = timetable['program_name']
        
        for lecture in timetable['lectures']:
            # Create a structured record for this lecture
            record = {
                'source': 'starplan_timetable',
                'program': program_name,
                'program_id': prog_id,
                'day': lecture.get('day', ''),
                'time': lecture.get('time', ''),
                'lecture_info': lecture.get('info', ''),
                'extracted_date': starplan_data['extracted_at'],
                'type': 'timetable'
            }
            
            # Create searchable text combining all fields
            text_parts = [
                f"Studiengang: {program_name}",
                f"Tag: {lecture.get('day', '')}",
                f"Zeit: {lecture.get('time', '')}",
                f"Informationen: {lecture.get('info', '')}"
            ]
            
            record['full_text'] = " | ".join(text_parts)
            record['url'] = f"https://vorlesungen.htw-aalen.de/splan/mobile?lan=de&sel=pg&og={prog_id}&pu=50&act=tt"
            
            records.append(record)
    
    logger.info(f"Created {len(records)} lecture records")
    
    return records

def generate_embeddings(records):
    """Generate sentence embeddings for all records"""
    
    logger.info("\nLoading SentenceTransformer model (this may take a moment)...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    logger.info("Generating embeddings...")
    
    texts = [record['full_text'] for record in records]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Attach embeddings to records
    for i, record in enumerate(records):
        record['embedding'] = embeddings[i].tolist()
    
    logger.info(f"Generated embeddings for {len(records)} records")
    
    return records

def save_indexed_data(records, filename='starplan_indexed_data.jsonl'):
    """Save records in JSONL format (one JSON object per line)"""
    
    logger.info(f"\nSaving indexed data to {filename}...")
    
    with open(filename, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    logger.info(f"✓ Saved {len(records)} records")

def main():
    logger.info("=" * 70)
    logger.info("Starplan Data Preparation for Qdrant")
    logger.info("=" * 70 + "\n")
    
    # Step 1: Prepare records
    records = prepare_starplan_data()
    
    # Show samples before embedding
    logger.info("\nSample records (first 3):")
    for i, record in enumerate(records[:3], 1):
        logger.info(f"\n  {i}. {record['program']} - {record['time']}")
        logger.info(f"     {record['lecture_info'][:80]}")
    
    # Step 2: Generate embeddings
    records = generate_embeddings(records)
    
    # Step 3: Save  
    save_indexed_data(records)
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ Done! Data ready for Qdrant indexing")
    logger.info("=" * 70)
    logger.info("\nNext step: Use scripts/index_to_qdrant.py to index starplan_indexed_data.jsonl")

if __name__ == "__main__":
    main()

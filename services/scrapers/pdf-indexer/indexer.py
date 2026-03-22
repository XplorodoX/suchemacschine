#!/usr/bin/env python3
"""
PDF Indexer Service
Extrahiert Text aus PDFs und speichert als JSONL
"""
import os
import json
import glob
from datetime import datetime
from pathlib import Path

import pdfplumber

PDF_SOURCES = os.getenv("PDF_SOURCES", "/pdf_sources")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/data")

def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF."""
    try:
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    text_parts.append(f"[Page {page_num}] {text}")
        
        return "\n".join(text_parts)
    except Exception as e:
        print(f"   ⚠️  Error extracting {pdf_path}: {e}")
        return ""

def index_pdfs():
    """Main PDF indexing function."""
    print(f"📄 Starting PDF indexer at {datetime.now()}")
    print(f"   PDF sources: {PDF_SOURCES}")
    
    # Find all PDF files
    pdf_files = []
    for root, dirs, files in os.walk(PDF_SOURCES):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    print(f"   Found {len(pdf_files)} PDF files")
    
    if not pdf_files:
        print("   ⚠️  No PDF files found")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"   [{i}/{len(pdf_files)}] Processing: {os.path.basename(pdf_file)}")
        
        text = extract_pdf_text(pdf_file)
        
        if text and len(text.strip()) > 100:
            results.append({
                "url": f"file://{os.path.abspath(pdf_file)}",
                "title": os.path.basename(pdf_file),
                "text": text,
                "scraped_at": datetime.now().isoformat(),
                "source": "pdf"
            })
    
    # Save results
    output_file = os.path.join(OUTPUT_DIR, "pdf_data.jsonl")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Indexed {len(results)} PDFs")
    print(f"   Saved to: {output_file}")

if __name__ == "__main__":
    index_pdfs()

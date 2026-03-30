import asyncio
import json
import logging
import re
import os
from typing import List, Dict, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

COLLECTION_NAME = "hs_aalen_hybrid"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

URLS = [
    "https://www.hs-aalen.de/person/harald-riegel",
    "https://www.hs-aalen.de/hochschule/leitung-und-organisation",
    "https://www.hs-aalen.de/hochschule/leitung-und-organisation/rektorat",
    "https://www.hs-aalen.de/hochschule/leitung-und-organisation/rektorat/ansprechpersonen"
]

GERMAN_STOPWORDS = {"der", "die", "das", "ein", "eine", "einer", "einem", "einen", "und", "oder", "aber", "mit", "ohne", "auf", "in", "im", "am", "an", "zu", "zum", "zur", "von", "für", "ist", "sind", "war", "wie", "was", "wer", "wo", "wann", "warum", "welche", "welcher", "welches", "ich", "du", "er", "sie", "es", "wir", "ihr", "nicht", "kein", "keine", "mehr", "auch", "den", "dem", "des", "bei", "über", "unter", "nach"}

def normalize_text(t: str) -> str:
    t = (t or "").lower()
    return re.sub(r"\s+", " ", t).strip()

def tokenize(t: str) -> List[str]:
    return [w for w in re.findall(r"[a-zA-Z0-9äöüß]{2,}", normalize_text(t)) if w not in GERMAN_STOPWORDS]

def sparse_encode(text: str) -> models.SparseVector:
    tokens = tokenize(text)
    if not tokens: return models.SparseVector(indices=[], values=[])
    counts = {}
    for tok in tokens:
        idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % 1000000
        counts[idx] = counts.get(idx, 0) + 1.0
    return models.SparseVector(indices=list(counts.keys()), values=list(counts.values()))

async def scrape_and_index():
    print("Loading models...")
    st_model = SentenceTransformer(MODEL_NAME)
    qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        for url in URLS:
            print(f"Scraping {url}...")
            try:
                res = await client.get(url)
                if res.status_code != 200:
                    print(f"Error {res.status_code} for {url}")
                    continue
                
                soup = BeautifulSoup(res.text, 'html.parser')
                for tag in soup(['script', 'style']):
                    tag.decompose()
                
                title_tag = soup.find('h1') or soup.find('title')
                title = title_tag.get_text(strip=True) if title_tag else "No title"
                
                content = soup.get_text(separator=' ', strip=True)
                content = re.sub(r'\s+', ' ', content)
                content = content[:5000] # Increased limit for better context
                
                print(f"Indexing {title}...")
                
                # We need to create a unique ID based on URL
                point_id = hashlib.md5(url.encode()).hexdigest()
                
                dense_vec = st_model.encode(content).tolist()
                sparse_vec = sparse_encode(content)
                
                qc.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[
                        models.PointStruct(
                            id=point_id,
                            vector={
                                "dense": dense_vec,
                                "sparse": sparse_vec
                            },
                            payload={
                                "url": url,
                                "title": title,
                                "content": content,
                                "source": "hs_aalen",
                                "type": "webpage"
                            }
                        )
                    ]
                )
                print(f"✓ Indexed {url}")
                
            except Exception as e:
                print(f"Failed to scrape {url}: {e}")

if __name__ == "__main__":
    asyncio.run(scrape_and_index())

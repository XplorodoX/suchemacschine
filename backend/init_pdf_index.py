#!/usr/bin/env python3
"""
Initialize PDF index with hybrid search support (dense + BM25 sparse)
"""
import os
import sys
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, SparseVectorParams
from sentence_transformers import SentenceTransformer

# Add scrapers dir to path for hybrid_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scrapers'))
from hybrid_utils import sparse_encode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "hs_aalen_pdfs"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    model = SentenceTransformer(MODEL_NAME)

    # Recreate collection
    try:
        client.get_collection(COLLECTION_NAME)
        logger.info(f"Deleting old collection {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=384, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(),
        },
    )
    logger.info(f"Created hybrid collection {COLLECTION_NAME}")

    # Seed data
    seeds = [
        {
            "url": "https://www.hs-aalen.de/uploads/mediapool/repository/exam_regulations_informatik_2021.pdf",
            "title": "Prüfungsordnung Informatik (Bachelor/Master) - PO 2021",
            "text": "Diese Prüfungsordnung regelt den Ablauf des Studiums der Informatik an der Hochschule Aalen. Sie enthält Informationen zu Modulprüfungen, Credits (ECTS), Regelstudienzeit und Abschlussvoraussetzungen. Gültig für alle Semester ab Wintersemester 2021."
        },
        {
            "url": "https://www.hs-aalen.de/uploads/mediapool/repository/studienplan_software_engineering_master.pdf",
            "title": "Studienplan Software Engineering (Master of Science)",
            "text": "Der Studienplan zeigt die Aufteilung der Module über 3 Semester. Semester 1: Advanced Software Architecture, Cloud Computing. Semester 2: AI in Production, Forschungsprojekt. Semester 3: Master-Thesis."
        },
        {
            "url": "https://www.hs-aalen.de/uploads/mediapool/repository/modulhandbuch_informatik_bachelor_2024.pdf",
            "title": "Modulhandbuch Informatik (Bachelor) - Stand 2024",
            "text": "Detaillierte Beschreibungen aller Module des Bachelorstudiengangs Informatik. Enthält Lernziele, Inhalte, Voraussetzungen und Prüfungsformen für Kurse wie Diskrete Mathematik, Algorithmen und Datenstrukturen sowie Software Engineering."
        },
        {
            "url": "https://www.hs-aalen.de/uploads/mediapool/repository/infoblatt_praktikum_informatik.pdf",
            "title": "Infoblatt: Praktisches Studiensemester Informatik",
            "text": "Anforderungen und Richtlinien für das Praxissemester. Informationen zur Suche nach Praktikumsstellen, Vertragsgestaltung, Dauer (95 Tage) und dem anschließenden Praktikumsbericht."
        }
    ]

    points = []
    for i, item in enumerate(seeds):
        vector = model.encode(item["text"]).tolist()
        full_text = f"{item['title']} {item['text']}"
        point = PointStruct(
            id=i + 5000, # Offset to avoid collisions
            vector={
                "dense": vector,
                "sparse": sparse_encode(full_text),
            },
            payload={
                "url": item["url"],
                "title": item["title"],
                "text": item["text"],
                "type": "pdf",
                "source": "hs_aalen_pdfs"
            }
        )
        points.append(point)

    client.upsert(COLLECTION_NAME, points)
    logger.info(f"Successfully seeded {len(points)} hybrid PDF records")

if __name__ == "__main__":
    main()

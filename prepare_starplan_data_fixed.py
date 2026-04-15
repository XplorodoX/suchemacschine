#!/usr/bin/env python3
"""
Prepare Starplan timetable data for indexing in Qdrant.
Consolidates fragmented lecture records into clean, structured entries.
"""

import json
import os
import sys
from datetime import datetime
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
INPUT_FILE = os.getenv("STARPLAN_INPUT", "starplan_data.json")
OUTPUT_FILE = os.getenv("STARPLAN_OUTPUT", "starplan_indexed_data.jsonl")


def consolidate_lectures(raw_timetables: dict):
    """
    The raw scraper produces 4+ entries per lecture slot (room, name, prof, group).
    This function groups them into one clean record per actual lecture.
    """
    lectures = []

    for prog_id, timetable in raw_timetables.items():
        program_name = timetable["program_name"]
        raw_lectures = timetable.get("lectures", [])

        # Group by (day, time) — all 4 fragments share the same day+time
        slots = {}
        for entry in raw_lectures:
            day = entry.get("day", "").strip()
            time = entry.get("time", "").strip()
            info = entry.get("info", "").strip()

            # Skip clearly useless entries
            if not info or info in ("Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag"):
                continue
            if info.startswith("Powered by") or info.startswith("Letzte Änderung"):
                continue
            if len(info) <= 5:
                continue

            key = (day, time)
            slots.setdefault(key, []).append(info)

        # For each (day, time) group, try to extract structured fields
        for (day, time), fragments in slots.items():
            if not day or not time or time == "17:00" and len(fragments) <= 2:
                continue

            lecture = _parse_fragments(fragments, program_name, day, time, prog_id)
            if lecture:
                lectures.append(lecture)

    return lectures


def _parse_fragments(fragments, program: str, day: str, time: str, prog_id: str):
    """
    Given a list of info fragments for one slot, extract:
    - room, lecture name, lecturer, group
    and build a clean searchable text.
    """
    room = ""
    name = ""
    lecturer = ""
    group = ""

    for fragment in fragments:
        # Lecturer pattern: "Firstname. Lastname" or "Prof. Dr. ..."
        if any(c in fragment for c in [".", ","]) and len(fragment.split()) >= 2:
            # Check if it's likely a person name
            parts = fragment.split(",")
            if all(len(p.strip()) < 30 for p in parts):
                # Could be "M. Hermann" or "J. Geiger, K. Müller"
                if any(ch.isupper() for ch in fragment[:3]):
                    lecturer = fragment
                    continue

        # Group pattern: ends with semester info like "AI S1", "BAN S4"
        if fragment.endswith(tuple([f"S{i}" for i in range(1, 10)])) or "S1+" in fragment or "S1_" in fragment:
            group = fragment
            continue

        # Room pattern: starts with building code (G1, G2, AH, WIN, etc.) or is a room number
        if fragment.split()[0] in ("G1", "G2", "G3", "G4", "AH", "WIN", "231", "111", "114", "129", "130", "131") or \
           (len(fragment) <= 20 and any(c.isdigit() for c in fragment[:4])):
            room = fragment
            continue

        # Module name pattern: usually contains "(XX-XXXXX)" module number
        if "(" in fragment and "-" in fragment:
            name = fragment
            continue

        # Fallback: if long enough, treat as name
        if not name and len(fragment) > 10:
            name = fragment

    if not name:
        return None  # Skip unclear entries

    # Build a rich searchable text combining all fields
    parts = [
        f"Studiengang: {program}",
        f"Vorlesung: {name}",
        f"Tag: {day}",
        f"Zeit: {time}",
    ]
    if room:
        parts.append(f"Raum: {room}")
    if lecturer:
        parts.append(f"Dozent: {lecturer}")
    if group:
        parts.append(f"Gruppe: {group}")

    return {
        "program": program,
        "program_id": prog_id,
        "name": name,
        "day": day,
        "time": time,
        "room": room,
        "lecturer": lecturer,
        "group": group,
        "full_text": " | ".join(parts),
        "url": f"https://vorlesungen.htw-aalen.de/splan/mobile?lan=de&sel=pg&og={prog_id}&pu=50&act=tt",
        "source": "starplan_timetable",
        "type": "timetable",
    }


def main():
    logger.info(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        starplan_data = json.load(f)

    raw_timetables = starplan_data.get("timetables", {})
    logger.info(f"Raw timetables: {len(raw_timetables)} programs")

    lectures = consolidate_lectures(raw_timetables)
    logger.info(f"Consolidated to {len(lectures)} clean lecture records")

    logger.info(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [l["full_text"] for l in lectures]
    logger.info(f"Generating embeddings for {len(texts)} records...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    logger.info(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for lecture, embedding in zip(lectures, embeddings):
            record = {**lecture, "embedding": embedding.tolist()}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Done! {len(lectures)} records written.")
    logger.info(f"Embedding dimension: {len(embeddings[0])}")

    # Print a few examples
    logger.info("\nSample records:")
    for r in lectures[:3]:
        logger.info(f"  [{r['program']}] {r['name']} — {r['day']} {r['time']} — {r['room']}")


if __name__ == "__main__":
    main()

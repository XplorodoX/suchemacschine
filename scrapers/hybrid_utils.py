import re
import unicodedata
from typing import List

GERMAN_STOPWORDS = {"der", "die", "das", "ein", "eine", "einer", "einem", "einen", "und", "oder", "aber", "mit", "ohne", "auf", "in", "im", "am", "an", "zu", "zum", "zur", "von", "für", "ist", "sind", "war", "wie", "was", "wer", "wo", "wann", "warum", "welche", "welcher", "welches", "ich", "du", "er", "sie", "es", "wir", "ihr", "nicht", "kein", "keine", "mehr", "auch", "den", "dem", "des", "bei", "über", "unter", "nach"}

def normalize_text(t: str) -> str:
    t = (t or "").lower()
    t = unicodedata.normalize("NFKC", t)
    return re.sub(r"\s+", " ", t).strip()

def tokenize(t: str) -> List[str]:
    return [w for w in re.findall(r"[a-zA-Z0-9äöüß]{2,}", normalize_text(t)) if w not in GERMAN_STOPWORDS]

def sparse_encode(text: str):
    tokens = tokenize(text)
    if not tokens:
        return {"indices": [], "values": []}
    counts = {}
    for tok in tokens:
        idx = hash(tok) % 1000000
        counts[idx] = counts.get(idx, 0) + 1.0
    return {
        "indices": list(counts.keys()),
        "values": list(counts.values())
    }

# Starplan Semester-Variationen - Dokumentation

## Überblick

Das System unterstützt nun Starplan-Daten für **mehrere Semester** separat. Jedes Semester wird als separate Qdrant-Collection indexiert und kann einzeln durchsucht werden.

## Verfügbare Semester

```
SoSe26  →  Sammersemester 2026 (aktiv)
WS25    →  Wintersemester 2025/26
SoSe25  →  Sommersemester 2025
WS24    →  Wintersemester 2024/25
```

## Setup & Ausführung

### 1. Multi-Semester Scraping

```bash
cd /Users/merluee/Desktop/suchemacschine
source .venv/bin/activate

python starplan_multi_semester_scraper.py
```

**Ergebnis:** 
- `starplan_all_semesters_data.json` (all semesters combined)
- `starplan_SoSe26_data.json`, `starplan_WS25_data.json`, etc. (individual files)

**Duration:** ~30-45 minutes (Depends on Starplan responsiveness)

### 2. Generate Embeddings for All Semesters

```bash
python prepare_starplan_semesters_data.py
```

**Erzeugt:**
- `starplan_SoSe26_indexed_data.jsonl`
- `starplan_WS25_indexed_data.jsonl`
- `starplan_SoSe25_indexed_data.jsonl`
- `starplan_WS24_indexed_data.jsonl`

**Duration:** ~5-10 minutes

### 3. Index to Qdrant

```bash
python index_starplan_semesters_to_qdrant.py
```

**Erzeugt separate Qdrant Collections:**
- `starplan_SoSe26` (Sommersemester 2026)
- `starplan_WS25` (Wintersemester 2025/26)
- `starplan_SoSe25` (Sommersemester 2025) 
- `starplan_WS24` (Wintersemester 2024/25)

**Verifikation:**
```bash
python3 << 'EOF'
from qdrant_client import QdrantClient
client = QdrantClient("localhost", port=6333)
collections = client.get_collections()
for c in collections.collections:
    if c.name.startswith('starplan_'):
        print(f"{c.name}: {c.points_count} points")
EOF
```

## API Integration

### Search mit Semester-Filter

Die REST API unterstützt nun den `semester` Parameter:

```bash
# Suche mit Standardsemester (SoSe26)
curl "http://localhost:8000/api/search?q=Informatik%20Vorlesung"

# Suche mit spezifischem Semester
curl "http://localhost:8000/api/search?q=Informatik%20Vorlesung&semester=WS25"

# Alle unterstützten Semester
curl "http://localhost:8000/api/search?q=Informatik%20Vorlesung&semester=SoSe26"
curl "http://localhost:8000/api/search?q=Informatik%20Vorlesung&semester=WS25"
curl "http://localhost:8000/api/search?q=Informatik%20Vorlesung&semester=SoSe25"
curl "http://localhost:8000/api/search?q=Informatik%20Vorlesung&semester=WS24"
```

### API Response

```json
{
  "results": [...],
  "total": 42,
  "query": "Informatik Vorlesung",
  "semester": "SoSe26",
  "page": 1,
  "per_page": 10
}
```

## Frontend Integration

### HTML/JS für Semester-Auswahl

```html
<select id="semesterSelect" onchange="updateSemester()">
  <option value="SoSe26">Sommersemester 2026</option>
  <option value="WS25">Wintersemester 2025/26</option>
  <option value="SoSe25">Sommersemester 2025</option>
  <option value="WS24">Wintersemester 2024/25</option>
</select>

<script>
function updateSemester() {
  const semester = document.getElementById('semesterSelect').value;
  const query = document.getElementById('searchInput').value;
  performSearch(query, semester);
}

async function performSearch(query, semester = 'SoSe26') {
  const response = await fetch(
    `/api/search?q=${encodeURIComponent(query)}&semester=${semester}`
  );
  const data = await response.json();
  displayResults(data);
}
</script>
```

## Backend Logik

### Fallback Behavior

Falls die semester-spezifische Collection nicht verfügbar ist, fällt das System automatisch auf die Standard-Collection (`starplan_timetable`) zurück.

```python
# In hybrid_search() function:
semester_collection = f"starplan_{semester}"

try:
    # Try semester-specific collection first
    timetable_results = client.query_points(
        collection_name=semester_collection,
        query=original_vector,
        limit=int(total_limit * 0.15),
    ).points
except Exception as e:
    # Fallback to main timetable
    timetable_results = client.query_points(
        collection_name="starplan_timetable",
        query=original_vector,
        limit=int(total_limit * 0.15),
    ).points
```

## Daten-Struktur

### Raw Data (JSON)

```json
{
  "semester": "SoSe26",
  "semester_name": "Sommersemester 2026",
  "start_date": "2026-03-22",
  "programs": [...],
  "timetables": {...},
  "lectures": [
    {
      "day": "Montag",
      "time": "09:00-10:30",
      "name": "Mathematik I",
      "lecturer": "Prof. Dr. Example",
      "room": "Raum 1.001",
      "info": "..."
    }
  ]
}
```

### Indexed Data (JSONL)

```jsonl
{"id": "SoSe26_1", "semester": "SoSe26", "content": "...", "metadata": {...}, "embedding": [...]}
{"id": "SoSe26_2", "semester": "SoSe26", "content": "...", "metadata": {...}, "embedding": [...]}
```

### Qdrant Payload

```json
{
  "title": "Mathematik I",
  "lecturer": "Prof. Dr. Example",
  "day": "Montag",
  "time": "09:00-10:30",
  "room": "Raum 1.001",
  "semester": "SoSe26",
  "type": "timetable",
  "content": "Mathematik I Prof. Dr. Example Montag 09:00-10:30 Raum 1.001"
}
```

## Workflow für neue Semester

Falls künftig ein neues Semester (z.B. SoSe27) hinzugefügt werden soll:

### 1. Update `starplan_multi_semester_scraper.py`

```python
SEMESTERS = {
    'SoSe27': {'start_date': '2027-03-22', 'name': 'Sommersemester 2027'},
    'SoSe26': {...},
    # ...
}
```

### 2. Update `index_starplan_semesters_to_qdrant.py`

```python
SEMESTER_COLLECTIONS = {
    'SoSe27': 'starplan_SoSe27',
    'SoSe26': 'starplan_SoSe26',
    # ...
}
```

### 3. Update API Default (optional)

```python
semester: str = Query("SoSe27")  # New default semester
```

### 4. Run Pipeline

```bash
python starplan_multi_semester_scraper.py
python prepare_starplan_semesters_data.py
python index_starplan_semesters_to_qdrant.py
```

## Performance

| Operation | Duration |
|-----------|----------|
| Multi-semester scraping (4x89 programs) | ~1.5 hours |
| Embedding generation (all semesters) | ~10 minutes |
| Indexing to Qdrant | ~5 minutes |
| Per-semester search query | <100ms |

## Troubleshooting

### Collection nicht gefunden

```bash
# Verify collections exist
python index_starplan_semesters_to_qdrant.py
```

### Leere Ergebnisse für spezifisches Semester

Prüfe ob die Collection existiert und Daten enthält:

```bash
python3 << 'EOF'
from qdrant_client import QdrantClient
client = QdrantClient("localhost", port=6333)
points = client.get_collection("starplan_SoSe26")
print(f"Points in collection: {points.points_count}")
EOF
```

## Nächste Schritte

- [ ] UI mit Semester-Dropdown integrieren
- [ ] Query-Expansion für Semester-spezifische Begriffe
- [ ] Semester-spezifische Icons/Farben im Frontend
- [ ] Combine API: Alle Semester in einer Abfrage durchsuchen
- [ ] Archiv älterer Semester (WS24, etc.)

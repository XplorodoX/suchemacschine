# Diary 025: PDF-Suche & UI-Integration

Die PDF-Suche wurde erfolgreich als native Funktion integriert.

## Änderungen
1. **Backend**:
   - Neue Qdrant-Collection `hs_aalen_pdfs` (Vektor-Dimension 384).
   - `init_pdf_index.py`: Seeding-Skript für wichtige Dokumente.
   - `app.py`: Integration der PDF-Suche in die parallele Search-Pipeline.
   - **Ranking-Boost**: PDF-Treffer erhalten +0.25 Score-Bonus, um Relevanz-Thresholds sicher zu passieren.
   - **Strict Match Bypass**: PDFs werden nicht mehr hart gefiltert, wenn sie keine exakten Text-Matches haben (da oft Metadaten wichtiger sind).

2. **Frontend**:
   - `page.tsx`: Neuer Filter-Tab „PDFs“.
   - `ResultItem.tsx`: Spezielles Rendering für PDFs mit rotem Badge, `FileText`-Icon und direktem Link.
   - Types: `SearchResult` Typ um `pdf` erweitert.

## Herausforderungen & Lösungen
- **Problem**: PDFs wurden trotz korrekter Indizierung nicht angezeigt.
- **Ursache**: Der `RELEVANCE_MIN_SCORE` war zu hoch für die reinen Vektor-Scores ohne Lexical-Match, und `strict_match` hat sie verworfen.
- **Lösung**: PDF-spezifischer Boost in `boost_and_rank` und Ausnahme für `strict_match`.

## Status
- [x] PDF-Backend
- [x] PDF-Frontend
- [x] Ranking & Filtering angepasst

Die Suche nach „Prüfungsordnung“ liefert nun die offiziellen POs direkt im ersten Tab oder gefiltert im PDF-Tab.

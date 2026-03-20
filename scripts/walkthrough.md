
## Optimierung der Suchqualität

Um die Genauigkeit massiv zu steigern, wurden folgende Maßnahmen umgesetzt:
1. **Multilinguales Embedding**: Wechsel auf paraphrase-multilingual-MiniLM-L12-v2 für vollen Support der deutschen Sprache.
2. **Hybrid-Search**: Kombination aus Original-Anfrage und LLM-erweiterter Anfrage für bessere Abdeckung.
3. **Keyword Boosting**: Zusätzliche Gewichtung von exakten Worttreffern in Text und URL.
4. **LLM Re-Ranking**: Die Top-Ergebnisse werden von DeepSeek auf Relevanz geprüft und neu sortiert, bevor sie angezeigt werden.

Das Ergebnis ist eine Suche, die nicht nur Vektoren vergleicht, sondern den Inhalt wirklich "versteht".


## Optimierung der Suchqualität

Um die Genauigkeit massiv zu steigern, wurden folgende Maßnahmen umgesetzt:
1. **Multilinguales Embedding**: Wechsel auf paraphrase-multilingual-MiniLM-L12-v2 für vollen Support der deutschen Sprache.
2. **Hybrid-Search**: Kombination aus Original-Anfrage und LLM-erweiterter Anfrage für bessere Abdeckung.
3. **Keyword Boosting**: Zusätzliche Gewichtung von exakten Worttreffern in Text und URL.
4. **LLM Re-Ranking**: Die Top-Ergebnisse werden von DeepSeek auf Relevanz geprüft und neu sortiert, bevor sie angezeigt werden.

Das Ergebnis ist eine Suche, die nicht nur Vektoren vergleicht, sondern den Inhalt wirklich "versteht".

## Erweiterte Sucheinstellungen

Die Suchoberfläche bietet nun volle Kontrolle über die KI-Pipeline:
- **Modellauswahl**: Ein dynamisches Dropdown zeigt alle lokal in Ollama verfügbaren Modelle an. So kann zwischen schnellen (z.B. Llama3) und präzisen (z.B. DeepSeek-R1) Modellen gewechselt werden.
- **Zusammenfassung an/aus**: Das LLM-Summary kann jederzeit deaktiviert werden, um nur die reinen Web-Ergebnisse zu sehen.
- **Re-Ranking an/aus**: Die intelligente Neusortierung der Top-Ergebnisse durch das LLM lässt sich umschalten. Ohne Re-Ranking wird die rein vektorbasierte Sortierung (mit Keyword-Boosting) verwendet.

Diese Regler erlauben es, die Balance zwischen Geschwindigkeit und Präzision perfekt auf die aktuelle Suche abzustimmen.

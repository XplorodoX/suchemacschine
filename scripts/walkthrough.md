
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

## User-Feedback-Loop (Kontinuierliche Verbesserung)

Um die Fakten-Genauigkeit der KI-Zusammenfassungen langfristig zu verbessern, wurde ein Feedback-System integriert:
- **Bewertung**: Unter jeder Zusammenfassung befinden sich nun "Daumen hoch" und "Daumen runter" Buttons.
- **Logging**: Jede Bewertung wird zusammen mit der Suchanfrage, der generierten Antwort und dem verwendeten Modell in einer `feedback.jsonl` Datei auf dem Server gespeichert.
- **Analyse**: Diese Daten ermöglichen es, systematische Fehler der KI zu identifizieren und die Prompts oder das Retrieval gezielt zu optimieren.

Damit ist der Grundstein für eine selbstlernende Suchmaschine gelegt, die mit jeder Benutzer-Interaktion besser wird.

## UX-Optimierungen & KI-Erweiterungssteuerung

Der Such-Workflow wurde weiter verfeinert:
- **Flüssige Paginierung**: Beim Wechseln zwischen den Ergebnisseiten (z.B. Seite 2) wird nur noch die Ergebnisliste aktualisiert. Das KI-Summary und das Seiten-Skelett bleiben stehen, was für ein deutlich ruhigeres und schnelleres Nutzererlebnis sorgt.
- **KI-Erweiterungs-Switch**: Nutzer können nun die "semantische Abfrageerweiterung" ein- oder ausschalten. 
  - **An (Standard)**: Das LLM generiert verwandte Suchbegriffe, um mehr Ergebnisse zu finden.
  - **Aus**: Es wird exakt nach dem gesucht, was eingegeben wurde (nützlich für spezifische Begriffe oder Code-Suchen).

Diese Kombination aus Performance und präziser Kontrolle macht die Suche extrem flexibel.

## Premium Feedback UI

Das Feedback-System wurde optisch aufgewertet:
- **Interaktive Buttons**: Große, abgerundete "Pill"-Buttons mit klaren "Ja/Nein"-Labels.
- **Micro-Animations**: Scale-Effekte beim Klicken und flüssige Hover-Animationen mit Ampel-Farben (Grün für korrekt, Rot für inkorrekt).
- **Nutzerführung**: Klare Tooltips und eine Bestätigungs-Animation nach der Stimmabgabe sorgen für ein wertiges Gefühl.

Damit ist das Einholen von Nutzer-Feedback nicht nur funktional, sondern macht auch Spaß – was die Datenqualität für spätere Optimierungen (RLHF/Prompt-Tuning) deutlich erhöht.

## Zero-Jump Pagination (Flüssiges Blättern)

Die Paginierung wurde technisch so umgesetzt, dass kein Layout-Sprung mehr entsteht:
- **State-Retention**: Die alten Ergebnisse bleiben als Orientierung sichtbar, während die neuen im Hintergrund geladen werden.
- **Atomic Update**: Der Austausch der Ergebnisliste erfolgt atomar in dem Moment, in dem die Daten vom Server eintreffen.
- **Visual Stability**: Durch den Verzicht auf das vollständige Leeren des Containers bleibt die Scroll-Position und das visuelle Gefüge stabil.

Damit fühlt sich die Suche jetzt so reaktionsschnell wie eine moderne Single-Page-App an.

## Finaler Polish: Scroll-Verhalten

Die Navigation zwischen den Seiten wurde für maximale Stabilität optimiert:
- **Kontext-sensitiver Scroll**: Nur bei einer komplett neuen Suche springt die Seite nach ganz oben. Beim Blättern (Seite 2+) scrollt die Ansicht sanft zum Anfang der Ergebnisliste, sodass die Suchleiste und die KI-Zusammenfassung im Blick bleiben, aber der Fokus sofort auf den neuen Treffern liegt.
- **Layout-Stabilität**: Durch das Vermeiden von unnötigen View-Switches und das atomare Update der Liste gibt es kein "Flackern" oder "Springen" mehr.

Damit ist die Benutzeroberfläche nun auf dem Niveau einer modernen Web-App.

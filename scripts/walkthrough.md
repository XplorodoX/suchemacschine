
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

## Performance-Turbo: Intelligentes Caching

Um die Wartezeiten bei aktivierten KI-Funktionen (Re-Ranking & Summary) auf ein Minimum zu reduzieren, wurde ein Server-seitiger Cache implementiert:
- **Redundanz-Vermeidung**: Teure LLM-Berechnungen (wie das Sortieren von 50 Ergebnissen) werden für jede Abfrage zwischengespeichert.
- **Instant Pagination**: Beim Klick auf Seite 2, 3 etc. kommen die Ergebnisse nun ohne jede Verzögerung direkt aus dem Arbeitsspeicher, da die gesamte Liste bereits beim ersten Aufruf verarbeitet wurde.
- **Ressourcen-Schonung**: Sowohl der Re-Ranker als auch die Zusammenfassung werden pro Suche nur noch einmal aufgerufen, was die Ollama-Instanz massiv entlastet.

Damit ist die Suche nun nicht mehr nur "smooth" im UI, sondern auch technisch extrem performant.

## Intelligente Snippets (Kontext-Vorschau)

Die Textvorschau unter den Suchergebnissen wurde radikal verbessert:
- **Kontext-Findung**: Das System sucht nun aktiv nach der Stelle im Dokument, an der deine Suchbegriffe vorkommen.
- **Sliding Window**: Statt einfach nur den Anfang der Seite anzuzeigen (wo oft nur Menüs oder Header stehen), wird ein Fenster von ca. 300 Zeichen um den Treffer herum ausgeschnitten.
- **Präzision**: Damit siehst du sofort, *warum* ein Ergebnis gefunden wurde und in welchem Zusammenhang dein Suchbegriff steht, noch bevor du auf den Link klickst.

Dies behebt das Problem von "unpassenden" Texten und macht die Ergebnisliste deutlich vertrauenswürdiger und hilfreicher.

## Authentische Vorschau-Texte

Um den Eindruck von "KI-generierten" Texten zu vermeiden, wurde die Extraktionslogik für Titel und Snippets verfeinert:
- **Präzise Titel**: Das System erkennt nun automatisch Navigations-Elemente (z.B. „Navigation überspringen“) und filtert diese aus. Der angezeigte Titel ist nun immer der erste inhaltlich relevante Satz der Seite.
- **Satz-basierte Snippets**: Vorschau-Texte beginnen nun bevorzugt am Anfang eines echten Satzes (nach einem Punkt), anstatt mitten im Wort oder Satz abzubrechen.
- **Original-Daten**: Die Texte stammen zu 100% aus dem originalen Webseiten-Inhalt ("Raw Scrape") und werden lediglich intelligent ausgeschnitten, um die maximale Relevanz zur Suchanfrage zu gewährleisten.

Dadurch wirken die Suchergebnisse nun so, wie man es von einer professionellen Suchmaschine erwartet – authentisch und direkt aus der Quelle.

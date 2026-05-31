#!/bin/bash
#
# reindex.sh — Baut alle Qdrant-Collections mit dem aktuellen Embedding-Modell
# (intfloat/multilingual-e5-base, 768-dim) neu auf.
#
# Notwendig nach einem Modellwechsel: die Vektordimension ändert sich (384 -> 768),
# und Qdrant-Collections haben eine feste Dimension. Die Index-Skripte löschen die
# alte Collection automatisch und legen sie passend neu an.
#
# Verwendung:
#   ./reindex.sh            # voller Lauf: bauen, scrapen+indexieren, Backend neu starten
#   ./reindex.sh --no-build # Images nicht neu bauen (schneller, falls schon aktuell)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Compose-Befehl automatisch erkennen (podman-compose bevorzugt, wie in deploy.sh)
if command -v podman-compose >/dev/null 2>&1; then
    COMPOSE="podman-compose"
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE="docker-compose"
elif docker compose version >/dev/null 2>&1; then
    COMPOSE="docker compose"
else
    echo "FEHLER: Weder podman-compose noch docker-compose gefunden." >&2
    exit 1
fi
echo "==> Verwende Compose-Befehl: $COMPOSE"

DO_BUILD=1
if [ "${1:-}" = "--no-build" ]; then
    DO_BUILD=0
fi

# Hilfsfunktion: Skript im scrapers-Container ausführen (Daten + Modell vorhanden)
run_scraper() {
    echo "    -> scrapers: python $1"
    $COMPOSE run --rm scrapers python "$1"
}

# Hilfsfunktion: Skript im backend-Container ausführen
run_backend() {
    echo "    -> backend: python $1"
    $COMPOSE run --rm backend python "$1"
}

if [ "$DO_BUILD" -eq 1 ]; then
    echo "==> Baue Images (backend + scrapers) mit e5-Modell neu..."
    $COMPOSE build backend scrapers
fi

echo "==> Stelle sicher, dass Qdrant läuft..."
$COMPOSE up -d qdrant
# Kurz warten, bis Qdrant Verbindungen annimmt
for i in $(seq 1 30); do
    if curl -sf http://localhost:6333/readyz >/dev/null 2>&1 \
       || curl -sf http://localhost:6333/ >/dev/null 2>&1; then
        echo "    Qdrant ist bereit."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "WARNUNG: Qdrant antwortet nach 30s nicht — versuche trotzdem fortzufahren." >&2
    fi
    sleep 1
done

echo ""
echo "==> Schritt 1/3: ASTA, Starplan, Webpage scrapen + prepare + indexieren (parallel)..."
# entspricht manage.py parallel -> Collections: asta_content,
# starplan_timetable, hs_aalen_website
$COMPOSE --profile scrapers run --rm scrapers parallel

echo ""
echo "==> Schritt 2/3: Restliche Collections..."
# Semester-Collections (z. B. starplan_SoSe26 — die nutzt die Live-Suche!)
run_scraper prepare_starplan_semesters_data.py
run_scraper index_starplan_semesters_to_qdrant.py
# hs_aalen_search: zuerst Embeddings mit dem neuen Modell neu berechnen.
# index_to_qdrant.py liest nur die fertigen Embeddings aus processed_data.jsonl —
# stammen die noch vom alten Modell (384-dim), schlägt der Upsert in die neue
# 768-dim-Collection fehl. prepare_data.py schreibt processed_data.jsonl neu.
run_backend prepare_data.py
run_backend index_to_qdrant.py
# hs_aalen_pdfs (init_pdf_index.py berechnet seine Embeddings selbst neu)
run_backend init_pdf_index.py

echo ""
echo "==> Schritt 3/3: Backend neu starten (lädt e5-Modell)..."
$COMPOSE up -d backend

echo ""
echo "==> Verifikation: Vektordimensionen der Collections (erwartet: 768)"
for coll in hs_aalen_search hs_aalen_website starplan_SoSe26 asta_content hs_aalen_pdfs; do
    size=$(curl -sf "http://localhost:6333/collections/${coll}" 2>/dev/null \
        | grep -o '"size":[0-9]*' | head -1 | grep -o '[0-9]*' || true)
    if [ -z "$size" ]; then
        echo "    ⚠️  ${coll}: nicht gefunden / leer"
    elif [ "$size" = "768" ]; then
        echo "    ✓ ${coll}: ${size}"
    else
        echo "    ✗ ${coll}: ${size} (erwartet 768 — bitte prüfen!)"
    fi
done

echo ""
echo "Fertig. Teste die Suche z. B. mit:"
echo "  curl 'http://localhost:6055/api/search?q=Prüfungsordnung+Informatik'"

#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Stoppe laufende Container..."
podman-compose down 2>/dev/null || true

echo "==> Entferne verwaiste Container (falls vorhanden)..."
for name in campusnow-mongo campusnow-scraper campusnow-api campusnow-g2-seeder \
            search-api-service search-frontend-service qdrant-service search-scrapers-service; do
    if podman container exists "$name" 2>/dev/null; then
        echo "    Entferne: $name"
        podman rm -f "$name"
    fi
done

echo "==> Baue und starte alle Services..."
podman-compose up -d --build

echo ""
echo "==> Status:"
podman-compose ps

echo ""
echo "Fertig! Erreichbar unter:"
echo "  Frontend:  http://$(hostname -I | awk '{print $1}'):6056"
echo "  Backend:   http://$(hostname -I | awk '{print $1}'):6055"
echo "  Qdrant:    http://$(hostname -I | awk '{print $1}'):6333"

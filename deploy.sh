#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Podman: unqualified-search-registries setzen damit lokale Images gefunden werden
REGISTRIES_CONF="${HOME}/.config/containers/registries.conf"
if [ ! -f "$REGISTRIES_CONF" ]; then
    mkdir -p "$(dirname "$REGISTRIES_CONF")"
    cat > "$REGISTRIES_CONF" << 'EOF'
unqualified-search-registries = ["docker.io"]
EOF
    echo "==> registries.conf erstellt ($REGISTRIES_CONF)"
fi

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

echo "==> Baue Images..."
podman-compose build

echo "==> Starte alle Services..."
podman-compose up -d

echo ""
echo "==> Status:"
podman-compose ps

echo ""
echo "Fertig! Erreichbar unter:"
echo "  Frontend:  http://$(hostname -I | awk '{print $1}'):6056"
echo "  Backend:   http://$(hostname -I | awk '{print $1}'):6055"
echo "  Qdrant:    http://$(hostname -I | awk '{print $1}'):6333"

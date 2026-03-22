#!/bin/bash
# HS Aalen AI Search - Services Manager

set -e

SERVICES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SERVICES_DIR/docker-compose.prod.yml"

if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
elif command -v podman-compose >/dev/null 2>&1; then
    COMPOSE_CMD="podman-compose"
else
    echo "Neither docker-compose nor podman-compose found in PATH"
    exit 1
fi

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}===================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

compose() {
    if [[ "$COMPOSE_CMD" == "podman-compose" ]]; then
        export CONTAINER_SOCKET_PATH
        CONTAINER_SOCKET_PATH="/run/podman/podman.sock"
    fi
    "$COMPOSE_CMD" -f "$COMPOSE_FILE" "$@"
}

# Commands
case "${1:-help}" in
    build)
        print_header "Building all services..."
        cd "$SERVICES_DIR"
        compose build
        print_success "Services built successfully!"
        ;;
    
    up)
        print_header "Starting services..."
        cd "$SERVICES_DIR"
        compose up -d
        sleep 5
        compose ps
        print_success "Services started!"
        ;;
    
    down)
        print_header "Stopping services..."
        cd "$SERVICES_DIR"
        compose down
        print_success "Services stopped!"
        ;;
    
    ps)
        print_header "Service Status"
        cd "$SERVICES_DIR"
        compose ps
        ;;
    
    logs)
        print_header "Live Logs (Press Ctrl+C to exit)"
        cd "$SERVICES_DIR"
        compose logs -f "${2:-search-api}"
        ;;
    
    run-pipeline)
        print_header "Running data update pipeline..."
        cd "$SERVICES_DIR"
        
        echo ""
        echo "1️⃣  Website Scraper..."
        compose --profile pipeline run --rm website-scraper || print_error "Website scraper failed"
        
        echo ""
        echo "2️⃣  PDF Indexer..."
        compose --profile pipeline run --rm pdf-indexer || print_error "PDF indexer failed"
        
        echo ""
        echo "3️⃣  Timetable Scraper..."
        compose --profile pipeline run --rm timetable-scraper || print_error "Timetable scraper failed"
        
        echo ""
        echo "4️⃣  Text Processor..."
        compose --profile pipeline run --rm text-processor || print_error "Text processor failed"
        
        echo ""
        echo "5️⃣  Embeddings Service..."
        compose --profile pipeline run --rm embeddings-service || print_error "Embeddings failed"
        
        echo ""
        echo "6️⃣  Qdrant Indexer..."
        compose --profile pipeline run --rm qdrant-indexer || print_error "Indexing failed"
        
        print_success "Pipeline complete!"
        ;;
    
    scrape)
        print_header "Running data scrapers..."
        cd "$SERVICES_DIR"
        compose --profile pipeline run --rm website-scraper
        compose --profile pipeline run --rm timetable-scraper
        print_success "Scraping complete!"
        ;;
    
    process)
        print_header "Processing scraped data..."
        cd "$SERVICES_DIR"
        compose --profile pipeline run --rm text-processor
        print_success "Processing complete!"
        ;;
    
    index)
        print_header "Indexing data into Qdrant..."
        cd "$SERVICES_DIR"
        compose --profile pipeline run --rm embeddings-service
        compose --profile pipeline run --rm qdrant-indexer
        print_success "Indexing complete!"
        ;;
    
    status)
        print_header "System Status"
        echo "🐳 Docker Status:"
        compose ps
        
        echo ""
        echo "🌐 Services:"
        echo "  Search API: http://localhost:8055"
        echo "  Qdrant: http://localhost:6333/dashboard"
        
        echo ""
        echo "📊 Quick Health Checks:"
        echo -n "  API: "
        curl -s http://localhost:8055/docs > /dev/null && echo "✅ OK" || echo "❌ DOWN"
        echo -n "  Qdrant: "
        curl -s http://localhost:6333/health > /dev/null && echo "✅ OK" || echo "❌ DOWN"
        ;;
    
    shell)
        print_header "Opening shell in ${2:-search-api} container"
        cd "$SERVICES_DIR"
        compose exec "${2:-search-api}" /bin/bash
        ;;

    reset-bootstrap)
        print_header "Resetting first-run bootstrap marker"
        cd "$SERVICES_DIR"
        compose exec scheduler sh -c 'rm -f /state/bootstrap_complete'
        print_success "Bootstrap marker removed. Next scheduler start will run full initial pipeline."
        ;;
    
    *)
        echo "Usage: $0 {build|up|down|ps|logs|run-pipeline|scrape|process|index|status|shell|reset-bootstrap|help}"
        echo ""
        echo "Commands:"
        echo "  build           - Build all Docker images"
        echo "  up              - Start all services"
        echo "  down            - Stop all services"
        echo "  ps              - Show service status"
        echo "  logs [service]  - View live logs"
        echo "  run-pipeline    - Run complete update pipeline"
        echo "  scrape          - Run scrapers only"
        echo "  process         - Run text processing only"
        echo "  index           - Run indexing only"
        echo "  status          - Show system status"
        echo "  shell [service] - Open shell in container"
        echo "  reset-bootstrap - Force initial pipeline on next scheduler start"
        echo "  help            - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 build"
        echo "  $0 up"
        echo "  $0 logs search-api"
        echo "  $0 run-pipeline"
        ;;
esac

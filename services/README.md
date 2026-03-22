# HS Aalen AI Search - Microservices Architecture

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        SCHEDULER (APScheduler)                    │
│                    (Runs every 14 days automatically)             │
└─────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
         ┌──────────────────┐ ┌────────────┐ ┌────────────────┐
         │ Website Scraper  │ │ PDF        │ │ Timetable      │
         │ (Websites)       │ │ Indexer    │ │ Scraper        │
         └──────────────────┘ └────────────┘ └────────────────┘
                    │             │             │
                    └─────────────┼─────────────┘
                                  ▼
                    ┌──────────────────────────┐
                    │  Text Processor          │
                    │  (Chunking & Tokenize)   │
                    └──────────────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │ Embeddings Service       │
                    │ (via Ollama)             │
                    └──────────────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │ Qdrant Indexer           │
                    │ (Vector DB)              │
                    └──────────────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │ Search API (FastAPI)     │
                    │ Port 8055                │
                    └──────────────────────────┘
```

## 📦 Services

### Scrapers
- **website-scraper**: Crawls HS Aalen website (configurable depth)
- **pdf-indexer**: Extracts text from PDFs
- **timetable-scraper**: Fetches timetable data

### Processing
- **text-processor**: Chunks and tokenizes text
- **embeddings-service**: Generates vectors using Ollama

### Indexing
- **qdrant-indexer**: Indexes vectors into Qdrant

### API & Orchestration
- **search-api**: FastAPI server (Port 8055)
- **scheduler**: APScheduler (orchestrates everything)

### Data Store
- **qdrant**: Vector database (Port 6333)

## 🚀 Quick Start

### 1. Build all services
```bash
cd /Users/merluee/Desktop/suchemacschine
docker-compose -f services/docker-compose.prod.yml build
```

### 2. Start the stack
```bash
docker-compose -f services/docker-compose.prod.yml up -d
```

For Podman on macOS, export the socket path before starting:
```bash
podman machine ssh 'chmod 666 /run/podman/podman.sock /run/user/0/podman/podman.sock || true'
export CONTAINER_SOCKET_PATH=/run/podman/podman.sock
podman-compose -f services/docker-compose.prod.yml up -d
```

On a fresh setup, the scheduler now runs the full pipeline automatically once:
1. Website scraper
2. PDF indexer
3. Timetable scraper
4. Text processor
5. Embeddings service
6. Qdrant indexer

After this bootstrap, only the regular interval run is executed.

### 3. Check status
```bash
docker-compose -f services/docker-compose.prod.yml ps
```

### 4. Access services
- **Search API**: http://localhost:8055
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## 📅 Automatic Updates

The scheduler runs the complete pipeline every 14 days:
1. Scrape all data sources
2. Process texts (chunking, tokenization)
3. Generate embeddings via Ollama
4. Index into Qdrant
5. API automatically uses new data

First-run bootstrap state is persisted in the `scheduler_state` volume. To force a new bootstrap run, remove `/state/bootstrap_complete` in the scheduler container.

Change interval in `.env`:
```
SCHEDULER_INTERVAL_DAYS=14
```

## 🛠️ Manual Pipeline Run

Run individual services:
```bash
# Scrape websites only
docker-compose -f services/docker-compose.prod.yml --profile pipeline run --rm website-scraper

# Process texts only
docker-compose -f services/docker-compose.prod.yml --profile pipeline run --rm text-processor

# Full pipeline
docker-compose -f services/docker-compose.prod.yml --profile pipeline run --rm website-scraper
docker-compose -f services/docker-compose.prod.yml --profile pipeline run --rm pdf-indexer
docker-compose -f services/docker-compose.prod.yml --profile pipeline run --rm timetable-scraper
docker-compose -f services/docker-compose.prod.yml --profile pipeline run --rm text-processor
docker-compose -f services/docker-compose.prod.yml --profile pipeline run --rm embeddings-service
docker-compose -f services/docker-compose.prod.yml --profile pipeline run --rm qdrant-indexer
```

## 📊 Data Flow

```
Raw Data (Websites, PDFs, Timetables)
           ▼
    /data/raw/*.jsonl
           ▼
    [Text Processing]
           ▼
    /data/processed/*.jsonl (chunked & tokenized)
           ▼
    [Embeddings via Ollama]
           ▼
    /data/embeddings/*.jsonl (with vectors)
           ▼
    [Qdrant Indexer]
           ▼
    Qdrant Collections (searchable)
           ▼
    [Search API]
           ▼
       http://localhost:8055
```

## 🔧 Configuration

Edit `.env` file:
```
SCHEDULER_INTERVAL_DAYS=14
TARGET_URLS=https://www.hs-aalen.de
MAX_DEPTH=3
CHUNK_SIZE=512
OVERLAP=50
OLLAMA_MODEL=nomic-embed-text
OLLAMA_HOST=http://host.docker.internal:11434
```

## 📝 Logs

View service logs:
```bash
docker-compose -f services/docker-compose.prod.yml logs -f search-api
docker-compose -f services/docker-compose.prod.yml logs -f scheduler
```

## 🐛 Troubleshooting

### Ollama not found
Make sure Ollama is running:
```bash
lsof -i :11434
```

### Vector dimension mismatch
Check Ollama embedding model is loaded:
```bash
ollama list
ollama pull nomic-embed-text
```

### No search results
Check Qdrant has data:
```bash
curl http://localhost:6333/collections
```

## 📚 Files Structure
 
```
services/
├── docker-compose.prod.yml          # Main orchestration
├── .env                             # Configuration
├── scrapers/
│   ├── website-scraper/
│   ├── pdf-indexer/
│   └── timetable-scraper/
├── processing/
│   ├── text-processor/
│   └── embeddings/
├── indexing/
│   └── qdrant-indexer/
├── api/
│   └── search-api/
└── scheduler/
```

## ✅ Next Steps

1. Configure `.env` with your URLs and settings
2. Place PDFs in a volume mount
3. Start the stack
4. First pipeline runs automatically on fresh startup
5. Then recurring runs happen every 14 days

---

**Built with** ❤️ using **FastAPI**, **Qdrant**, **Ollama**, and **APScheduler**

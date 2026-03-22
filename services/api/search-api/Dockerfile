FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    qdrant-client \
    requests \
    aiofiles \
    numpy \
    pydantic

COPY app.py .
COPY static/ static/

ENV QDRANT_HOST=qdrant
ENV QDRANT_PORT=6333

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

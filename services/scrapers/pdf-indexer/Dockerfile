FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pypdf \
    pdfplumber \
    PyMuPDF

COPY indexer.py .

CMD ["python", "indexer.py"]

FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    qdrant-client \
    numpy

COPY indexer.py .

CMD ["python", "indexer.py"]

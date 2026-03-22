FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    requests \
    numpy

COPY embedder.py .

CMD ["python", "embedder.py"]

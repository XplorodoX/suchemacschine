FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    requests \
    beautifulsoup4 \
    lxml \
    selenium \
    playwright

RUN playwright install chromium

COPY scraper.py .

CMD ["python", "scraper.py"]

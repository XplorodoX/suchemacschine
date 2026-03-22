FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    requests \
    beautifulsoup4 \
    icalendar \
    lxml

COPY scraper.py .

CMD ["python", "scraper.py"]

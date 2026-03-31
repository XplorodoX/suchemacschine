FROM mcr.microsoft.com/playwright/python:v1.49.0-noble

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install browsers
RUN playwright install chromium

# Copy all scrapers and data management
COPY . .

# Environment variables
ENV QDRANT_HOST=qdrant
ENV QDRANT_PORT=6333
ENV DATA_DIR=/app/data

# Default command (does nothing, keeps container alive if needed or used as base)
ENTRYPOINT ["python", "manage.py"]

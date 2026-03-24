FROM docker.io/library/python:3.9-slim

WORKDIR /app

# Install system dependencies if any (none needed here but good practice)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies using pip
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy the application code
COPY *.py ./

# Environment Variables (Overridden by compose)
ENV QDRANT_HOST=qdrant
ENV QDRANT_PORT=6333
ENV OLLAMA_URL=http://host.containers.internal:11434/api/generate
ENV OLLAMA_MODEL=deepseek-r1:8b

# Expose the API port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

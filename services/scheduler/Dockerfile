FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    APScheduler \
    docker \
    requests

COPY scheduler.py .

CMD ["python", "scheduler.py"]

FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    nltk \
    spacy

RUN python -m spacy download de_core_news_sm

COPY processor.py .

CMD ["python", "processor.py"]

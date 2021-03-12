FROM tiangolo/uvicorn-gunicorn:python3.8-slim

RUN pip install --no-cache-dir fastapi nltk scikit-learn pandas tensorflow

COPY ./app /app
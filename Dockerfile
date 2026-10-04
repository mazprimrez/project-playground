# Playground API - every project's API in one small image (no model at runtime). Build from the repo root:
#   docker build -t playground-api .
#   docker run -p 7860:7860 -v $PWD/genrec/artifacts/lookup:/app/genrec/artifacts/lookup playground-api
# Data is never baked in (e.g. genrec's tables are MovieLens-derived): mount it, or on a Hugging Face Space set each
# project's env vars (genrec: GENREC_HF_REPO + HF_TOKEN for a private dataset repo).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/tmp/hf

WORKDIR /app

# one block per project: its package with the [serve] extras, its serving/ folder and its env
COPY genrec/pyproject.toml genrec/README.md genrec/
COPY genrec/src genrec/src
RUN pip install "./genrec[serve]"
COPY genrec/serving genrec/serving
ENV GENREC_LOOKUP_DIR=/app/genrec/artifacts/lookup

COPY app.py .

# Hugging Face Spaces run containers as a non-root user and expect port 7860
RUN useradd -m -u 1000 user && mkdir -p /app/genrec/artifacts && chown -R user /app
USER user
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

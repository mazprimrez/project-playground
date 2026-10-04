# Project Playground - the portfolio UI and every project's API in one small image (no model at runtime).
# Build from the repo root:
#   docker build -t playground .
#   docker run -p 8080:8080 -v $PWD/genrec/artifacts/lookup:/app/genrec/artifacts/lookup playground
# Data is never baked in (e.g. genrec's tables are MovieLens-derived): mount it, or set each project's env vars
# (genrec: GENREC_HF_REPO + HF_TOKEN for a private Hugging Face dataset repo).

# ---- stage 1: build the UI (web/ -> static files)
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-fund --no-audit
COPY web/ ./
RUN npm run build

# ---- stage 2: the Python server (app.py serves the UI at / and the APIs under /api)
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
COPY --from=web /web/dist web/dist

# runs as a non-root user; listens on $PORT (Cloud Run sets it; 8080 otherwise)
RUN useradd -m -u 1000 user && mkdir -p /app/genrec/artifacts && chown -R user /app
USER user
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]

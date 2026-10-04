# GenRec lookup API - small image, no model at runtime. Build from the genrec/ folder:
#   docker build -f serving/Dockerfile -t genrec-api .
#   docker run -p 7860:7860 -v $PWD/artifacts/lookup:/app/artifacts/lookup genrec-api
# The lookup tables (artifacts/lookup/) contain MovieLens-derived data and are not baked into the image:
# mount them, or on a Hugging Face Space set GENREC_HF_REPO (+ HF_TOKEN for a private dataset repo).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/tmp/hf GENREC_LOOKUP_DIR=/app/artifacts/lookup

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve]"
COPY serving ./serving

# Hugging Face Spaces run containers as a non-root user and expect port 7860
RUN useradd -m -u 1000 user && mkdir -p /app/artifacts && chown -R user /app
USER user
EXPOSE 7860
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "7860"]

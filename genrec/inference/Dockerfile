# GenRec live inference - Cloud Run service `genrec-model` (CPU). The model is downloaded at startup from the private
# Hugging Face model repo with the HF_TOKEN secret; nothing private is baked into the image.
#   gcloud run deploy genrec-model --source genrec/inference ...   (see .github/workflows/deploy-model.yml)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/tmp/hf
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch (the default wheel would pull ~2.5 GB of CUDA libraries), then the genrec package from GitHub
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install "genrec[torch] @ git+https://github.com/mazprimrez/project-playground@main#subdirectory=genrec" \
                fastapi "uvicorn[standard]"

RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
COPY --chown=user app.py .
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]

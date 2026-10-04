# Project Playground

A collection of self-contained experiments. Each folder is its own project with its own README, dependencies,
notebooks and tests - nothing is shared between them.

| project | what | highlights |
|---|---|---|
| [`genrec/`](genrec/) | **Generative recommendation**: Qwen2.5-0.5B fine-tuned to know the movie catalogue and predict what a user watches next, compared with SASRec, TensorFlow Recommenders and collaborative filtering on MovieLens-1M; a lookup API | HR@10 0.236 (LLM) vs 0.294 (SASRec) vs 0.024 (popularity), full ranking |

## Portfolio site and API

One server for everything: [`app.py`](app.py) serves the portfolio UI ([`web/`](web/), React + TypeScript + Vite)
at `/` and every project's API under `/api/<project>/`. It only mounts each project's own `serving/app.py`, which
still runs on its own; a project that fails to import or start answers 503, and the rest keeps working.

| URL | what |
|---|---|
| `/` , `/projects/<project>` | the portfolio UI |
| `/api` , `/api/health` , `/api/docs` | the list of projects, their status, the server's API docs |
| `/api/genrec/...` | genrec's API (docs: `/api/genrec/docs`) |

**Develop** (two terminals; the UI reloads on save and forwards `/api` to the Python server):

```bash
pip install -e "genrec[serve]" && uvicorn app:app --reload     # API on http://localhost:8000
cd web && npm install && npm run dev                            # UI on http://localhost:5173
```

**Run it like production** (one server, one URL):

```bash
cd web && npm run build && cd ..        # -> web/dist
uvicorn app:app                         # UI + API on http://localhost:8000
docker build -t playground .            # or: one image with both (port 8080 / $PORT, ready for Cloud Run)
```

**Deploy** (Google Cloud Run, project `project-playground-mazi`, region `asia-southeast2`). The data tables are not in
the image: they live in the private bucket `gs://project-playground-mazi-data`, mounted read-only at `/mnt/data`.

```bash
# update the data (e.g. a new genrec lookup table), then restart by redeploying
gcloud storage cp genrec/artifacts/lookup/*.csv gs://project-playground-mazi-data/genrec/lookup/
# rebuild and deploy the code (UI + API); .gcloudignore keeps data and models out of the upload
gcloud run deploy playground --source . --region=asia-southeast2 --project=project-playground-mazi
```

**Add a project:** one line in `PROJECTS` in `app.py`, its install block in the [`Dockerfile`](Dockerfile), and an
entry + page in [`web/src/projects/`](web/src/projects/). Tests: `pytest` from the repo root.

## Conventions

- One folder per experiment, installable on its own (`pip install -e <folder>`), with its own `README.md`.
- Datasets and trained models are never committed (`data/`, `artifacts/` are git-ignored): each project has a download
  script, and models go to the Hugging Face Hub.
- Notebooks keep their outputs, so results are readable on GitHub without re-running.

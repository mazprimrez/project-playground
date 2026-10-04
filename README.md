# Project Playground

A collection of self-contained experiments. Each folder is its own project with its own README, dependencies,
notebooks and tests - nothing is shared between them.

| project | what | highlights |
|---|---|---|
| [`genrec/`](genrec/) | **Generative recommendation**: Qwen2.5-0.5B fine-tuned to know the movie catalogue and predict what a user watches next, compared with SASRec, TensorFlow Recommenders and collaborative filtering on MovieLens-1M; a cold-start dataset (MovieTweetings); a lookup API | HR@10 0.236 (LLM) vs 0.294 (SASRec) vs 0.024 (popularity), full ranking |

## API

[`app.py`](app.py) serves every project's API from one server, each under its own prefix. It only mounts each
project's own `serving/app.py`, which still runs and deploys on its own. A project that fails to import or start
answers 503, and the other projects keep working.

```bash
pip install -e "genrec[serve]"
uvicorn app:app --reload          # index: http://localhost:8000/  health: /health  genrec: /genrec/docs
docker build -t playground-api .  # one image for all of them (Hugging Face Spaces-ready, port 7860)
```

To add a project, add one line to `PROJECTS` in `app.py` and add its install block to the [`Dockerfile`](Dockerfile).
Root tests: `pytest` from the repo root.

## Conventions

- One folder per experiment, installable on its own (`pip install -e <folder>`), with its own `README.md`.
- Datasets and trained models are never committed (`data/`, `artifacts/` are git-ignored): each project has a download
  script, and models go to the Hugging Face Hub.
- Notebooks keep their outputs, so results are readable on GitHub without re-running.

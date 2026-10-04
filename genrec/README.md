# GenRec: generative movie recommendation

**Qwen2.5-0.5B fine-tuned to know the movie catalogue and to predict what a user watches next**, compared against
SASRec, TensorFlow Recommenders and classic collaborative filtering on MovieLens-1M - plus a lookup API with every
model's recommendations and probabilities for the test users.

## Results (MovieLens-1M)

Next-movie prediction for 1,000 test users, each user's **last** rating held out. **Full ranking** over all ~3,700
movies (already-rated movies skipped) - much stricter than the "100 sampled negatives" protocol many papers use,
which is why the numbers are lower than e.g. the SASRec paper's.

| model | hit@1 | HR@10 | NDCG@10 |
|---|---|---|---|
| random | 0.001 | 0.005 | 0.003 |
| most popular | 0.005 | 0.024 | 0.013 |
| TFRS two-tower (user ID) | 0.013 | 0.057 | 0.030 |
| UserKNN (cosine) | 0.014 | 0.081 | 0.042 |
| ItemKNN (cosine) | 0.013 | 0.085 | 0.043 |
| ItemKNN, last 20 ratings | 0.021 | 0.141 | 0.071 |
| Qwen2.5-0.5B v1 (5 targets/user) | 0.041 | 0.160 | 0.093 |
| TFRS sequential (GRU, last 20) | 0.026 | 0.167 | 0.089 |
| **Qwen2.5-0.5B v2 (20 targets/user)** | **0.067** | **0.236** | **0.138** |
| SASRec-20 (last 20 ratings) | 0.066 | 0.263 | 0.152 |
| SASRec-200 | 0.080 | 0.294 | 0.171 |

With 1,000 test users the sampling uncertainty is about ±0.014 HR@10 and ±0.008 hit@1. SASRec and TFRS training
is not bit-for-bit repeatable (GPU / multi-threaded CPU): re-runs move their numbers by about that much.

What we learned:
- **Order matters**: models that read the recent sequence beat order-blind ones (CF, two-tower) by a wide margin.
- **Recommendation training data is the main lever for the LLM**: 5 → 20 next-movie examples per user took HR@10
  from 0.160 to 0.236 (+48%), closing most of the gap to SASRec-20, which sees the same 20 ratings.
- **2 epochs** over the same data is enough - validation loss rises from epoch 3 (overfitting).
- The fine-tuned model still knows the catalogue (exact-match recall on trained movies: director 97%, genres 91%,
  "which movie is this plot?" 98%) - things SASRec can't do.

## Layout

```
src/genrec/            the package (used by the notebooks and the API)
  data.py              MovieLens-format loading, leave-last-out split, test users, item index
  prompts.py           training samples + prompts - the exact text the model is trained on
  metrics.py           hit@1 / HR@10 / NDCG@10 with full ranking
  training.py          Qwen fine-tuning: answer-only loss, memory-saving trainer, recall checks
  lookup.py            the pre-computed recommendation tables for the API
  models/              baselines (random, popularity, ItemKNN, UserKNN), sasrec, qwen (inference), tfrs_models
notebooks/
  movielens/           data overview (interaction-analysis), Wikipedia matching (phase-1), Qwen training/
                       evaluation/scoring (Colab), SASRec, TFRS, baselines (+ the combined results table)
serving/               the lookup API (FastAPI) + Dockerfile
scripts/               data download, lookup-table build, notebook builders
results/               result tables (CSV)
tests/                 pytest - incl. checks that the package reproduces the notebook results exactly
data/, artifacts/      datasets and trained models (local only, git-ignored)
```

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[train,prep,notebooks,serve,dev]"

python scripts/download_data.py          # MovieLens-1M into data/
```

Then build the Wikipedia-enriched movie file (`movies_wiki.csv`): run `notebooks/movielens/phase-1.ipynb`, which
matches movies to the Kaggle dataset `jrobischon/wikipedia-movie-plots`.

TensorFlow Recommenders needs its own environment (TFRS requires Keras 2):

```bash
python3.12 -m venv .venv-tf
.venv-tf/bin/pip install "tensorflow==2.21.*" "tf-keras==2.21.*" absl-py jinja2 ipykernel pandas
.venv-tf/bin/pip install --no-deps tensorflow-recommenders==0.7.7   # its metadata still asks for tensorflow-macos
.venv-tf/bin/pip install --no-deps -e .
.venv-tf/bin/python -m ipykernel install --user --name genrec-tf --display-name "Python (.venv-tf)"
```

## Reproducing

| step | notebook | where / time |
|---|---|---|
| train the Qwen recommender | `notebooks/movielens/train_qwen_colab.ipynb` | Colab GPU, ~2-4 h on an L4 |
| evaluate it (top-10 lists + metrics) | `notebooks/movielens/eval_qwen_colab.ipynb` | Colab GPU, ~25 min |
| SASRec (also saves the models) | `notebooks/movielens/sasrec.ipynb` | laptop, ~15 min |
| TensorFlow Recommenders | `notebooks/movielens/tfrs.ipynb` (kernel *Python (.venv-tf)*) | laptop CPU, ~15 min |
| classic baselines + the combined table | `notebooks/movielens/baselines.ipynb` | laptop, ~1 min |

The Colab notebooks install this package from GitHub (`GENREC_REPO` in their first cell, installed with
`#subdirectory=genrec`) and read the data from Google Drive. Notebooks are generated from
`scripts/notebook_builders/*_cells.py` (`python scripts/notebook_builders/to_nb.py <cells.py> <notebook.ipynb> [--run]`).

```bash
pytest                    # tests marked `data` need data/ml-1m
```

## Lookup API

Pre-computed recommendations for the 1,000 test users from five models - random, most popular, TFRS sequential,
SASRec-200 and Qwen v2 - each with a **probability**: the model's P(next movie = X) over the movies the user hasn't
rated (uniform for random, rating share for popularity, softmax for SASRec/TFRS, the probability of answering exactly
that title for Qwen). They are not calibrated chances and are spread over thousands of movies, so they are small.
Qwen's lists in the API were regenerated rather than taken from the evaluation run; GPU beam search isn't bit-for-bit
repeatable, so they score HR@10 0.232 instead of the reported 0.236 (hit@1 is the same, 0.067).

Build the tables, then serve them:

```bash
# Qwen's lists + probabilities: run notebooks/movielens/score_qwen_colab.ipynb on Colab and download qwen.csv
#   into artifacts/lookup_inputs/; TFRS's come from notebooks/movielens/tfrs.ipynb, SASRec's from its saved model
python scripts/build_lookup_table.py           # -> artifacts/lookup/ (and prints each model's metrics)
uvicorn serving.app:app --reload               # interactive docs: http://localhost:8000/docs
```

| endpoint | returns |
|---|---|
| `GET /health` | status, loaded models |
| `GET /models` | the models with their test metrics |
| `GET /users?offset=0&limit=100` | the test users' IDs |
| `GET /users/{user_id}?history=20` | recent history + the movie they actually rated next |
| `GET /users/{user_id}/recommendations?model=qwen&k=10` | ranked movies with `probability` and `is_next_movie` (default model: qwen if loaded, else the best by HR@10) |
| `GET /users/{user_id}/compare` | every model's list for the user (side by side) |
| `GET /movies/{movie_id}` | title, genres, year |

```bash
curl "localhost:8000/users/3156/recommendations?model=sasrec&k=3"
```

**Docker / Hugging Face Spaces** - the image has no model or GPU dependency (~200 MB), so the free CPU tier is plenty:

```bash
docker build -f serving/Dockerfile -t genrec-api .
docker run -p 7860:7860 -v $PWD/artifacts/lookup:/app/artifacts/lookup genrec-api
```

The tables contain MovieLens-derived data (users' rating histories), which MovieLens doesn't allow redistributing:
they are git-ignored and not baked into the image. On a Space, upload them to a **private** Hugging Face dataset repo
and set `GENREC_HF_REPO` and `HF_TOKEN`.

## Data and licenses

- **MovieLens-1M** (GroupLens): research use; it may not be redistributed, so it is downloaded, not included.
- **Wikipedia movie plots** (Kaggle `jrobischon/wikipedia-movie-plots`): text from Wikipedia (CC BY-SA).
- **Qwen2.5-0.5B** (Alibaba Qwen): Apache-2.0.
- Code: MIT.

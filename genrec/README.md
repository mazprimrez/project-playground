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

## Recent movies (MovieLens 32M, cut to 1M ratings)

ML-1M stops in 2000. `notebooks/ml32m-recent/` builds an ML-1M-sized dataset of **recent movies** from
[MovieLens 32M](https://grouplens.org/datasets/movielens/32m/): the 4,000 most-rated movies released 2010-2023,
10,628 users with at least 20 ratings on them, 999,878 ratings (rated 2009-2023; median 53 ratings per user vs 96 in
ML-1M). Plots, directors, cast and posters come from TMDB (MovieLens links every movie to its TMDB id). Same split,
the same kind of 1,000 test users, full ranking over the 4,000 movies.

| model | hit@1 | HR@10 | NDCG@10 | on ML-1M (HR@10) |
|---|---|---|---|---|
| random | 0.000 | 0.001 | 0.000 | 0.005 |
| most popular | 0.018 | 0.088 | 0.046 | 0.024 |
| TFRS two-tower (user ID) | 0.018 | 0.097 | 0.051 | 0.057 |
| UserKNN (cosine, k=200) | 0.028 | 0.112 | 0.064 | 0.081 |
| ItemKNN (cosine, k=50) | 0.021 | 0.118 | 0.063 | 0.085 |
| ItemKNN (cosine, k=50), last 20 ratings | 0.022 | 0.125 | 0.065 | 0.141 |
| TFRS sequential (GRU, last 20) | 0.021 | 0.107 | 0.056 | 0.167 |
| SASRec-20 (last 20 ratings) | 0.040 | 0.174 | 0.097 | 0.263 |
| **SASRec-200** | **0.042** | **0.182** | **0.100** | 0.294 |
| Qwen2.5-0.5B, 2 epochs (10 targets/user) | 0.019 | 0.139 | 0.069 | |
| **Qwen2.5-0.5B, 4 epochs** | **0.023** | **0.152** | **0.076** | 0.236 |

Qwen runs on Colab (`qwen_colab.ipynb`: train, evaluate and write `qwen.csv` for the API in one go). It beats TFRS
and popularity but trails SASRec; 4 epochs reach 84% of SASRec-200's HR@10 (ML-1M's best recipe: 80%). Unlike
ML-1M, where validation loss rose after epoch 2, more epochs still help here. The demo serves the 4-epoch lists.
Early observations: popularity is a much stronger baseline here (recent hits get rated by everyone), and the
sequential models drop more than the order-blind two-tower - with half as many ratings per user, there is less
sequence to learn from.

| notebook | what | where / time |
|---|---|---|
| `01_prep.ipynb` | ML-32M → the recent 1M-rating dataset (`data/ml-32m-recent/`) | laptop, ~2 min |
| `02_tmdb_metadata.ipynb` | TMDB plot / director / cast / poster per movie (needs `TMDB_API_KEY`) | laptop, ~8 min |
| `sasrec.ipynb` | SASRec-200 and SASRec-20 + random / popular | laptop, ~25 min |
| `tfrs.ipynb` | TFRS two-tower and sequential (kernel *Python (.venv-tf)*) | laptop CPU, ~20 min |
| `cf.ipynb` | collaborative filtering with cosine similarity: ItemKNN and UserKNN, k tuned on validation | laptop, ~1 min |
| `qwen_colab.ipynb` | Qwen: train, evaluate, export `qwen.csv` | Colab GPU, ~2-4 h on an L4 |

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
  movielens/           data overview (interaction-analysis), SASRec, TFRS
  ml32m-recent/        recent movies (MovieLens 32M cut to 1M): data prep, TMDB details, SASRec, TFRS, Qwen (Colab)
serving/               the lookup API (FastAPI) + Dockerfile
results/               result tables (CSV)
tests/                 pytest - incl. checks that the package reproduces the notebook results exactly
data/, artifacts/      datasets and trained models (local only, git-ignored)
```

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[train,prep,notebooks,serve,dev]"

# MovieLens-1M: download https://files.grouplens.org/datasets/movielens/ml-1m.zip and unzip it into data/ml-1m/
```

`movies_wiki.csv` (MovieLens movies matched to Wikipedia plots from the Kaggle dataset
`jrobischon/wikipedia-movie-plots`) was built by a notebook that is now only in the git history (`phase-1.ipynb`).

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
| SASRec (also saves the models) | `notebooks/movielens/sasrec.ipynb` | laptop, ~15 min |
| TensorFlow Recommenders | `notebooks/movielens/tfrs.ipynb` (kernel *Python (.venv-tf)*) | laptop CPU, ~15 min |

The Qwen training / evaluation / scoring notebooks (Colab) and the classic baselines notebook are in the git
history; their results are in the table above and in `results/`.

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

The API only reads the four CSV tables in `artifacts/lookup/` (movies, users, recommendations, models) - no model,
no dataset. They were built once from the models' outputs and are not in the repository (MovieLens-derived data).

```bash
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
| `GET /movies?q=inter&limit=10` | catalogue search by title (for the live demo) |
| `GET /movies/{movie_id}` | title, genres, year, `poster_url` (TMDB, when `movies.csv` has a `poster_path`) |

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

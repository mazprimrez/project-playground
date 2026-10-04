# %% [markdown]
# # SASRec baseline on MovieLens-1M
#
# **SASRec** (Kang & McAuley, 2018, "Self-Attentive Sequential Recommendation") is the standard sequential
# recommender: a small transformer that reads a user's rating history (movie IDs only - no titles, plots or genres)
# and predicts the next movie (`genrec.models.sasrec`). It is the usual yardstick for models like the fine-tuned Qwen.
#
# Same split and test as every other model: train on `items[:-2]`, validate on `items[-2]`, test on `items[-1]`; the
# same 1,000 test users; full ranking over the catalogue, already-rated movies skipped; hit@1, HR@10, NDCG@10.
#
# Two variants: **SASRec-200** (the standard setup, up to 200 past ratings) and **SASRec-20** (only the last 20, the
# same information the Qwen prompt gets). Trains in ~15 minutes on a laptop; the models are saved to `artifacts/`
# for the serving endpoint.
#
# Note on numbers in papers: the SASRec paper reports Hit@10 = 0.82 on ML-1M because it ranks the true movie among
# only 100 sampled negatives; here every movie in the catalogue is a candidate (full ranking), which is much harder
# and the more reliable protocol (Krichene & Rendle, 2020).

# %%
import pandas as pd
import torch

from genrec.data import Dataset, history, sample_test_users, target
from genrec.models import sasrec
from genrec.paths import ARTIFACTS_DIR, ML1M_DIR, RESULTS_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
VARIANTS = {"SASRec-200": sasrec.SASRecConfig(max_len=200), "SASRec-20": sasrec.SASRecConfig(max_len=20)}
print("device:", DEVICE)

# %% [markdown]
# ## 1. Data

# %%
ds = Dataset.load(ML1M_DIR)
seqs = ds.encoded()
test_users = sample_test_users(ds.users)
title = dict(zip(range(1, ds.index.n_cols), ds.movies.set_index("MovieID").loc[ds.index.movie_ids, "Title"]))
lengths = pd.Series([len(s) for s in seqs.values()])
print(f"{len(seqs)} users, {ds.index.n_cols - 1} movies, {len(ds.ratings)} ratings; "
      f"ratings per user: median {lengths.median():.0f}, max {lengths.max()}")

# %% [markdown]
# ## 2. Train both variants
#
# Every epoch: a random window of each user's training history, every position predicted (full softmax). The epoch
# with the best validation NDCG@10 is kept.

# %%
models = {}
for name, cfg in VARIANTS.items():
    print(f"--- {name}")
    models[name], _ = sasrec.train(seqs, ds.index.n_cols - 1, cfg, DEVICE, log=lambda m, n=name: print(f"{n} {m}"))
    sasrec.save(models[name], ds.index.movie_ids, ARTIFACTS_DIR / name.lower())
print("saved to", ARTIFACTS_DIR)

# %% [markdown]
# ## 3. Test-set comparison

# %%
def test_metrics(model, us):
    return sasrec.evaluate(model, [history(seqs[u], "test") for u in us], [target(seqs[u], "test") for u in us], DEVICE)


results = {name: test_metrics(m, test_users) for name, m in models.items()}
table = pd.DataFrame(results).T[["hit@1", "HR@10", "NDCG@10"]]
RESULTS_DIR.mkdir(exist_ok=True)
table.to_csv(RESULTS_DIR / "sasrec_test.csv")   # picked up by baselines.ipynb
print(f"test: last rating of {len(test_users)} users")
table.style.format("{:.3f}")

# %% [markdown]
# ### All 6,040 users

# %%
pd.DataFrame({name: test_metrics(m, ds.users) for name, m in models.items()}).T.style.format("{:.3f}")

# %% [markdown]
# ## 4. Example recommendations (SASRec-200)

# %%
for u in test_users[:5]:
    hist, tgt = history(seqs[u], "test"), target(seqs[u], "test")
    top = sasrec.recommend(models["SASRec-200"], hist, DEVICE)
    verdict = f"HIT at rank {top.index(tgt) + 1}" if tgt in top else "miss"
    print(f"user {u}  last rated: {'; '.join(title[m] for m in hist[-5:])}")
    print(f"   real next: {title[tgt]}   {verdict}")
    print(f"   top 10:    {'; '.join(title[m] for m in top)}\n")

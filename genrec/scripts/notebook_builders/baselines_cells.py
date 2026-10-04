# %% [markdown]
# # Classic baselines on MovieLens-1M: random, popularity, collaborative filtering
#
# Non-neural reference points for the fine-tuned Qwen recommender, SASRec and TensorFlow Recommenders:
#
# | model | idea |
# |---|---|
# | **Random** | 10 random movies the user hasn't rated - the floor |
# | **Most popular** | the 10 most-rated movies the user hasn't rated |
# | **ItemKNN** | item-based collaborative filtering: two movies are similar when the same users rated them (cosine similarity of their user columns); a movie's score is its total similarity to the user's history |
# | **UserKNN** | user-based collaborative filtering: find the users most similar to this one (cosine similarity of their rating rows) and recommend what they rated |
#
# Collaborative filtering ignores the *order* of a user's ratings, so it answers "what does this user like?" rather
# than "what comes next?". **ItemKNN (last 20)** scores from the 20 most recent ratings only - the same information
# the Qwen prompt gets.
#
# Same split and test for every model (`genrec.data`): train on `items[:-2]`, validate on `items[-2]`, test on
# `items[-1]`; the same 1,000 test users; full ranking over the catalogue, already-rated movies skipped; hit@1,
# HR@10, NDCG@10 (`genrec.metrics`). The neighbourhood size `k` is tuned on the validation item. The final table
# also collects the other models' results from `results/`.

# %%
import time

import pandas as pd

from genrec.data import Dataset, history, sample_test_users, target
from genrec.metrics import evaluate_scores, top_k
from genrec.models import baselines as B
from genrec.paths import ML1M_DIR, RESULTS_DIR

K_GRID = [10, 20, 50, 100, 200, 400, None]   # neighbourhood sizes to try (None = all neighbours)
RECENT = 20                                  # ItemKNN (last 20): history window, same as the Qwen prompt

# Qwen results from eval_qwen_colab.ipynb (the same 1,000 users)
QWEN = {
    "Qwen2.5-0.5B v1 (5 targets/user)": {"hit@1": 0.041, "HR@10": 0.160, "NDCG@10": 0.093},
    "Qwen2.5-0.5B v2 (20 targets/user)": {"hit@1": 0.067, "HR@10": 0.236, "NDCG@10": 0.138},
}

# %% [markdown]
# ## 1. Data

# %%
ds = Dataset.load(ML1M_DIR)
seqs = ds.encoded()                  # user -> item indices (1..n) in rating order
users = ds.users
test_users = sample_test_users(users)
row_of = {u: r for r, u in enumerate(users)}
title = dict(zip(range(1, ds.index.n_cols), ds.movies.set_index("MovieID").loc[ds.index.movie_ids, "Title"]))

# users x movies 0/1 matrices of everything rated before the validation / test target
X = {split: B.interaction_matrix([history(seqs[u], split) for u in users], ds.index.n_cols) for split in ["val", "test"]}
print(f"{len(users)} users, {ds.index.n_cols - 1} movies; density {X['test'].mean():.1%}")


def split_lists(us, split):
    return [history(seqs[u], split) for u in us], [target(seqs[u], split) for u in us]

# %% [markdown]
# ## 2. Tune the neighbourhood size on the validation item (all users)

# %%
val_hist, val_tgt = split_lists(users, "val")
tuning, best_k = [], {}
for name in ["ItemKNN", f"ItemKNN (last {RECENT})", "UserKNN"]:
    t0 = time.time()
    for k in K_GRID:
        if name == "UserKNN":
            scores = B.userknn_scores(X["val"], list(range(len(users))), k)
        else:
            scores = B.itemknn_scores(val_hist, B.item_similarity(X["val"], k), recent=RECENT if "last" in name else None)
        tuning.append({"model": name, "k": "all" if k is None else k, **evaluate_scores(scores, val_hist, val_tgt)})
    best = max((r for r in tuning if r["model"] == name), key=lambda r: r["NDCG@10"])
    best_k[name] = None if best["k"] == "all" else best["k"]
    print(f"{name}: best k = {best['k']} (validation NDCG@10 {best['NDCG@10']:.3f}), {time.time() - t0:.0f}s")

pd.DataFrame(tuning).pivot(index="k", columns="model", values="NDCG@10").style.format("{:.3f}") \
    .set_caption("validation NDCG@10 by neighbourhood size")

# %% [markdown]
# ## 3. Test-set comparison
#
# The same 1,000 users for every model. Most popular should be HR@10 0.024 (as in the Qwen evaluation); SASRec and
# TFRS rows appear once `sasrec.ipynb` / `tfrs.ipynb` have been run.

# %%
def all_scores(us):
    hist, _ = split_lists(us, "test")
    return {
        "random": B.random_scores(len(us), ds.index.n_cols),
        "most popular": B.popularity_scores(X["test"], len(us)),
        "ItemKNN": B.itemknn_scores(hist, B.item_similarity(X["test"], best_k["ItemKNN"])),
        f"ItemKNN (last {RECENT})": B.itemknn_scores(hist, B.item_similarity(X["test"], best_k[f"ItemKNN (last {RECENT})"]),
                                                     recent=RECENT),
        "UserKNN": B.userknn_scores(X["test"], [row_of[u] for u in us], best_k["UserKNN"]),
    }


test_hist, test_tgt = split_lists(test_users, "test")
test_scores = all_scores(test_users)
results = {name: evaluate_scores(s, test_hist, test_tgt) for name, s in test_scores.items()}
RESULTS_DIR.mkdir(exist_ok=True)
pd.DataFrame(results).T.to_csv(RESULTS_DIR / "baselines_test.csv")

table = pd.DataFrame({**results, **QWEN}).T
for csv, notebook in [("sasrec_test.csv", "sasrec.ipynb"), ("tfrs_test.csv", "tfrs.ipynb")]:
    if (RESULTS_DIR / csv).exists():
        table = pd.concat([table, pd.read_csv(RESULTS_DIR / csv, index_col=0)])
    else:
        print(f"no results from {notebook} yet")
print(f"test: last rating of {len(test_users)} users")
table.sort_values("NDCG@10")[["hit@1", "HR@10", "NDCG@10"]].style.format("{:.3f}")

# %% [markdown]
# ### All 6,040 users (the Qwen evaluation used the 1,000-user sample)

# %%
all_hist, all_tgt = split_lists(users, "test")
full = {name: evaluate_scores(s, all_hist, all_tgt) for name, s in all_scores(users).items()}
pd.DataFrame(full).T.sort_values("NDCG@10").style.format("{:.3f}")

# %% [markdown]
# ## 4. Example recommendations

# %%
for r, u in enumerate(test_users[:3]):
    hist, tgt = test_hist[r], test_tgt[r]
    print(f"user {u}  last rated: {'; '.join(title[m] for m in hist[-5:])}")
    print(f"   real next: {title[tgt]}")
    for name in ["ItemKNN", "UserKNN"]:
        top = top_k(test_scores[name][r:r + 1], [hist])[0].tolist()
        verdict = f"HIT at rank {top.index(tgt) + 1}" if tgt in top else "miss"
        print(f"   {name:8s} ({verdict}): {'; '.join(title[m] for m in top[:5])} ...")
    print()

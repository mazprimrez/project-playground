# %% [markdown]
# # TensorFlow Recommenders on MovieLens-1M
#
# Two standard [TensorFlow Recommenders (TFRS)](https://www.tensorflow.org/recommenders) retrieval models
# (`genrec.models.tfrs_models`), evaluated exactly like every other model:
#
# | model | query tower ("who is asking") | order of ratings |
# |---|---|---|
# | **TFRS two-tower** | a learned vector per user (user-ID embedding) | ignored - like collaborative filtering, but learned |
# | **TFRS sequential (GRU, last 20)** | a GRU reading the user's last 20 movies | used - comparable to SASRec-20 and the Qwen prompt |
#
# Both are trained with TFRS's `Retrieval` task (in-batch softmax). Same split, the same 1,000 test users, full
# ranking, already-rated movies skipped. The number of epochs is picked on the validation item.
#
# **Runs in the separate TensorFlow environment** (kernel *Python (.venv-tf)*, see the README). CPU only, ~15 min.
# TensorFlow training on CPU is not bit-for-bit repeatable: expect differences of ~0.01 HR@10 between runs.

# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time

import pandas as pd
import tensorflow as tf
import tf_keras as keras

from genrec.data import Dataset, history, sample_test_users, target
from genrec.metrics import evaluate_scores, top_k
from genrec.models import tfrs_models as T
from genrec.paths import ML1M_DIR, RESULTS_DIR

MAX_EPOCHS = 30
PATIENCE = 3       # stop when validation NDCG@10 hasn't improved for this many epochs
CONTEXT = 20
SEED = 42
print(f"tensorflow {tf.__version__}, tf_keras {keras.__version__}")

# %% [markdown]
# ## 1. Data

# %%
ds = Dataset.load(ML1M_DIR)
seqs = ds.encoded()
users = ds.users
test_users = sample_test_users(users)
row_of = {u: r for r, u in enumerate(users)}
title = dict(zip(range(1, ds.index.n_cols), ds.movies.set_index("MovieID").loc[ds.index.movie_ids, "Title"]))


def split_lists(us, split):
    return [history(seqs[u], split) for u in us], [target(seqs[u], split) for u in us]


val_hist, val_tgt = split_lists(users, "val")


def fit_best(model, train_ds, score_fn, name):
    """Train epoch by epoch, keep the weights with the best validation NDCG@10 (all users)."""
    best, best_epoch, best_weights, t0 = -1.0, 0, None, time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        hist = model.fit(train_ds, epochs=1, verbose=0)
        val = evaluate_scores(score_fn(users, val_hist), val_hist, val_tgt)
        if val["NDCG@10"] > best:
            best, best_epoch, best_weights = val["NDCG@10"], epoch, model.get_weights()
        print(f"{name} epoch {epoch:2d}  loss {hist.history['loss'][0]:9.1f}  val HR@10 {val['HR@10']:.3f}  "
              f"NDCG@10 {val['NDCG@10']:.3f}  (best {best:.3f} @ {best_epoch})  {time.time() - t0:.0f}s")
        if epoch - best_epoch >= PATIENCE:
            break
    model.set_weights(best_weights)
    return best_epoch

# %% [markdown]
# ## 2. Two-tower (user ID -> movie)
#
# One embedding per user, so it only knows what the user rated during training: validation trains on `items[:-2]`,
# the test model is retrained on `items[:-1]` for the number of epochs that was best on validation.

# %%
def two_tower():
    tf.random.set_seed(SEED)
    model = T.TwoTower(len(users), ds.index.n_cols)
    model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=0.1))
    return model


twotower = two_tower()
best_epochs = fit_best(twotower, T.pairs_dataset(range(len(users)), val_hist),
                       lambda us, _: twotower.scores([row_of[u] for u in us]), "two-tower")
twotower = two_tower()
twotower.fit(T.pairs_dataset(range(len(users)), split_lists(users, "test")[0]), epochs=best_epochs, verbose=0)
print(f"two-tower retrained on items[:-1] for {best_epochs} epochs")

# %% [markdown]
# ## 3. Sequential: GRU over the last 20 movies -> next movie
#
# Reads any history at prediction time, so no retraining for the test (like SASRec). Trained on every position of
# `items[:-2]`.

# %%
tf.random.set_seed(SEED)
sequential = T.SequentialRetrieval(ds.index.n_cols, context=CONTEXT)
sequential.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))
train_ds, n_examples = T.sequence_dataset(val_hist, CONTEXT)
print(f"{n_examples} training examples")
fit_best(sequential, train_ds, lambda _, hist: sequential.scores(hist), "sequential")

# %% [markdown]
# ## 4. Test-set comparison

# %%
def test_scores(us):
    hist, _ = split_lists(us, "test")
    return {"TFRS two-tower": twotower.scores([row_of[u] for u in us]),
            f"TFRS sequential (GRU, last {CONTEXT})": sequential.scores(hist)}


test_hist, test_tgt = split_lists(test_users, "test")
scores = test_scores(test_users)
table = pd.DataFrame({name: evaluate_scores(s, test_hist, test_tgt) for name, s in scores.items()}).T
RESULTS_DIR.mkdir(exist_ok=True)
table.to_csv(RESULTS_DIR / "tfrs_test.csv")   # picked up by baselines.ipynb
print(f"test: last rating of {len(test_users)} users")
table.style.format("{:.3f}")

# %%
# export the sequential model's top-10 + probabilities for the lookup API (scripts/build_lookup_table.py)
from genrec import lookup
from genrec.paths import ARTIFACTS_DIR

seq_name = f"TFRS sequential (GRU, last {CONTEXT})"
export = lookup.to_rows("tfrs-sequential", test_users, lookup.softmax_top_k(scores[seq_name], test_hist),
                        decode=lambda i: ds.index.movie_ids[i - 1])
(ARTIFACTS_DIR / "lookup_inputs").mkdir(parents=True, exist_ok=True)
export.to_csv(ARTIFACTS_DIR / "lookup_inputs" / "tfrs_sequential.csv", index=False)
print(f"exported {len(export)} recommendations -> {ARTIFACTS_DIR / 'lookup_inputs' / 'tfrs_sequential.csv'}")
export.head()

# %% [markdown]
# ### All 6,040 users

# %%
all_hist, all_tgt = split_lists(users, "test")
pd.DataFrame({name: evaluate_scores(s, all_hist, all_tgt) for name, s in test_scores(users).items()}).T \
    .style.format("{:.3f}")

# %% [markdown]
# ## 5. Example recommendations

# %%
for r, u in enumerate(test_users[:3]):
    hist, tgt = test_hist[r], test_tgt[r]
    print(f"user {u}  last rated: {'; '.join(title[m] for m in hist[-5:])}")
    print(f"   real next: {title[tgt]}")
    for name, s in scores.items():
        top = top_k(s[r:r + 1], [hist])[0].tolist()
        verdict = f"HIT at rank {top.index(tgt) + 1}" if tgt in top else "miss"
        print(f"   {name[:15]:15s} ({verdict}): {'; '.join(title[m] for m in top[:5])} ...")
    print()

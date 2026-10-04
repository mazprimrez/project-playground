"""The refactored package must reproduce what the notebooks produced."""
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from genrec.data import history, sample_test_users, target
from genrec.metrics import evaluate_scores
from genrec.models import baselines as B
from genrec.paths import ML1M_DIR, RESULTS_DIR, ROOT
from genrec.prompts import build_samples

pytestmark = pytest.mark.data


def notebook_build_samples():
    """build_samples as written in notebooks/movielens/train_qwen.ipynb (the code the models were trained with)."""
    nb = json.loads((ROOT / "notebooks/movielens/train_qwen.ipynb").read_text())
    cells = {c.get("id"): "".join(c["source"]) for c in nb["cells"]}
    ns = {"pd": pd, "random": random, "re": __import__("re")}
    exec(cells["cc255aec"], ns)   # constants, display_title, clean_plot, ..., load_movies
    exec(cells["45f9e8bd"], ns)   # metadata_samples, next_movie_prompt, next_movie_samples, build_samples
    return ns["build_samples"]


@pytest.mark.parametrize("next_per_user,plot_words", [(5, 150), (20, 40)])
def test_samples_identical_to_notebook(next_per_user, plot_words):
    cfg = SimpleNamespace(data_dir=ML1M_DIR, tasks=["plot", "genres", "details", "identify", "next_movie"],
                          plot_words=plot_words, snippet_words=80, max_cast=5, history_len=20, min_history=5,
                          next_per_user=next_per_user, next_eval_users=500, eval_frac=0.02, seed=42)
    assert build_samples(cfg) == notebook_build_samples()(cfg)


def test_baselines_match_saved_results(ml1m):
    """Popularity / ItemKNN / UserKNN on the 1,000 test users, k as tuned in notebooks/movielens/baselines.ipynb."""
    saved = pd.read_csv(RESULTS_DIR / "baselines_test.csv", index_col=0)
    seqs = ml1m.encoded()
    users = sample_test_users(ml1m.users)
    hist = [history(seqs[u], "test") for u in users]
    tgt = [target(seqs[u], "test") for u in users]
    X = B.interaction_matrix([history(seqs[u], "test") for u in ml1m.users], ml1m.index.n_cols)
    row = {u: r for r, u in enumerate(ml1m.users)}
    got = {
        "random": evaluate_scores(B.random_scores(len(users), ml1m.index.n_cols), hist, tgt),
        "most popular": evaluate_scores(B.popularity_scores(X, len(users)), hist, tgt),
        "ItemKNN": evaluate_scores(B.itemknn_scores(hist, B.item_similarity(X, 10)), hist, tgt),
        "ItemKNN (last 20)": evaluate_scores(B.itemknn_scores(hist, B.item_similarity(X, 20), recent=20), hist, tgt),
        "UserKNN": evaluate_scores(B.userknn_scores(X, [row[u] for u in users], 20), hist, tgt),
    }
    for name, metrics in got.items():
        for metric, value in metrics.items():
            assert value == pytest.approx(saved.loc[name, metric], abs=1e-9), (name, metric)

"""The package must reproduce the saved results."""
import pandas as pd
import pytest

from genrec.data import history, sample_test_users, target
from genrec.metrics import evaluate_scores
from genrec.models import baselines as B
from genrec.paths import RESULTS_DIR

pytestmark = pytest.mark.data



def test_baselines_match_saved_results(ml1m):
    """Popularity / ItemKNN / UserKNN on the 1,000 test users (k as tuned when the baselines were run)."""
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

"""The predictors and the backtest on the synthetic tour (no data or Prophet needed)."""
import numpy as np

from eras import backtest as B
from eras import models as M
from eras.data import OUTFIT_CATEGORIES, POINTS, SONG_POINTS


def test_choice_model_learns_the_better_feature():
    rng = np.random.default_rng(0)
    events = []
    for _ in range(200):
        X = rng.normal(size=(4, 2))
        events.append((X, int(X[:, 0].argmax())))         # the choice follows feature 0, not feature 1
    m = M.ChoiceModel(2).fit(events)
    assert m.w[0] > 1 and abs(m.w[1]) < abs(m.w[0]) / 3
    p = m.probabilities(np.array([[2.0, 0], [0, 0]]))
    assert np.isclose(p.sum(), 1) and p[0] > p[1]


def test_outfit_features_use_only_the_past(tour):
    cands, X = M.outfit_candidates(tour, "Fearless dress", 2)
    assert cands == ["V1", "V2"] and X.shape == (2, len(M.OUTFIT_FEATURES))
    assert M.outfit_candidates(tour, "Fearless dress", 0) is None
    assert M.outfit_baseline("most_worn", tour, "Reputation outfit", 5) == "V1"
    assert M.outfit_baseline("rotation", tour, "Fearless dress", 9) == "V1"   # V1 is the longest unworn


def test_song_candidates_skip_the_main_set(tour):
    songs, X = M.song_candidates(tour, "piano", 5)
    assert "Shake It Off" not in songs and X.shape == (20, len(M.SONG_FEATURES))


def test_backtest_rows_and_points(tour):
    r = B.run(tour, warmup=10, prophet=False)
    assert set(r.show) == set(range(10, tour.n))
    assert set(r.category) == set(OUTFIT_CATEGORIES) | {"Surprise song (guitar)", "Surprise song (piano)"}
    outfits = r[~r.category.str.startswith("Surprise song (")]
    assert set(outfits.points) <= {0, POINTS} and (outfits.points == POINTS * outfits.correct).all()
    songs = r[r.category.str.startswith("Surprise song (") & (r.method != "random")]
    assert set(songs.points) <= {0, SONG_POINTS}
    acc = B.accuracy_by_category(r)
    assert acc.loc["Reputation outfit", "choice"] == 1.0
    assert acc.loc["Fearless dress", "choice"] > 0.9        # a fixed rotation is learnable
    g = B.game_points(r, "choice", "choice")
    assert len(g) == tour.n - 10 and g.max() <= 13 * POINTS + 2 * SONG_POINTS


def test_prophet_lag_features_use_only_earlier_shows(tour):
    # Fearless dress rotates V1, V2, V3; cities change every 3 shows
    f = M.lag_features(tour, "Fearless dress", "V1", 6)
    assert list(f.columns) == M.LAG_FEATURES and len(f) == 7
    assert f.previous_show.tolist() == [0, 1, 0, 0, 1, 0, 0]          # V1 worn at shows 0, 3, 6
    assert f.earlier_in_city.tolist() == [0, 1, 1, 0, 1, 1, 0]        # cities: shows 0-2, 3-5, 6-8
    assert f.share_last10[6] == 2 / 6

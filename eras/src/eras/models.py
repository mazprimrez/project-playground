"""Predictors for one show, using only the shows before it.

Outfits (per game category, pick one version):
    most_worn     the version worn most so far
    recent        the version worn most in the last 10 shows
    rotation      the version not worn for the longest (among those worn at least twice)
    prophet       the original article: a 0/1 series per version, Prophet forecasts the show's date, highest wins
    choice        conditional logit: every candidate version gets features (overall / recent / this-leg share, shows
                  since last worn, worn earlier in this city, rotation rank, ...); one set of weights is learned
                  across all categories from the past shows; the highest-scoring candidate wins

Surprise songs (per instrument, pick one song; correct if it was played on that instrument):
    random        a song not yet played on the tour, uniformly (scored by its expected points)
    most_played   the song played most on this instrument so far
    choice        conditional logit over the catalogue with song features (played before, on which instrument, how
                  often, how long ago, how complete its album is, album played in the last shows, ...)
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

from .data import Tour

TTPD_RELEASE = pd.Timestamp("2024-04-19")


# ---------------------------------------------------------------- conditional logit

class ChoiceModel:
    """Multinomial choice over a variable candidate set: P(candidate) = softmax(features @ w). Fit by full-batch Adam
    on the mean negative log-likelihood + L2 (small data: a few thousand choices, ~10-20 features)."""

    def __init__(self, n_features: int, l2=1e-2, steps: int = 300, lr: float = 0.05):
        self.w = np.zeros(n_features)
        self.l2 = np.broadcast_to(np.asarray(l2, dtype=float), (n_features,))   # one value, or one per feature
        self.steps, self.lr = steps, lr

    def fit(self, events: list) -> "ChoiceModel":
        """events: [(X candidates x features, index of the chosen candidate), ...]; warm-starts from the last fit."""
        if not events:
            return self
        X = np.concatenate([x for x, _ in events])
        sizes = np.array([len(x) for x, _ in events])
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        group = np.repeat(np.arange(len(events)), sizes)
        chosen = starts + np.array([y for _, y in events])
        m, v = np.zeros_like(self.w), np.zeros_like(self.w)
        for step in range(1, self.steps + 1):
            z = X @ self.w
            z = z - np.repeat(np.maximum.reduceat(z, starts), sizes)              # stable softmax per choice
            e = np.exp(z)
            p = e / np.bincount(group, e)[group]
            grad = (X * p[:, None]).sum(0) - X[chosen].sum(0)                 # d(-loglik)
            grad = grad / len(events) + 2 * self.l2 * self.w
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad ** 2
            self.w -= self.lr * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
        return self

    def probabilities(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.w
        e = np.exp(z - z.max())
        return e / e.sum()


# ---------------------------------------------------------------- outfits

OUTFIT_FEATURES = ["share", "share_last5", "share_last10", "share_this_leg", "log_shows_since", "previous_show",
                   "earlier_in_city", "rotation_rank", "log_age", "worn_once", "not_yet_this_leg"]


def outfit_candidates(tour: Tour, category: str, t: int):
    """(candidate versions, feature matrix) for show t from shows < t; None if nothing was worn yet."""
    hist = tour.outfits[category][:t]
    seen = [w for w in hist if w is not None]
    if not seen:
        return None
    cands = sorted(set(seen))
    cnt, last, first = Counter(seen), {}, {}
    for k, w in enumerate(hist):
        if w is not None:
            last[w] = k
            first.setdefault(w, k)
    shows = tour.shows
    leg = shows.leg[t]
    in_leg = [hist[k] for k in range(t) if shows.leg[k] == leg and hist[k] is not None]
    run, k = set(), t - 1                       # earlier nights of the same city, back to back
    while k >= 0 and shows.city[k] == shows.city[t]:
        run.add(hist[k])
        k -= 1
    order = sorted(cands, key=lambda w: last[w])              # 0 = not worn for the longest
    last5, last10, leg_cnt = Counter(seen[-5:]), Counter(seen[-10:]), Counter(in_leg)
    X = np.array([[
        cnt[w] / len(seen),
        last5[w] / min(5, len(seen)),
        last10[w] / min(10, len(seen)),
        leg_cnt[w] / len(in_leg) if in_leg else 0.0,
        math.log1p(t - last[w]),
        float(t - last[w] == 1),
        float(w in run),
        order.index(w) / max(1, len(cands) - 1),
        math.log1p(t - first[w]),
        float(cnt[w] == 1),
        float(bool(in_leg) and leg_cnt[w] == 0),
    ] for w in cands])
    return cands, X


def outfit_baseline(method: str, tour: Tour, category: str, t: int):
    hist = tour.outfits[category][:t]
    seen = [w for w in hist if w is not None]
    if not seen:
        return None
    if method == "most_worn":
        return Counter(seen).most_common(1)[0][0]
    if method == "recent":
        return Counter(seen[-10:]).most_common(1)[0][0]
    if method == "rotation":
        cnt, last = Counter(seen), {w: k for k, w in enumerate(hist) if w is not None}
        live = [w for w in last if cnt[w] >= 2] or list(last)
        return min(live, key=lambda w: last[w])
    raise ValueError(method)


def prophet_pick(tour: Tour, category: str, t: int):
    """The article's method. Needs `pip install prophet`; slow (one fit per version), so run it in parallel."""
    import logging
    logging.getLogger("cmdstanpy").disabled = True
    from prophet import Prophet

    hist = tour.outfits[category][:t]
    cands = sorted({w for w in hist if w is not None})
    if not cands:
        return None
    dates = tour.shows.date
    best, best_y = None, -np.inf
    for v in cands:
        df = pd.DataFrame({"ds": dates[:t].values, "y": [float(w == v) for w in hist]})
        y = Prophet().fit(df).predict(pd.DataFrame({"ds": [dates[t]]})).yhat.iloc[0]
        if y > best_y:
            best, best_y = v, y
    return best


# ---------------------------------------------------------------- surprise songs

SONG_FEATURES = ["played_before", "played_same_instrument", "played_other_instrument", "log_times_played",
                 "log_shows_since_played", "repeat_2024", "album_share_played", "album_in_last_3_shows",
                 "album_in_last_show", "never_played_album_left", "other_album", "new_album",
                 "same_instrument_2024", "log_times_same_instrument_2024"]


def song_candidates(tour: Tour, instrument: str, t: int):
    """(candidate songs, feature matrix) for one instrument at show t, from shows < t."""
    cat = tour.catalogue
    date = tour.shows.date[t]
    pool = cat[~cat.main_set]
    pool = pool[~((pool.album == "TTPD") & (date < TTPD_RELEASE))]           # not released yet
    same = tour.guitar if instrument == "guitar" else tour.piano
    other = tour.piano if instrument == "guitar" else tour.guitar
    played_same = Counter(s for k in range(t) for s in same[k])
    played_other = Counter(s for k in range(t) for s in other[k])
    last_played = {}
    for k in range(t):
        for s in same[k] | other[k]:
            last_played[s] = k
    album_of = dict(zip(cat.song, cat.album))
    played_any = set(played_same) | set(played_other)
    album_total = pool.album.value_counts()
    album_done = Counter(album_of.get(s) for s in played_any if s in set(pool.song))
    recent_albums = [album_of.get(s) for k in range(max(0, t - 3), t) for s in same[k] | other[k]]
    last_albums = {album_of.get(s) for s in (same[t - 1] | other[t - 1])} if t else set()
    is_2024 = float(date.year >= 2024)
    rows, songs = [], []
    for s, album in zip(pool.song, pool.album):
        before = s in played_any
        share = album_done[album] / album_total[album]
        rows.append([
            float(before),
            float(played_same[s] > 0),
            float(played_other[s] > 0),
            math.log1p(played_same[s] + played_other[s]),
            math.log1p(t - last_played[s]) if before else 0.0,
            is_2024 * float(before),
            share,
            float(album in recent_albums),
            float(album in last_albums),
            float(not before) * (1 - share),
            float(album == "OTHER"),
            float(album == "TTPD"),
            is_2024 * float(played_same[s] > 0),
            is_2024 * math.log1p(played_same[s]),
        ])
        songs.append(s)
    return songs, np.array(rows)


def song_baseline(method: str, tour: Tour, instrument: str, t: int):
    """most_played: the pick; random: None (scored as an expectation in the backtest)."""
    same = tour.guitar if instrument == "guitar" else tour.piano
    played = Counter(s for k in range(t) for s in same[k])
    if method == "most_played":
        return played.most_common(1)[0][0] if played else None
    raise ValueError(method)

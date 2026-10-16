"""Walk-forward backtest, scored like the Swift Alert Mastermind game.

For every show from `warmup` on, each method predicts using only the shows before it; then the prediction is compared
with what actually happened. Outfit categories are worth 4 points, each surprise song guess (guitar, piano) 2 points.

Output: one row per (show, category, method) with the pick, its probability (choice model), the truth, whether it
was right and the points it scored.
"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import models as M
from .data import OUTFIT_CATEGORIES, POINTS, SONG_POINTS, TTPD_PARTS, Tour

INSTRUMENTS = ["guitar", "piano"]


SHARED_L2, CATEGORY_L2 = 1e-2, 0.3      # category weights only move away from 0 when that category's data insists


def outfit_rows(tour: Tour, warmup: int, progress=None) -> list:
    # the choice model predicts every game outfit category except TTPD, which it builds from the dress + two piece
    modelled = [c for c in OUTFIT_CATEGORIES if c != "TTPD outfit"] + list(TTPD_PARTS)
    nf = len(M.OUTFIT_FEATURES)

    def augmented(c, X):
        """[shared features | the same features in this category's own block | zeros for the other blocks]"""
        out = np.zeros((len(X), nf * (1 + len(modelled))))
        out[:, :nf] = X
        k = 1 + modelled.index(c)
        out[:, k * nf:(k + 1) * nf] = X
        return out

    cache = {}
    for c in modelled:
        for t in range(1, tour.n):
            cand = M.outfit_candidates(tour, c, t)
            cache[(c, t)] = None if cand is None else (cand[0], augmented(c, cand[1]))
    l2 = np.r_[np.full(nf, SHARED_L2), np.full(nf * len(modelled), CATEGORY_L2)]

    def event(c, t):
        cand, truth = cache[(c, t)], tour.outfits[c][t]
        if cand is None or truth is None or truth not in cand[0]:
            return None
        return cand[1], cand[0].index(truth)

    def choose(c, t):
        if cache[(c, t)] is None:
            return None, None
        cands, X = cache[(c, t)]
        p = model.probabilities(X)
        return cands[int(p.argmax())], float(p.max())

    model, events, rows = M.ChoiceModel(nf * (1 + len(modelled)), l2=l2), [], []
    for t in range(1, tour.n):
        if t >= warmup:
            model.fit(events)
            for c in OUTFIT_CATEGORIES:
                truth = tour.outfits[c][t]
                if truth is None:
                    continue
                picks = {m: (M.outfit_baseline(m, tour, c, t), None) for m in ["most_worn", "recent", "rotation"]}
                if c == "TTPD outfit":
                    (d, pd_), (s, ps) = choose(TTPD_PARTS[0], t), choose(TTPD_PARTS[1], t)
                    picks["choice"] = (f"{d} + {s}", pd_ * ps) if d and s else (None, None)
                else:
                    picks["choice"] = choose(c, t)
                for method, (pick, prob) in picks.items():
                    rows.append({"show": t, "category": c, "method": method, "pick": pick, "probability": prob,
                                 "truth": truth, "correct": pick == truth, "points": POINTS * (pick == truth)})
        events += [e for c in modelled if (e := event(c, t))]
        if progress and t % 25 == 0:
            progress(f"outfits: show {t}/{tour.n - 1}")
    return rows


def song_rows(tour: Tour, warmup: int, progress=None) -> list:
    sets = {"guitar": tour.guitar, "piano": tour.piano}
    model, events, rows = M.ChoiceModel(len(M.SONG_FEATURES)), [], []
    for t in range(1, tour.n):
        cand = {i: M.song_candidates(tour, i, t) for i in INSTRUMENTS}
        if t >= warmup:
            model.fit(events)
            for i in INSTRUMENTS:
                truth = sets[i][t]
                if not truth:
                    continue
                songs, X = cand[i]
                p = model.probabilities(X)
                pick = songs[int(p.argmax())]
                label = " / ".join(sorted(truth))
                played = {s for k in range(t) for s in tour.guitar[k] | tour.piano[k]}
                unplayed = [s for s in songs if s not in played]
                expected = SONG_POINTS * len(truth & set(unplayed)) / len(unplayed) if unplayed else 0.0
                most = M.song_baseline("most_played", tour, i, t)
                cat = f"Surprise song ({i})"
                for method, pk, prob, pts in [("choice", pick, float(p.max()), SONG_POINTS * (pick in truth)),
                                              ("most_played", most, None, SONG_POINTS * (most in truth)),
                                              ("random", None, None, expected)]:
                    rows.append({"show": t, "category": cat, "method": method, "pick": pk, "probability": prob,
                                 "truth": label, "correct": pk in truth if pk else None, "points": pts})
        for i in INSTRUMENTS:                                    # learn from this show for the next ones
            songs, X = cand[i]
            events += [(X, songs.index(s)) for s in sets[i][t] if s in songs]
        if progress and t % 25 == 0:
            progress(f"songs: show {t}/{tour.n - 1}")
    return rows


def _prophet_job(args):
    tour, c, t, method = args
    return (c, t, method, *M.prophet_pick(tour, c, t, night=method == "prophet_night"))


def prophet_rows(tour: Tour, warmup: int, processes: int = 8) -> list:
    """The article's method, and the same with the night number, on the same shows (needs prophet; ~5 minutes on
    8 cores)."""
    jobs = [(tour, c, t, m) for m in ["prophet", "prophet_night"] for c in OUTFIT_CATEGORIES
            for t in range(warmup, tour.n) if tour.outfits[c][t] is not None]
    with Pool(processes) as pool:
        out = pool.map(_prophet_job, jobs, chunksize=4)
    return [{"show": t, "category": c, "method": m, "pick": pick, "probability": prob,
             "truth": tour.outfits[c][t], "correct": pick == tour.outfits[c][t],
             "points": POINTS * (pick == tour.outfits[c][t])} for c, t, m, pick, prob in out]


def run(tour: Tour, warmup: int = 10, prophet: bool = True, progress=None) -> pd.DataFrame:
    rows = outfit_rows(tour, warmup, progress) + song_rows(tour, warmup, progress)
    if prophet:
        rows += prophet_rows(tour, warmup)
    return pd.DataFrame(rows)


def points_per_show(results: pd.DataFrame) -> pd.DataFrame:
    """Shows x methods: total game points. Methods that only cover outfits (prophet, baselines) get the songs of a
    stated fallback added elsewhere; here every method is summed over the categories it predicts."""
    return results.pivot_table(index="show", columns="method", values="points", aggfunc="sum")


def game_points(results: pd.DataFrame, outfit_method: str, song_method: str) -> pd.Series:
    """A full game entry per show: one method for the outfit categories + one for the two surprise songs."""
    outfits = results[(results.method == outfit_method) & ~results.category.str.startswith("Surprise song (")]
    songs = results[(results.method == song_method) & results.category.str.startswith("Surprise song (")]
    return outfits.groupby("show").points.sum().add(songs.groupby("show").points.sum(), fill_value=0)


def accuracy_by_category(results: pd.DataFrame) -> pd.DataFrame:
    hit = results.dropna(subset=["correct"])
    return hit.pivot_table(index="category", columns="method", values="correct", aggfunc="mean")

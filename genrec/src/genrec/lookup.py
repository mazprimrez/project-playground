"""Pre-computed recommendation tables for the lookup API (genrec/serving).

For each test user and model: the top-10 recommendations with a probability, plus the user's history and the movie
they actually rated next. Probabilities are each model's P(next movie = X), over the movies the user hasn't rated:

    random        uniform: 1 / (number of unrated movies)
    most popular  the movie's share of all ratings, among the unrated movies
    SASRec, TFRS  softmax of the model's scores over the unrated movies
    Qwen          P(the model answers exactly this title), from QwenRecommender.title_probabilities

They are not calibrated chances and they are spread over thousands of movies, so they are small (top picks ~0.01-0.2).

Files (all CSV): movies (movie_id, title, genres, year), users (user_id, history, next_movie_id),
recommendations (user_id, model, rank, movie_id, probability), models (model, description, hit@1, HR@10, NDCG@10).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import evaluate_lists, top_k

DESCRIPTIONS = {
    "random": "10 random movies the user hasn't rated (the floor)",
    "most popular": "the 10 most-rated movies the user hasn't rated",
    "tfrs-sequential": "TensorFlow Recommenders retrieval model: a GRU over the last 20 movies",
    "sasrec": "SASRec: a small transformer over the user's last 200 movie IDs (the standard sequential recommender)",
    "qwen": "Qwen2.5-0.5B fine-tuned on the catalogue and next-movie prediction, reads the last 20 titles",
    "cf-cosine": "Collaborative filtering (ItemKNN): movies similar to the user's last 20, by cosine similarity",
}


def masked(scores: np.ndarray, histories: list) -> np.ndarray:
    s = scores.astype(np.float64, copy=True)
    s[:, 0] = -np.inf
    for r, h in enumerate(histories):
        s[r, h] = -np.inf
    return s


def softmax_top_k(scores: np.ndarray, histories: list, k: int = 10) -> list:
    """Top-k (column, probability) per row; probability = softmax over the unrated columns."""
    s = masked(scores, histories)
    p = np.exp(s - s.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    top = top_k(scores, histories, k)
    return [[(int(c), float(p[r, c])) for c in row] for r, row in enumerate(top)]


def normalized_top_k(scores: np.ndarray, weights: np.ndarray, histories: list, k: int = 10) -> list:
    """Top-k by `scores`; probability = weight / sum of weights over the unrated columns (popularity, uniform)."""
    w = np.where(np.isfinite(masked(scores, histories)), weights, 0.0)
    w = w / w.sum(axis=1, keepdims=True)
    return [[(int(c), float(w[r, c])) for c in row] for r, row in enumerate(top_k(scores, histories, k))]


def to_rows(model: str, user_ids: list, lists: list, decode=None) -> pd.DataFrame:
    """One row per recommendation; lists[i] = [(item, probability), ...] for user_ids[i]."""
    rows = [{"user_id": int(u), "model": model, "rank": rank, "movie_id": int(decode(m) if decode else m),
             "probability": p}
            for u, recs in zip(user_ids, lists) for rank, (m, p) in enumerate(recs, 1)]
    return pd.DataFrame(rows)


def model_metrics(recommendations: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """hit@1 / HR@10 / NDCG@10 per model, recomputed from the table itself (a consistency check)."""
    nxt = users.set_index("user_id")["next_movie_id"]
    out = []
    for model, g in recommendations.sort_values(["user_id", "rank"]).groupby("model", sort=False):
        lists = g.groupby("user_id")["movie_id"].apply(list)
        lists = lists.reindex(users["user_id"]).apply(lambda x: x if isinstance(x, list) else [])
        out.append({"model": model, "description": DESCRIPTIONS.get(model, ""),
                    **evaluate_lists(lists.tolist(), nxt.reindex(lists.index).tolist()), "users": len(g.user_id.unique())})
    return pd.DataFrame(out)


def write(folder, movies: pd.DataFrame, users: pd.DataFrame, recommendations: pd.DataFrame) -> pd.DataFrame:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    users = users.assign(history=users["history"].map(json.dumps))
    movies.to_csv(folder / "movies.csv", index=False)
    users.to_csv(folder / "users.csv", index=False)
    recommendations.to_csv(folder / "recommendations.csv", index=False)
    metrics = model_metrics(recommendations, users)
    metrics.to_csv(folder / "models.csv", index=False)
    return metrics

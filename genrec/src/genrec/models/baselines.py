"""Classic, non-neural recommenders. Each returns a (users x n_cols) score matrix; rank with genrec.metrics.

random       - the floor
popularity   - how many users rated each movie
ItemKNN      - item-based collaborative filtering: cosine similarity between movies' user columns;
               a movie's score is its summed similarity to the user's history (optionally the last N only)
UserKNN      - user-based collaborative filtering: cosine similarity between users' rows; a movie's score is the
               summed similarity of the k most similar users who rated it
"""
from __future__ import annotations

import numpy as np


def interaction_matrix(histories: list, n_cols: int) -> np.ndarray:
    """users x movies 0/1 matrix (column 0 unused)."""
    X = np.zeros((len(histories), n_cols), dtype=np.float32)
    for r, h in enumerate(histories):
        X[r, h] = 1
    return X


def random_scores(n_users: int, n_cols: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((n_users, n_cols))


def popularity_scores(X: np.ndarray, n_users: int) -> np.ndarray:
    return np.tile(X.sum(axis=0), (n_users, 1))


def item_similarity(X: np.ndarray, k: int | None = None) -> np.ndarray:
    """Cosine similarity between movies (columns of X), keeping each movie's k nearest neighbours (None = all)."""
    norms = np.linalg.norm(X, axis=0)
    norms[norms == 0] = 1
    Xn = X / norms
    S = Xn.T @ Xn
    np.fill_diagonal(S, 0)
    if k is not None:
        drop = np.argpartition(-S, k, axis=0)[k:]
        np.put_along_axis(S, drop, 0, axis=0)
    return S


def itemknn_scores(histories: list, S: np.ndarray, recent: int | None = None) -> np.ndarray:
    """score(movie) = sum of its similarities to the history (all of it, or the `recent` most recent movies)."""
    H = np.zeros((len(histories), S.shape[0]), dtype=np.float32)
    for r, h in enumerate(histories):
        H[r, h[-recent:] if recent else h] = 1
    return H @ S


def userknn_scores(X: np.ndarray, rows: list, k: int | None = None, batch: int = 1000) -> np.ndarray:
    """For the users at `rows` of X: summed rows of their k most similar other users (cosine)."""
    norms = np.linalg.norm(X, axis=1)
    norms[norms == 0] = 1
    Xn = X / norms[:, None]
    out = np.zeros((len(rows), X.shape[1]), dtype=np.float32)
    for b in range(0, len(rows), batch):
        rb = rows[b:b + batch]
        sims = Xn[rb] @ Xn.T
        sims[np.arange(len(rb)), rb] = 0          # not similar to yourself
        if k is not None:
            drop = np.argpartition(-sims, k, axis=1)[:, k:]
            np.put_along_axis(sims, drop, 0, axis=1)
        out[b:b + len(rb)] = sims @ X
    return out

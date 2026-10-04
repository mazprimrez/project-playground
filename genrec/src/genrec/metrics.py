"""Ranking metrics for the next-movie test: full ranking over the catalogue, already-rated movies skipped.

hit@1   - the real next movie is ranked first
HR@10   - it is in the top 10
NDCG@10 - like HR@10, but a hit at rank r counts 1 / log2(r + 1), so higher ranks count more
"""
from __future__ import annotations

import math

import numpy as np

METRICS = ["hit@1", "HR@10", "NDCG@10"]


def rank_metrics(top: list, target) -> tuple:
    """(hit@1, hit@k, NDCG@k) for one user, given their ranked top-k list."""
    if target not in top:
        return 0.0, 0.0, 0.0
    rank = top.index(target)
    return float(rank == 0), 1.0, 1 / math.log2(rank + 2)


def mean_metrics(rows) -> dict:
    return dict(zip(METRICS, np.mean(np.asarray(rows, dtype=float).reshape(-1, 3), axis=0)))


def evaluate_lists(tops: list, targets: list) -> dict:
    """Mean metrics for ranked lists (one per user) against their targets."""
    return mean_metrics([rank_metrics(list(t), g) for t, g in zip(tops, targets)])


def top_k(scores: np.ndarray, histories: list, k: int = 10) -> np.ndarray:
    """Top-k columns per row of a (users x n_cols) score matrix, never column 0 (padding) or an already-rated movie.
    Stable argsort: ties keep column order, so results are reproducible."""
    scores = scores.astype(np.float64, copy=True)
    scores[:, 0] = -np.inf
    for r, h in enumerate(histories):
        scores[r, h] = -np.inf
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def evaluate_scores(scores: np.ndarray, histories: list, targets: list, k: int = 10) -> dict:
    """Full-ranking metrics from a score matrix (rows = users, columns = item indices 0..n)."""
    return evaluate_lists(top_k(scores, histories, k).tolist(), targets)

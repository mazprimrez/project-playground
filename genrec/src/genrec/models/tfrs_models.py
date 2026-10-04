"""TensorFlow Recommenders retrieval models. Needs the separate TensorFlow environment (`pip install -e ".[tfrs]"` in
its own venv): TFRS requires Keras 2 (tf_keras), which conflicts with nothing here but is heavy.

two-tower   - user-ID embedding vs movie embedding (ignores order)
sequential  - GRU over the last `context` movies vs movie embedding (next-movie retrieval)
Both use TFRS's Retrieval task: in-batch softmax (the other movies in the batch are the negatives).
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")   # TFRS needs Keras 2; must be set before importing tensorflow

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tf_keras as keras


class TwoTower(tfrs.Model):
    def __init__(self, n_users: int, n_cols: int, dim: int = 64):
        super().__init__()
        self.user_model = keras.layers.Embedding(n_users, dim)
        self.movie_model = keras.layers.Embedding(n_cols, dim)
        self.task = tfrs.tasks.Retrieval()

    def compute_loss(self, features, training=False):
        return self.task(self.user_model(features["user"]), self.movie_model(features["movie"]),
                         compute_metrics=False)

    def scores(self, user_rows) -> np.ndarray:
        user_vecs = self.user_model(np.asarray(user_rows, np.int32)).numpy()
        return user_vecs @ self.movie_model.embeddings.numpy().T


class SequentialRetrieval(tfrs.Model):
    def __init__(self, n_cols: int, dim: int = 64, context: int = 20):
        super().__init__()
        self.context = context
        self.query_model = keras.Sequential([keras.layers.Embedding(n_cols, dim, mask_zero=True), keras.layers.GRU(dim)])
        self.movie_model = keras.layers.Embedding(n_cols, dim)
        self.task = tfrs.tasks.Retrieval()

    def compute_loss(self, features, training=False):
        return self.task(self.query_model(features["context"]), self.movie_model(features["movie"]),
                         compute_metrics=False)

    def scores(self, histories) -> np.ndarray:
        ctx = np.array([left_pad(h, self.context) for h in histories], np.int32)
        return self.query_model.predict(ctx, batch_size=2048, verbose=0) @ self.movie_model.embeddings.numpy().T


def left_pad(items, length):
    items = items[-length:]
    return [0] * (length - len(items)) + items


def pairs_dataset(user_rows, histories, batch_size=4096, seed=42):
    """(user, movie) pairs from each user's history - training data for the two-tower model."""
    users, items = [], []
    for r, h in zip(user_rows, histories):
        users += [r] * len(h)
        items += h
    ds = tf.data.Dataset.from_tensor_slices({"user": np.array(users, np.int32), "movie": np.array(items, np.int32)})
    return ds.shuffle(len(users), seed=seed, reshuffle_each_iteration=True).batch(batch_size).cache()


def sequence_dataset(histories, context=20, batch_size=1024, seed=42):
    """(last `context` movies -> next movie) for every position of every training history."""
    contexts, labels = [], []
    for h in histories:
        for t in range(1, len(h)):
            contexts.append(left_pad(h[:t], context))
            labels.append(h[t])
    ds = tf.data.Dataset.from_tensor_slices({"context": np.array(contexts, np.int32), "movie": np.array(labels, np.int32)})
    return ds.shuffle(len(labels), seed=seed, reshuffle_each_iteration=True).batch(batch_size).cache(), len(labels)

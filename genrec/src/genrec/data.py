"""MovieLens-format data (ML-1M, MovieTweetings) and the leave-last-out split shared by every experiment.

The split: each user's ratings are ordered by timestamp (stable sort, so same-second ratings keep file order);
train on items[:-2], validate on items[-2], test on items[-1].
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ARTICLES = r"The|A|An|La|Le|Les|L'|Il|El|Das|Der|Die"
RATING_COLUMNS = ["UserID", "MovieID", "Rating", "Timestamp"]
SPLITS = {"val": -2, "test": -1}   # history = items[:cut], target = items[cut]


def load_ratings(data_dir) -> pd.DataFrame:
    """ratings.dat ('UserID::MovieID::Rating::Timestamp'). Read with ':' and the C parser, keeping every other
    column - ~10x faster than sep='::' with the python engine, same result."""
    return pd.read_csv(Path(data_dir) / "ratings.dat", sep=":", header=None, usecols=[0, 2, 4, 6],
                       names=RATING_COLUMNS)


def user_sequences(ratings: pd.DataFrame) -> pd.Series:
    """UserID -> that user's MovieIDs in rating order."""
    return ratings.sort_values(["UserID", "Timestamp"], kind="stable").groupby("UserID").MovieID.apply(list)


def history(items: list, split: str) -> list:
    return items[:SPLITS[split]]


def target(items: list, split: str):
    return items[SPLITS[split]]


def sample_test_users(users, n: int = 1000, seed: int = 0) -> list:
    """The evaluation users: random.Random(0).sample(all users, 1000) - the same 1,000 for every model."""
    users = list(users)
    return random.Random(seed).sample(users, min(n, len(users)))


def display_title(ml_title: str) -> str:
    """'City of Lost Children, The (Cité des enfants perdus, La) (1995)' -> 'The City of Lost Children (1995)'"""
    m = re.match(r"^(.*?)\s*\((\d{4})\)\s*$", ml_title)
    name, year = (m.group(1), m.group(2)) if m else (ml_title, None)
    name = re.sub(r"\s*\([^()]*\)", "", name).strip() or name   # drop alternate titles
    name = re.sub(rf"^(.*), ({ARTICLES})$",
                  lambda a: a[2] + ("" if a[2].endswith("'") else " ") + a[1], name)
    return f"{name} ({year})" if year else name


def load_movies(data_dir) -> pd.DataFrame:
    """movies_wiki.csv (MovieLens movies + matched Wikipedia plot/director/cast) with a unique `display` title -
    the form the LLM reads and writes."""
    movies = pd.read_csv(Path(data_dir) / "movies_wiki.csv")
    movies["Genres"] = movies["Genres"].fillna("")
    movies["display"] = movies["Title"].map(display_title)
    # two different movies must never share a display title (it is the answer for identify / next_movie)
    dup = movies["display"].duplicated(keep=False)
    movies.loc[dup, "display"] = movies.loc[dup, "Title"]
    assert movies["display"].is_unique, movies.loc[movies["display"].duplicated(keep=False), "display"]
    return movies


@dataclass
class ItemIndex:
    """MovieID <-> dense index 1..n (0 = padding), used by SASRec, the classic baselines and TFRS."""
    movie_ids: list
    to_idx: dict = field(init=False, repr=False)

    def __post_init__(self):
        self.movie_ids = [int(m) for m in self.movie_ids]
        self.to_idx = {m: i + 1 for i, m in enumerate(self.movie_ids)}

    @classmethod
    def from_ratings(cls, ratings: pd.DataFrame) -> "ItemIndex":
        return cls(sorted(ratings.MovieID.unique()))

    @property
    def n_cols(self) -> int:
        """Score-vector length: every movie plus the padding column 0."""
        return len(self.movie_ids) + 1

    def encode(self, movie_ids) -> list:
        return [self.to_idx[m] for m in movie_ids]

    def decode(self, idxs) -> list:
        return [self.movie_ids[i - 1] for i in idxs]


@dataclass
class Dataset:
    """Everything an experiment needs, loaded once."""
    data_dir: Path
    ratings: pd.DataFrame
    movies: pd.DataFrame
    sequences: pd.Series          # UserID -> MovieIDs in order
    index: ItemIndex

    @classmethod
    def load(cls, data_dir) -> "Dataset":
        ratings = load_ratings(data_dir)
        return cls(Path(data_dir), ratings, load_movies(data_dir), user_sequences(ratings), ItemIndex.from_ratings(ratings))

    @property
    def users(self) -> list:
        return list(self.sequences.index)

    def encoded(self) -> dict:
        """UserID -> item indices (1..n) in order."""
        return {u: self.index.encode(items) for u, items in self.sequences.items()}

    def titles(self) -> dict:
        """MovieID -> display title."""
        return dict(zip(self.movies.MovieID, self.movies.display))

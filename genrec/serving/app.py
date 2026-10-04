"""GenRec lookup API: pre-computed movie recommendations for 1,000 MovieLens test users, from up to five models
(random, most popular, TFRS sequential, SASRec, fine-tuned Qwen), with each model's probability.

No model runs at request time - responses are table lookups (artifacts/lookup/), so it is fast and runs
on the smallest machine.

Configuration (environment variables):
    GENREC_LOOKUP_DIR   folder with movies.csv, users.csv, recommendations.csv, models.csv (default artifacts/lookup)
    GENREC_HF_REPO      optional: a Hugging Face *dataset* repo with those files (downloaded at startup; for a private
                        repo also set HF_TOKEN)

Run:  uvicorn serving.app:app --reload        (interactive docs at http://localhost:8000/docs)
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from genrec.paths import ARTIFACTS_DIR

TABLES: dict = {}
POSTER_BASE = "https://image.tmdb.org/t/p/w342"   # TMDB image CDN, 342 px wide


def lookup_dir() -> Path:
    if os.environ.get("GENREC_HF_REPO"):
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(os.environ["GENREC_HF_REPO"], repo_type="dataset"))
    return Path(os.environ.get("GENREC_LOOKUP_DIR", ARTIFACTS_DIR / "lookup"))


def load_tables(folder: Path) -> dict:
    movies = pd.read_csv(folder / "movies.csv").fillna({"genres": ""}).set_index("movie_id", drop=False)
    users = pd.read_csv(folder / "users.csv")
    users["history"] = users["history"].map(json.loads)
    recs = pd.read_csv(folder / "recommendations.csv").sort_values(["user_id", "model", "rank"])
    models = pd.read_csv(folder / "models.csv")
    return {"movies": movies, "users": users.set_index("user_id", drop=False), "models": models,
            "recs": {key: g for key, g in recs.groupby(["user_id", "model"])}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    TABLES.update(load_tables(lookup_dir()))
    yield
    TABLES.clear()


app = FastAPI(title="GenRec lookup API", version="0.1.0", lifespan=lifespan,
              description="Pre-computed movie recommendations from several models for 1,000 MovieLens test users.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ schemas

class Movie(BaseModel):
    movie_id: int
    title: str
    genres: list[str]
    year: int | None = None
    poster_url: str | None = None   # TMDB poster, when movies.csv has a poster_path


class ModelInfo(BaseModel):
    model: str
    description: str
    hit_at_1: float
    hr_at_10: float
    ndcg_at_10: float


class User(BaseModel):
    user_id: int
    n_ratings: int
    history: list[Movie]          # most recent last
    next_movie: Movie             # what the user actually rated next (the test target)


class Recommendation(BaseModel):
    rank: int
    movie: Movie
    probability: float            # the model's P(next movie = this), see /models
    is_next_movie: bool           # True if this is the movie the user actually rated next


class Recommendations(BaseModel):
    user_id: int
    model: str
    hit: bool                     # the actual next movie is in this list
    recommendations: list[Recommendation]


# ------------------------------------------------------------------ helpers

def movie(movie_id: int) -> Movie:
    row = TABLES["movies"].loc[movie_id]
    year = int(row["year"]) if pd.notna(row["year"]) else None
    path = row.get("poster_path")
    poster = f"{POSTER_BASE}{path}" if isinstance(path, str) and path else None
    return Movie(movie_id=int(movie_id), title=row["title"], genres=[g for g in row["genres"].split("|") if g], year=year,
                 poster_url=poster)


def get_user(user_id: int) -> pd.Series:
    if user_id not in TABLES["users"].index:
        raise HTTPException(404, f"unknown user {user_id} - see GET /users for the 1,000 test users")
    return TABLES["users"].loc[user_id]


def recommendations_for(user_id: int, model: str, k: int) -> Recommendations:
    user = get_user(user_id)
    if model not in set(TABLES["models"]["model"]):
        raise HTTPException(404, f"unknown model {model!r} - see GET /models")
    g = TABLES["recs"].get((user_id, model))
    rows = [] if g is None else g.head(k).itertuples()
    recs = [Recommendation(rank=r.rank, movie=movie(r.movie_id), probability=r.probability,
                           is_next_movie=r.movie_id == user["next_movie_id"]) for r in rows]
    return Recommendations(user_id=user_id, model=model, hit=any(r.is_next_movie for r in recs), recommendations=recs)


# ------------------------------------------------------------------ endpoints

@app.get("/health")
def health():
    return {"status": "ok", "users": len(TABLES.get("users", [])), "models": TABLES["models"]["model"].tolist()}


@app.get("/models", response_model=list[ModelInfo])
def models():
    """The models, with their test metrics on these 1,000 users (full ranking, ~3,700 movies)."""
    return [ModelInfo(model=r["model"], description=r["description"], hit_at_1=r["hit@1"], hr_at_10=r["HR@10"],
                      ndcg_at_10=r["NDCG@10"])
            for r in TABLES["models"].to_dict("records")]


@app.get("/users", response_model=list[int])
def users(offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)):
    """IDs of the test users (paged)."""
    return TABLES["users"]["user_id"].iloc[offset:offset + limit].tolist()


@app.get("/users/{user_id}", response_model=User)
def user(user_id: int, history: int = Query(20, ge=0, le=500, description="how many recent movies to return")):
    u = get_user(user_id)
    recent = u["history"][-history:] if history else []
    return User(user_id=user_id, n_ratings=len(u["history"]) + 1, history=[movie(m) for m in recent],
                next_movie=movie(u["next_movie_id"]))


def default_model() -> str:
    """qwen when its lists are loaded, else the loaded model with the best HR@10."""
    models = TABLES["models"]
    return "qwen" if "qwen" in set(models["model"]) else models.sort_values("HR@10")["model"].iloc[-1]


@app.get("/users/{user_id}/recommendations", response_model=Recommendations)
def recommend(user_id: int, model: str | None = Query(None, description="see GET /models; default qwen if loaded, "
                                                                        "else the best by HR@10"),
              k: int = Query(10, ge=1, le=10)):
    return recommendations_for(user_id, model or default_model(), k)


@app.get("/users/{user_id}/compare", response_model=list[Recommendations])
def compare(user_id: int, k: int = Query(10, ge=1, le=10)):
    """Every model's list for this user - for a side-by-side view."""
    return [recommendations_for(user_id, m, k) for m in TABLES["models"]["model"]]


@app.get("/movies/{movie_id}", response_model=Movie)
def get_movie(movie_id: int):
    if movie_id not in TABLES["movies"].index:
        raise HTTPException(404, f"unknown movie {movie_id}")
    return movie(movie_id)

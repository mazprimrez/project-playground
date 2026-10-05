"""GenRec live inference (Cloud Run service `genrec-model`): the fine-tuned Qwen2.5-0.5B recommends the next movie for
any list of movies from its 4,000-movie catalogue (2010-2023). CPU only: one request at a time, tens of seconds each.

    POST /recommend {"movie_ids": [79132, 109487, 134130], "k": 10}      MovieLens IDs, oldest first
    POST /recommend {..., "explain": true}     + why: for each pick, the movies whose removal lowers its chance most

Environment: HF_TOKEN (secret, reads the private Hugging Face model repo), MODEL_REPO, NUM_BEAMS (fewer = faster),
NUM_THREADS (PyTorch threads: the vCPUs the service gets), MODEL_PATH (a local folder instead of the Hub, for testing).
"""
from __future__ import annotations

import math
import os
import threading
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field

from genrec.data import load_movies
from genrec.models.qwen import QwenRecommender

MODEL_REPO = os.environ.get("MODEL_REPO", "sparklingdust/genrec-qwen2.5-0.5b-movies")
NUM_BEAMS = int(os.environ.get("NUM_BEAMS", "20"))
EXPLAIN_LAST = 10              # explanations test the user's last 10 movies (each one costs a pass over all picks)
# os.cpu_count() can report the whole host, not the vCPUs this container gets: too many threads slow PyTorch down
NUM_THREADS = int(os.environ.get("NUM_THREADS") or os.cpu_count() or 2)
STATE: dict = {}
LOCK = threading.Lock()          # one generation at a time: they would only compete for the same CPU cores


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(NUM_THREADS)
    path = os.environ.get("MODEL_PATH") or snapshot_download(MODEL_REPO, token=os.environ.get("HF_TOKEN"))
    movies = load_movies(os.path.join(path, "catalogue"))
    STATE["rec"] = QwenRecommender(path, movies, device="cpu")
    STATE["catalogue"] = set(movies.MovieID)
    STATE["movies"] = movies.set_index("MovieID")
    yield
    STATE.clear()


app = FastAPI(title="GenRec live recommendations", lifespan=lifespan,
              description="Qwen2.5-0.5B fine-tuned to recommend the next movie. Slow on CPU: up to a minute per call.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


class RecommendRequest(BaseModel):
    movie_ids: list[int] = Field(min_length=1, max_length=50, description="MovieLens IDs the user watched, oldest first")
    k: int = Field(10, ge=1, le=10)
    explain: bool = Field(False, description=f"also explain each pick (tests your last {EXPLAIN_LAST} movies; "
                                             "several times slower)")


class Reason(BaseModel):
    movie_id: int
    title: str
    drop: float                   # how much the pick's probability falls without this movie (0.45 = -45%)


class Pick(BaseModel):
    rank: int
    movie_id: int
    title: str
    genres: list[str]
    year: int | None
    poster_url: str | None
    probability: float            # P(the model writes exactly this title); small and uncalibrated
    because: list[Reason] | None = None


class RecommendResponse(BaseModel):
    recommendations: list[Pick]
    seconds: float
    num_beams: int


def explain(rec, history: list, titles: list, base: list) -> list:
    """Leave-one-out: drop each of the last EXPLAIN_LAST movies from the history, re-score every pick, and keep, per
    pick, the (up to 3) movies whose removal lowers its probability most (by 5% or more). Measured on the model, not made up."""
    drops = {}
    for movie in history[-EXPLAIN_LAST:]:
        without = rec.title_logprobs([h for h in history if h != movie], titles)
        drops[movie] = [1 - math.exp(w - b) for w, b in zip(without, base)]
    out = []
    for i in range(len(titles)):
        ranked = sorted(drops, key=lambda m: drops[m][i], reverse=True)
        out.append([Reason(movie_id=int(m), title=rec.display[m], drop=round(drops[m][i], 3))
                    for m in ranked[:3] if drops[m][i] >= 0.05])
    return out


def movie_pick(rank: int, movie_id: int, title: str, probability: float, because: list | None = None) -> Pick:
    row = STATE["movies"].loc[movie_id]
    year = int(row["Year"]) if row["Year"] == row["Year"] else None           # NaN check
    poster = row.get("poster_path")
    return Pick(rank=rank, movie_id=int(movie_id), title=title, genres=[g for g in str(row["Genres"]).split("|") if g],
                year=year, poster_url=f"https://image.tmdb.org/t/p/w342{poster}" if isinstance(poster, str) else None,
                probability=probability, because=because)


@app.get("/")
def index():
    return {"model": MODEL_REPO, "ready": "rec" in STATE, "num_beams": NUM_BEAMS, "threads": torch.get_num_threads(),
            "cpu_count": os.cpu_count(), "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok" if "rec" in STATE else "loading"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    rec = STATE.get("rec")
    if rec is None:
        raise HTTPException(503, "the model is still loading - try again in a minute")
    unknown = [m for m in req.movie_ids if m not in STATE["catalogue"]]
    if unknown:
        raise HTTPException(400, f"not in the catalogue: {unknown}")
    history = list(dict.fromkeys(req.movie_ids))          # drop repeats, keep the order
    with LOCK:
        t0 = time.time()
        top = rec.recommend([history], k=req.k, num_beams=NUM_BEAMS)[0]
        titles = [rec.display[m] for m in top]
        base = rec.title_logprobs(history, titles) if top else []
        reasons = explain(rec, history, titles, base) if req.explain and top else [None] * len(top)
        seconds = time.time() - t0
    picks = [movie_pick(i + 1, m, t, math.exp(lp), r) for i, (m, t, lp, r) in enumerate(zip(top, titles, base, reasons))]
    return RecommendResponse(recommendations=picks, seconds=round(seconds, 1), num_beams=NUM_BEAMS)

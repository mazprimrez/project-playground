"""Eras Tour replay API: for every Eras Tour show, what Taylor Swift wore and played, what each method predicted from
the shows before it, and the points it would have scored in the Swift Alert Mastermind game.

No model runs at request time - the backtest is pre-computed (eras.lookup) and served from three small tables.

Configuration (environment variables):
    ERAS_LOOKUP_DIR   folder with shows.csv, picks.csv, entries.csv (default artifacts/lookup)

Run:  uvicorn serving.app:app --reload        (interactive docs at http://localhost:8000/docs)
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from eras import lookup
from eras.data import GAME, POINTS, SONG_POINTS, SHEET_URL
from eras.paths import ARTIFACTS_DIR

TABLES: dict = {}


def lookup_dir() -> Path:
    return Path(os.environ.get("ERAS_LOOKUP_DIR", ARTIFACTS_DIR / "lookup"))


def load_tables(folder: Path) -> dict:
    t = lookup.read(folder)
    picks = t["picks.csv"]
    return {"shows": t["shows.csv"].set_index("show", drop=False), "entries": t["entries.csv"],
            "picks": {show: g for show, g in picks.groupby("show")}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    TABLES.update(load_tables(lookup_dir()))
    yield
    TABLES.clear()


app = FastAPI(title="Eras Tour replay API", version="0.1.0", lifespan=lifespan,
              description="Predicted vs actual Eras Tour outfits and surprise songs, scored like Swift Alert "
                          "Mastermind.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# game order; the two song guesses take the place of "Surprise songs", the Lover guitar isn't tracked
CATEGORIES = [c for c, v in GAME.items() if v not in (None, "songs")]
CATEGORIES.insert(CATEGORIES.index("Surprise song dress") + 1, "Surprise song (guitar)")
CATEGORIES.insert(CATEGORIES.index("Surprise song (guitar)") + 1, "Surprise song (piano)")
MAX_POINTS = sum(SONG_POINTS if c.startswith("Surprise song (") else POINTS for c in CATEGORIES)


# ------------------------------------------------------------------ schemas

class Entry(BaseModel):
    entry: str
    name: str
    description: str
    points_per_show: float        # average over the replayed shows, out of max_points
    outfit_points: float
    song_points: float


class Summary(BaseModel):
    shows: int
    max_points: int               # 13 outfits x 4 + 2 songs x 2 (the Lover guitar, 4 more, isn't in the data)
    categories: list[str]
    entries: list[Entry]
    source: str


class Show(BaseModel):
    show: int
    date: str
    city: str
    night: int
    leg: str
    points: dict[str, float]      # entry -> points at this show


class Pick(BaseModel):
    pick: str | None
    probability: float | None     # the choice model's probability for its pick
    correct: bool | None
    points: float


class Category(BaseModel):
    category: str
    max_points: int
    actual: str                   # what she wore; for songs, the songs played on that instrument (" / ")
    picks: dict[str, Pick]        # entry -> its pick


class ShowDetail(Show):
    categories: list[Category]


# ------------------------------------------------------------------ helpers

def entry_methods() -> dict[str, tuple[str, str]]:
    return {r.entry: (r.outfit_method, r.song_method) for r in TABLES["entries"].itertuples()}


def none(x):
    return None if pd.isna(x) else x


def show_points(show: int) -> dict[str, float]:
    g = TABLES["picks"][show]
    songs = g.category.str.startswith("Surprise song (")
    return {e: float(g[(g.method == o) & ~songs].points.sum() + g[(g.method == s) & songs].points.sum())
            for e, (o, s) in entry_methods().items()}


def show_info(show: int) -> Show:
    r = TABLES["shows"].loc[show]
    return Show(show=show, date=r["date"], city=r["city"], night=int(r["night"]), leg=r["leg"],
                points=show_points(show))


# ------------------------------------------------------------------ endpoints

@app.get("/health")
def health():
    return {"status": "ok", "shows": len(TABLES.get("shows", [])), "entries": TABLES["entries"]["entry"].tolist()}


@app.get("/summary", response_model=Summary)
def summary():
    """The methods ("entries") with their average game points per show, best first."""
    e = TABLES["entries"].sort_values("points_per_show", ascending=False)
    return Summary(shows=len(TABLES["shows"]), max_points=MAX_POINTS, categories=CATEGORIES, source=SHEET_URL,
                   entries=[Entry(**{k: r[k] for k in Entry.model_fields}) for r in e.to_dict("records")])


@app.get("/shows", response_model=list[Show])
def shows():
    """Every replayed show (the first 10 are only used for learning), with each entry's points."""
    return [show_info(int(s)) for s in TABLES["shows"].index]


@app.get("/shows/{show}", response_model=ShowDetail)
def show_detail(show: int):
    """One show: per game category, what she wore / played and every entry's pick."""
    if show not in TABLES["picks"]:
        raise HTTPException(404, f"unknown show {show} - see GET /shows")
    g = TABLES["picks"][show].set_index(["category", "method"])
    cats = []
    for c in CATEGORIES:
        if c not in g.index.get_level_values(0):
            continue                                       # e.g. no TTPD set before the Paris shows
        rows = g.loc[c]
        song = c.startswith("Surprise song (")
        picks = {}
        for e, (o, s) in entry_methods().items():
            m = s if song else o
            if m not in rows.index:
                continue
            r = rows.loc[m]
            picks[e] = Pick(pick=none(r["pick"]), probability=none(r["probability"]),
                            correct=None if pd.isna(r["correct"]) else bool(r["correct"]), points=float(r["points"]))
        cats.append(Category(category=c, max_points=SONG_POINTS if song else POINTS,
                             actual=rows["truth"].iloc[0], picks=picks))
    return ShowDetail(**show_info(show).model_dump(), categories=cats)

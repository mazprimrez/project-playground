"""The API's tables, built from a backtest (`backtest.run`). The serving app only reads these; nothing is fitted at
request time.

    shows.csv     show, date, city, night, leg
    picks.csv     show, category, method, pick, probability, truth, correct, points   (the backtest rows)
    entries.csv   entry, name, description, outfit_method, song_method, points_per_show, outfit_points, song_points

An entry is a full game card: one method for the 13 outfit categories plus one for the two surprise song guesses.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import Tour

# entry -> (name, outfit method, song method, description); the article didn't guess songs, so the baselines all use
# the same simple song guess and differ only in their outfits
ENTRIES = {
    "choice": ("Choice model", "choice", "choice",
               "Conditional logit: features per candidate (how often / how recently / where it was worn), weights "
               "learned from the earlier shows, shared across categories plus a category-specific adjustment."),
    "prophet": ("Prophet (the article)", "prophet", "most_played",
                "The original Medium experiment: one Prophet forecast per outfit version on its worn / not worn "
                "series over the show dates; songs: the most played so far."),
    "prophet_lags": ("Prophet, improved", "prophet_lags", "most_played",
                     "Prophet with what the earlier shows say about each version as regressors (worn at the previous "
                     "show, earlier in this city, shows since last worn, share of the last 10), on a show-number "
                     "axis without seasonality; songs: the most played so far."),
    "prophet_night": ("Prophet + night #", "prophet_night", "most_played",
                      "The article's Prophet with the night number in the city (1, 2, 3...) as an extra regressor; "
                      "songs: the most played so far."),
    "recent": ("Worn most lately", "recent", "most_played",
               "The version worn most in the last 10 shows; songs: the most played so far."),
    "most_worn": ("Worn most overall", "most_worn", "most_played",
                  "The version worn most so far; songs: the most played so far."),
    "rotation": ("Rotation", "rotation", "most_played",
                 "The version not worn for the longest (of those worn twice or more); songs: the most played so far."),
}
FILES = ["shows.csv", "picks.csv", "entries.csv"]


def is_song(category: pd.Series) -> pd.Series:
    return category.str.startswith("Surprise song (")


def entries(results: pd.DataFrame) -> pd.DataFrame:
    n = results.show.nunique()
    rows = []
    for entry, (name, outfit, song, description) in ENTRIES.items():
        if outfit not in set(results.method):
            continue                                     # e.g. a backtest run without prophet
        o = results[(results.method == outfit) & ~is_song(results.category)].points.sum() / n
        s = results[(results.method == song) & is_song(results.category)].points.sum() / n
        rows.append({"entry": entry, "name": name, "description": description, "outfit_method": outfit,
                     "song_method": song, "points_per_show": o + s, "outfit_points": o, "song_points": s})
    return pd.DataFrame(rows)


def build(tour: Tour, results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    shows = tour.shows.assign(date=tour.shows.date.dt.strftime("%Y-%m-%d"))
    shows = shows[shows.show.isin(results.show)]
    return {"shows.csv": shows, "picks.csv": results, "entries.csv": entries(results)}


def write(folder, tables: dict[str, pd.DataFrame]) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(folder / name, index=False)


def read(folder) -> dict[str, pd.DataFrame]:
    folder = Path(folder)
    return {name: pd.read_csv(folder / name, keep_default_na=False, na_values=[""]) for name in FILES}


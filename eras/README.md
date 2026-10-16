# Eras Tour replay: guessing Taylor Swift's outfits

Before every Eras Tour show, Swift Alert's **Mastermind** game asked fans to guess the night's outfit versions and
surprise songs. There are 15 picks at 4 points each; the surprise songs are one guess per instrument, at 2 points
each. This project replays the whole tour (2023-2024, 149 shows). At every show, each method predicts using **only
the earlier shows**, and its picks are scored like the game.

It improves on my Medium experiment,
[Beating my own self on Swift Alert Mastermind](https://medium.com/@sparklingdust/beating-my-own-self-on-swift-alert-mastermind-game-in-guessing-taylor-swift-eras-tour-outfit-52b6535e0d8f),
which forecast every outfit version with Prophet.

## Results

There are 139 replayed shows; the first 10 are only used for learning. The maximum is 56 points: the Lover guitar
(4 points) isn't in the data.

| entry (outfits + songs) | points per show | outfits | songs |
|---|---|---|---|
| **Choice model** | **21.9** | 21.8 | 0.10 |
| Prophet (the article) + most played song | 17.1 | 17.0 | 0.12 |
| Worn most in the last 10 shows + most played song | 15.7 | 15.6 | 0.12 |
| Worn most overall + most played song | 15.0 | 14.9 | 0.12 |
| Rotation (longest unworn) + most played song | 15.0 | 14.9 | 0.12 |

The choice model scores **+28%** over the article's method and beats it at 91 of the 139 shows (16 ties, 32 worse).

Share of shows predicted right, by category:

| category | choice model | Prophet | worn most lately | rotation |
|---|---|---|---|---|
| Reputation outfit | **0.96** | **0.96** | **0.96** | 0.87 |
| Fearless dress | **0.58** | 0.38 | 0.38 | 0.32 |
| Midnights bodysuit | **0.56** | 0.42 | 0.50 | 0.27 |
| Karma jacket | **0.49** | 0.43 | 0.32 | 0.03 |
| Surprise song dress | **0.47** | 0.24 | 0.21 | 0.21 |
| Red shirt | **0.40** | 0.29 | 0.19 | **0.40** |
| Speak Now dress | **0.37** | 0.22 | 0.22 | 0.11 |
| Lover bodysuit | 0.36 | 0.32 | 0.27 | **0.44** |
| The Man jacket | 0.33 | 0.31 | 0.29 | **0.46** |
| Midnights shirt | **0.30** | 0.19 | 0.18 | 0.13 |
| Folkmore dress | **0.28** | 0.26 | 0.22 | 0.21 |
| 1989 set | **0.27** | 0.18 | 0.13 | 0.21 |
| TTPD outfit (dress + set) | **0.18** | 0.11 | 0.08 | 0.12 |

What we learned:
- **The outfits follow patterns a choice model can learn.** These include repeating within a run of nights in one
  city, saving versions for a new leg, and rotation in some categories (the Lover bodysuit, The Man jacket). A
  conditional logit scores every candidate on such features. It learns one set of weights from every category (a
  few thousand past choices instead of ~100 per category), plus a small per-category adjustment, so each category
  can use the patterns that fit it.
- **Prophet is the wrong tool here.** It forecasts each version's worn / not worn series on its own, as a smooth
  trend over calendar time. The versions of one category never compete with each other, and "worn last night in
  this city" is invisible to it.
- **Surprise songs are luck.** Each night she played a different song (or mashup) per instrument from a pool of
  ~200 songs. Every method is correct at ~1-5% of the shows, and the choice model's song guess is no better
  than the most played song so far.

## How it works

- `src/eras/data.py`: downloads and parses the two tabs of the fan sheet into a `Tour`:
  - shows (date, city, night, leg);
  - the version worn per game category;
  - the guitar and piano songs per show;
  - the song catalogue.

  Song names are matched to the catalogue through abbreviations, aliases and unique prefixes.
- `src/eras/models.py`: the conditional logit (numpy, Adam), the outfit and song features, the baselines, and the
  article's Prophet method.
- `src/eras/backtest.py`: the walk-forward replay and game scoring. The TTPD outfit is predicted as its dress and
  its two-piece set separately, then combined.
- `src/eras/lookup.py`: the API's three tables (shows, every pick, entry summary).
- `serving/app.py`: the replay API, which only reads the tables (`ERAS_LOOKUP_DIR`).
- `notebooks/eras_backtest.ipynb`: the data, the replay, charts, and saving the results and tables.

```bash
pip install -e ".[serve,dev,notebooks,prophet]"
jupyter lab notebooks/eras_backtest.ipynb          # downloads the data, replays the tour (~3 min with Prophet)
uvicorn serving.app:app --reload                   # API docs at http://localhost:8000/docs
pytest                                             # synthetic data, no download needed
```

API: `GET /summary` returns the entries and their average points. `GET /shows` lists every show with each entry's
points, and `GET /shows/{show}` returns the worn version and every entry's pick per category. On the site it is
served under `/api/eras/`, and the replay page is at `/projects/eras`.

## Data

The outfit and surprise song data come from a fan-made Eras Tour tracker (a public Google Sheet:
[link](https://docs.google.com/spreadsheets/d/1caBoaDi1id6k4s1sP1bTbo846inOXzN13p0C5OrFY2E)). Thanks to its
maintainers. It is downloaded by `eras.data.download` and not stored in this repo.

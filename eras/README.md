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
| **Prophet, improved** + most played song | **22.0** | 21.9 | 0.12 |
| **Choice model** | **21.9** | 21.8 | 0.10 |
| Prophet (the article) + most played song | 17.1 | 17.0 | 0.12 |
| Prophet + night number as a regressor + most played song | 16.8 | 16.7 | 0.12 |
| Worn most in the last 10 shows + most played song | 15.7 | 15.6 | 0.12 |
| Worn most overall + most played song | 15.0 | 14.9 | 0.12 |
| Rotation (longest unworn) + most played song | 15.0 | 14.9 | 0.12 |

Both improvements score about **+28%** over the article's method. The choice model beats the article's Prophet at
91 of the 139 shows, the improved Prophet at 87. Against each other they are even: the choice model is better at 56
shows, the improved Prophet at 62, and 21 are ties.

Share of shows predicted right, by category:

| category | choice model | Prophet, improved | Prophet (article) | worn most lately | rotation |
|---|---|---|---|---|---|
| Reputation outfit | 0.96 | **0.99** | 0.96 | 0.96 | 0.87 |
| Midnights bodysuit | 0.56 | **0.60** | 0.42 | 0.50 | 0.27 |
| Fearless dress | **0.58** | 0.50 | 0.38 | 0.38 | 0.32 |
| Karma jacket | **0.49** | **0.49** | 0.43 | 0.32 | 0.03 |
| Surprise song dress | **0.47** | 0.42 | 0.24 | 0.21 | 0.21 |
| The Man jacket | 0.33 | 0.45 | 0.31 | 0.29 | **0.46** |
| Lover bodysuit | 0.36 | 0.41 | 0.32 | 0.27 | **0.44** |
| Red shirt | **0.40** | 0.38 | 0.29 | 0.19 | **0.40** |
| Speak Now dress | **0.37** | 0.28 | 0.22 | 0.22 | 0.11 |
| Midnights shirt | 0.30 | **0.34** | 0.19 | 0.18 | 0.13 |
| Folkmore dress | 0.28 | **0.29** | 0.26 | 0.22 | 0.21 |
| 1989 set | 0.27 | **0.29** | 0.18 | 0.13 | 0.21 |
| TTPD outfit (dress + set) | **0.18** | 0.11 | 0.11 | 0.08 | 0.12 |

### Improving Prophet, one step at a time

Each row adds one change to the row above it (outfit points per show; songs left out):

| Prophet setup | outfit points | right | mean probability |
|---|---|---|---|
| the article (calendar dates, default settings) | 17.0 | 34% | 53% |
| + show number as the time axis (no gaps between legs) | 15.7 | 32% | 50% |
| + no weekly / yearly seasonality | 15.3 | 31% | 45% |
| + logistic growth between 0 and 1 | 14.5 | 29% | 46% |
| + a more flexible trend (changepoint prior 0.5) | 15.5 | 31% | 44% |
| **+ past-show regressors** (= Prophet, improved) | **21.9** | **44%** | 50% |
| the article + past-show regressors only | 21.1 | 42% | 57% |

The past-show regressors are four facts about each version, all from the shows before this one: worn at the
previous show, worn earlier in this city, shows since last worn, and share of the last 10 shows.

What we learned:
- **The features matter, not the model.** Prophet's own tools don't help. A time axis, seasonality, growth curves
  and trend flexibility all make it worse, and so does the night number (16.8). Give Prophet the same kind of
  information the choice model has, about how often, how recently and where each version was worn, and it jumps
  from 17.0 to 21.9 outfit points. Two very different models land within 0.2 points of each other.
- **The outfits follow patterns.** These include repeating within a run of nights in one city, saving versions for
  a new leg, and rotation in some categories (the Lover bodysuit, The Man jacket). The choice model compares the
  versions of a category directly. It learns one set of weights from every category (a few thousand past choices),
  plus a small per-category adjustment. Prophet still forecasts each version alone, but the regressors carry the
  same signal.
- **Probabilities:** the choice model is calibrated. It is 44% sure on average and right 44% of the time. Prophet's
  forecasts, normalised over the versions, were overconfident in the article's setup (53% sure, 34% right). The
  improved Prophet is close (50% sure, 44% right).
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
- `src/eras/models.py`: the conditional logit (numpy, Adam), the outfit and song features, the baselines, and
  Prophet: the article's method and its variants (`PROPHET_VARIANTS`).
- `src/eras/backtest.py`: the walk-forward replay and game scoring. The TTPD outfit is predicted as its dress and
  its two-piece set separately, then combined.
- `src/eras/lookup.py`: the API's three tables (shows, every pick, entry summary).
- `serving/app.py`: the replay API, which only reads the tables (`ERAS_LOOKUP_DIR`).
- `notebooks/eras_backtest.ipynb`: the data, the replay, charts, and saving the results and tables.

```bash
pip install -e ".[serve,dev,notebooks,prophet]"
jupyter lab notebooks/eras_backtest.ipynb          # downloads the data, replays the tour (~10+ min with Prophet)
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

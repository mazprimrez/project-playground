# %% [markdown]
# # MovieTweetings: prepare a cold-start dataset (same format as ML-1M)
#
# [MovieTweetings](https://github.com/sidooms/MovieTweetings) (Dooms et al., CrowdRec workshop at RecSys 2013 -
# please cite it) collects movie ratings people tweeted from the IMDb app, 2013-2021. Unlike ML-1M, where every
# user has rated at least 20 movies, most users here rated only one or two: a realistic **cold-start** setting.
#
# This notebook
# 1. loads the raw files (`data/movietweetings/raw/`, from `python scripts/download_data.py movietweetings`),
# 2. shows how skewed the user histories are,
# 3. matches every movie to the **same Wikipedia plot dataset** used for ML-1M (`training/phase-1.ipynb`, Kaggle
#    `jrobischon/wikipedia-movie-plots`) with the same matching rules, so plots, directors and cast are comparable,
# 4. writes `data/movietweetings/movies_wiki.csv` and `data/movietweetings/ratings.dat` in the ML-1M format, so the
#    training and evaluation notebooks can run on it by changing the data folder.
#
# Movie IDs are IMDb IDs (`0114508` = tt0114508); ratings are on IMDb's 0-10 scale (treated as "watched" anyway).

# %%
import re
import unicodedata
from pathlib import Path

import kagglehub
import pandas as pd
from rapidfuzz import fuzz, process

from genrec.paths import MOVIETWEETINGS_DIR

DATA = MOVIETWEETINGS_DIR            # data/movietweetings (raw files: python scripts/download_data.py movietweetings)
RAW = DATA / "raw"

ratings = pd.read_csv(RAW / "ratings.dat", sep="::", engine="python",
                      names=["UserID", "MovieID", "Rating", "Timestamp"])
movies_df = pd.read_csv(RAW / "movies.dat", sep="::", engine="python", names=["MovieID", "Title", "Genres"])
print(f"ratings {ratings.shape}, movies {movies_df.shape}, users {ratings.UserID.nunique():,}, "
      f"{pd.to_datetime(ratings.Timestamp.min(), unit='s').date()} .. {pd.to_datetime(ratings.Timestamp.max(), unit='s').date()}")

# %% [markdown]
# ### Clean the raw movie list
# - ~300 titles contain HTML entities (`Mr. &amp; Mrs. Smith`, and a malformed `Dracula&x27;s Daughter`).
# - 5 movies are listed twice (sometimes one copy has no genres): keep one row per IMDb id, preferring one with genres.

# %%
import html

entities = movies_df["Title"].str.contains(r"&[#a-zA-Z0-9]+;", regex=True)
movies_df["Title"] = movies_df["Title"].str.replace("&x27;", "'", regex=False).map(html.unescape)
before = len(movies_df)
movies_df = (movies_df.assign(_no_genre=movies_df["Genres"].isna())
             .sort_values(["MovieID", "_no_genre"]).drop_duplicates("MovieID").drop(columns="_no_genre")
             .sort_index().reset_index(drop=True))
assert ratings["MovieID"].isin(movies_df["MovieID"]).all()
print(f"unescaped {entities.sum()} titles; dropped {before - len(movies_df)} duplicate rows -> {len(movies_df):,} movies")

# %% [markdown]
# ## 1. How skewed are the user histories?

# %%
per_user = ratings.groupby("UserID").size()
bins = [0, 1, 2, 4, 9, 19, 49, per_user.max()]
labels = ["1", "2", "3-4", "5-9", "10-19", "20-49", "50+"]
bucket = pd.cut(per_user, bins=bins, labels=labels)
skew = pd.DataFrame({
    "users": bucket.value_counts().reindex(labels),
    "share of users": bucket.value_counts(normalize=True).reindex(labels),
    "share of ratings": per_user.groupby(bucket, observed=False).sum().reindex(labels) / len(ratings),
})
print(f"ratings per user: median {per_user.median():.0f}, mean {per_user.mean():.1f}, max {per_user.max()} "
      f"(ML-1M: median 96, minimum 20)")
skew.style.format({"users": "{:,}", "share of users": "{:.1%}", "share of ratings": "{:.1%}"})

# %% [markdown]
# ## 2. Wikipedia plots (same source and matching rules as `training/phase-1.ipynb`)

# %%
wiki = pd.read_csv(Path(kagglehub.dataset_download("jrobischon/wikipedia-movie-plots")) / "wiki_movie_plots_deduped.csv")
print(wiki.shape, "release years", wiki["Release Year"].min(), "-", wiki["Release Year"].max())

# %%
ARTICLES = r"The|A|An|La|Le|Les|L'|Il|El|Das|Der|Die"
NUMBERS = {"ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7", "viii": "8",
           "one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}

def normalize(title):
    """'Naked Gun 2 1/2: The Smell of Fear, The' -> 'naked gun 2 1 2 smell of fear'"""
    t = unicodedata.normalize("NFKD", str(title)).encode("ascii", "ignore").decode()
    t = re.sub(rf"^(.*), ({ARTICLES})$", r"\2 \1", t.strip())   # "Matrix, The" -> "The Matrix"
    t = t.lower().replace("&", " and ").replace("+", " and ")
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    words = [NUMBERS.get(w, w) for w in t.split() if w != "part"]
    if words and words[0] in {"the", "a", "an"}:
        words = words[1:]
    return " ".join(words)

def split_names(title):
    """'Mad Max 2 (a.k.a. The Road Warrior)' -> ['Mad Max 2', 'The Road Warrior']"""
    alts = re.findall(r"\(([^()]*)\)", title)
    main = re.sub(r"\s*\([^()]*\)", "", title).strip()
    return [main] + [re.sub(r"^(a\.?k\.?a\.?)\s*", "", a, flags=re.I) for a in alts]

def title_keys(title):
    """One row per name: (key, base, is_main). base = part before ':' / ' - ' (for subtitle matches).
    The subtitle itself is added as an alternate name: 'Star Wars: Episode V - The Empire Strikes Back'."""
    names = split_names(title)
    main_parts = re.split(r":| - ", names[0])
    if len(main_parts) > 1 and len(main_parts[-1].split()) >= 3:
        names.append(main_parts[-1])
    rows = []
    for i, name in enumerate(names):
        base = re.split(r":| - ", name)[0]
        rows.append((normalize(name), normalize(base), i == 0))
    return [r for r in rows if r[0]]

# %%
# MovieTweetings side: "Toy Story (1995)" -> Year 1995 + title keys
mt = movies_df[["MovieID", "Title"]].copy()
parsed = mt["Title"].str.extract(r"^(?P<Name>.*?)\s*\((?P<Year>\d{4})\)\s*$")
mt["Year"] = parsed["Year"].astype(int)
mt["Keys"] = parsed["Name"].map(title_keys)

# Wiki side: one row per (movie, name), grouped by release year for fast candidate lookup
wk = wiki[["Title", "Release Year", "Origin/Ethnicity"]].copy()
wk["wiki_idx"] = wiki.index
wk["Keys"] = wk["Title"].map(title_keys)
wk = wk.explode("Keys").dropna(subset=["Keys"])
wk[["key", "base", "is_main"]] = pd.DataFrame(wk.pop("Keys").tolist(), index=wk.index)
wk["compact"] = wk["key"].str.replace(" ", "")
wk_by_year = {y: g.reset_index(drop=True) for y, g in wk.groupby("Release Year")}

# %%
METHODS = ["exact", "fuzzy", "subtitle"]   # in order of trust

def match_movie(keys, year, fuzzy_cutoff=90):
    """Best wiki row for one movie -> (wiki_idx, score, method, year_diff) or None.
    exact    : same normalized title (ignoring spaces), release year within +-1 (same year for alternate titles)
    fuzzy    : token_sort_ratio >= fuzzy_cutoff, same year     ('U.S. Marshalls' ~ 'U.S. Marshals')
    subtitle : one title is the other minus its subtitle, same year  ('Godzilla 2000' ~ 'Godzilla 2000: Millennium')
    """
    near = [wk_by_year[y] for y in (year - 1, year, year + 1) if y in wk_by_year]
    if not near:
        return None
    near = pd.concat(near, ignore_index=True)
    same = near[near["Release Year"] == year].reset_index(drop=True)
    found = []
    for key, base, is_main in keys:
        for _, c in near[near["compact"] == key.replace(" ", "")].iterrows():
            dy = abs(c["Release Year"] - year)
            if dy == 0 or (is_main and c["is_main"]):
                found.append(("exact", 100.0, dy, c))
        if len(same):
            for _, score, pos in process.extract(key, same["key"], scorer=fuzz.token_sort_ratio,
                                                 score_cutoff=fuzzy_cutoff, limit=3):
                found.append(("fuzzy", score, 0, same.iloc[pos]))
            sub = same[((same["base"] == key) & (same["key"] != key)) |
                       ((same["key"] == base) & (base != key))]
            for _, c in sub.iterrows():
                found.append(("subtitle", 100.0, 0, c))
    if not found:
        return None
    method, score, dy, c = min(found, key=lambda f: (METHODS.index(f[0]), -f[1], f[2],
                                                     f[3]["Origin/Ethnicity"] != "American"))
    return c["wiki_idx"], score, method, dy


res = [match_movie(k, y) for k, y in zip(mt["Keys"], mt["Year"])]
matches = pd.DataFrame([r if r else (None, None, None, None) for r in res],
                       columns=["wiki_idx", "match_score", "match_method", "year_diff"], index=mt.index)
mt = pd.concat([mt, matches], axis=1)

# a wiki row can only belong to one movie: keep the most trustworthy claim
mt["_rank"] = mt["match_method"].map({m: i for i, m in enumerate(METHODS)})
mt = mt.sort_values(["_rank", "match_score", "year_diff"], ascending=[True, False, True])
dup = mt["wiki_idx"].notna() & mt.duplicated("wiki_idx")
print(f"dropped {dup.sum()} duplicate claims on the same Wikipedia film")
mt.loc[dup, ["wiki_idx", "match_score", "match_method", "year_diff"]] = None
mt = mt.drop(columns="_rank").sort_index()

# %%
movies = movies_df.merge(
    mt[["MovieID", "Year", "wiki_idx", "match_score", "match_method"]], on="MovieID", how="left"
).merge(
    wiki.rename(columns={"Title": "Wiki Title", "Genre": "Wiki Genre"}),
    left_on="wiki_idx", right_index=True, how="left",
).drop(columns="wiki_idx")
movies["match_method"].value_counts(dropna=False)

# %% [markdown]
# ## 3. Coverage
#
# Per movie the coverage is low (short films, documentaries, foreign titles and everything after 2017 are missing from
# the Kaggle dataset), but popular movies are covered much better - and those carry most of the ratings.

# %%
has_plot = movies.set_index("MovieID")["Plot"].notna()
rated = ratings.MovieID.map(has_plot)
print(f"movies with a plot:               {has_plot.mean():.1%} ({has_plot.sum():,} of {len(has_plot):,})")
print(f"  ... released up to 2017:        {has_plot[movies.set_index('MovieID')['Year'] <= 2017].mean():.1%}")
print(f"ratings whose movie has a plot:   {rated.mean():.1%}")
pop = ratings.groupby("MovieID").size()
for k in [5, 20, 100]:
    ids = pop[pop >= k].index
    print(f"movies with >= {k:3d} ratings:        {has_plot[ids].mean():.1%} have a plot ({len(ids):,} movies)")
by_year = movies.assign(decade=(movies.Year // 10) * 10).groupby("decade")["Plot"].apply(lambda p: p.notna().mean())
by_year.rename("plot coverage").to_frame().T.style.format("{:.0%}")

# %%
# review a sample of the non-exact matches
movies[movies["match_method"].isin(["fuzzy", "subtitle"])].sample(15, random_state=0)[
    ["Title", "Wiki Title", "Release Year", "Origin/Ethnicity", "match_score", "match_method"]]

# %% [markdown]
# ## 4. Save in the ML-1M format
#
# - `Genres`: about 50 movies have none in MovieTweetings - left empty.
# - `Title` must be unique (it is the answer for the `identify` / `next_movie` tasks): MovieTweetings has a few
#   different films with the same title and year, so those get their IMDb id: `Hamlet [tt0123456] (1990)`.

# %%
movies["Genres"] = movies["Genres"].fillna("")
dupe = movies["Title"].duplicated(keep=False)
movies.loc[dupe, "Title"] = movies.loc[dupe].apply(
    lambda r: re.sub(r"\s*\((\d{4})\)$", rf" [tt{r.MovieID:07d}] (\1)", r.Title), axis=1)
assert movies["Title"].is_unique
print(f"{dupe.sum()} movies with a shared title got their IMDb id, e.g.:")
print(movies.loc[dupe, "Title"].head(4).to_string())

movies.to_csv(DATA / "movies_wiki.csv", index=False)
ratings[["UserID", "MovieID", "Rating", "Timestamp"]].to_csv(DATA / "ratings.dat", sep=":", header=False, index=False)
# pandas only writes single-character separators: turn ":" into "::" like ML-1M (no ":" inside these four numbers)
path = DATA / "ratings.dat"
path.write_text(path.read_text().replace(":", "::"))
check = pd.read_csv(path, sep="::", engine="python", names=["UserID", "MovieID", "Rating", "Timestamp"])
assert check.equals(ratings[["UserID", "MovieID", "Rating", "Timestamp"]])
print(f"saved {movies.shape} -> {DATA / 'movies_wiki.csv'}")
print(f"saved {check.shape} -> {path}")

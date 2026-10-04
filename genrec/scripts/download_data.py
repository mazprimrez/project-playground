"""Download the raw datasets into data/ (they are not in the repository).

    python scripts/download_data.py              # both
    python scripts/download_data.py ml-1m
    python scripts/download_data.py movietweetings

Then build the Wikipedia-enriched movie files (movies_wiki.csv):
    ML-1M:          notebooks/movielens/phase-1.ipynb
    MovieTweetings: notebooks/coldstart/01_movietweetings_prep.ipynb
Both match movies to the Kaggle dataset jrobischon/wikipedia-movie-plots (downloaded with kagglehub).

Licenses: MovieLens (GroupLens) may be used for research but not redistributed - see data/ml-1m/README.
MovieTweetings: please cite Dooms et al., CrowdRec 2013 (see data/movietweetings/raw/README.md).
"""
import io
import sys
import urllib.request
import zipfile

from genrec.paths import ML1M_DIR, MOVIETWEETINGS_DIR

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MT_URL = "https://raw.githubusercontent.com/sidooms/MovieTweetings/master/{}"


def download(url: str) -> bytes:
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as r:
        return r.read()


def ml1m():
    ML1M_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(download(ML1M_URL))) as z:
        for name in z.namelist():
            if not name.endswith("/"):
                (ML1M_DIR / name.split("/")[-1]).write_bytes(z.read(name))
    print(f"-> {ML1M_DIR}: {sorted(p.name for p in ML1M_DIR.iterdir())}")


def movietweetings():
    raw = MOVIETWEETINGS_DIR / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in ["ratings.dat", "movies.dat", "users.dat"]:
        (raw / name).write_bytes(download(MT_URL.format(f"latest/{name}")))
    (raw / "README.md").write_bytes(download(MT_URL.format("README.md")))
    print(f"-> {raw}: {sorted(p.name for p in raw.iterdir())}")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["ml-1m", "movietweetings"]
    for t in targets:
        {"ml-1m": ml1m, "movietweetings": movietweetings}[t]()

"""Download MovieLens-1M into data/ (it is not in the repository).

    python scripts/download_data.py

Then build the Wikipedia-enriched movie file (movies_wiki.csv) with notebooks/movielens/phase-1.ipynb, which matches
movies to the Kaggle dataset jrobischon/wikipedia-movie-plots (downloaded with kagglehub).

License: MovieLens (GroupLens) may be used for research but not redistributed - see data/ml-1m/README.
"""
import io
import urllib.request
import zipfile

from genrec.paths import ML1M_DIR

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


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


if __name__ == "__main__":
    ml1m()

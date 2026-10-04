"""Default locations. Every function that reads data also takes an explicit directory, so the package works the same
from this repo, from Colab (paths on Google Drive) or from the serving container."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # the genrec/ project folder (src layout: genrec/src/genrec/paths.py)
DATA_DIR = Path(os.environ.get("GENREC_DATA_DIR", ROOT / "data"))
ML1M_DIR = DATA_DIR / "ml-1m"
RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = Path(os.environ.get("GENREC_ARTIFACTS_DIR", ROOT / "artifacts"))

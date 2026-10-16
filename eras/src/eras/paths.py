"""Default locations (every function that reads data also takes an explicit path)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]        # the eras/ project folder
DATA_DIR = Path(os.environ.get("ERAS_DATA_DIR", ROOT / "data"))
ARTIFACTS_DIR = Path(os.environ.get("ERAS_ARTIFACTS_DIR", ROOT / "artifacts"))
RESULTS_DIR = ROOT / "results"

import numpy as np
import pandas as pd
import pytest

from eras.data import OUTFIT_CATEGORIES, TTPD_PARTS, Tour


@pytest.fixture(scope="session")
def tour():
    """A small synthetic tour: 30 shows, two cities a leg, outfits rotating through 3 versions, a few songs."""
    rng = np.random.default_rng(0)
    n = 30
    dates = pd.date_range("2024-05-01", periods=n, freq="3D")
    shows = pd.DataFrame({"show": range(n), "date": dates, "city": [f"City {k // 3}" for k in range(n)],
                          "night": [k % 3 + 1 for k in range(n)],
                          "leg": ["Europe 2024" if k < 20 else "North America 2024" for k in range(n)]})
    outfits = {c: [f"V{k % 3 + 1}" for k in range(n)] for c in OUTFIT_CATEGORIES}
    outfits["Reputation outfit"] = ["V1"] * n
    for part in TTPD_PARTS:
        outfits[part] = [f"V{k % 2 + 1}" for k in range(n)]
    outfits["TTPD outfit"] = [f"{d} + {s}" for d, s in zip(*(outfits[p] for p in TTPD_PARTS))]
    songs = [f"Song {i}" for i in range(20)]
    catalogue = pd.DataFrame({"song": songs + ["Shake It Off"], "album": ["1989"] * 10 + ["RED"] * 10 + ["1989"],
                              "main_set": [False] * 20 + [True]})
    guitar = [{songs[int(rng.integers(0, 20))]} for _ in range(n)]
    piano = [{songs[int(rng.integers(0, 20))], songs[int(rng.integers(0, 20))]} for _ in range(n)]
    return Tour(shows, outfits, guitar, piano, catalogue)

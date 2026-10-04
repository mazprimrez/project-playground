"""The lookup API end to end on a small synthetic table."""
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from genrec import lookup  # noqa: E402

MOVIES = pd.DataFrame({"movie_id": [1, 2, 3, 4, 5], "title": ["Heat (1995)", "Fargo (1996)", "Alien (1979)",
                                                             "Big (1988)", "Jaws (1975)"],
                       "genres": ["Action|Crime", "Crime", "Horror", "", "Thriller"], "year": [1995, 1996, 1979, 1988, 1975],
                       "poster_path": ["/heat.jpg", None, None, "/big.jpg", None]})
USERS = pd.DataFrame({"user_id": [10, 20], "history": [[1, 2], [3]], "next_movie_id": [4, 5]})
RECS = pd.concat([
    lookup.to_rows("sasrec", [10, 20], [[(4, 0.5), (3, 0.2)], [(1, 0.3), (5, 0.1)]]),
    lookup.to_rows("qwen", [10, 20], [[(5, 0.4), (3, 0.1)], [(5, 0.6), (2, 0.2)]]),
], ignore_index=True)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    lookup.write(tmp_path, MOVIES, USERS, RECS)
    monkeypatch.setenv("GENREC_LOOKUP_DIR", str(tmp_path))
    from serving.app import app
    with TestClient(app) as c:
        yield c


def test_health_models_users(client):
    assert client.get("/health").json()["models"] == ["sasrec", "qwen"]
    m = {x["model"]: x for x in client.get("/models").json()}
    # user 10 (next = 4): sasrec [4, 3] hit at rank 1, qwen [5, 3] miss; user 20 (next = 5): sasrec [1, 5], qwen [5, 2]
    assert (m["sasrec"]["hr_at_10"], m["sasrec"]["hit_at_1"]) == (1.0, 0.5)
    assert (m["qwen"]["hr_at_10"], m["qwen"]["hit_at_1"]) == (0.5, 0.5)
    assert client.get("/users").json() == [10, 20]


def test_user(client):
    u = client.get("/users/10").json()
    assert [m["title"] for m in u["history"]] == ["Heat (1995)", "Fargo (1996)"]
    assert u["next_movie"]["title"] == "Big (1988)" and u["next_movie"]["genres"] == [] and u["n_ratings"] == 3
    assert client.get("/users/99").status_code == 404


def test_recommendations(client):
    r = client.get("/users/10/recommendations", params={"model": "sasrec"}).json()
    assert r["hit"] and [x["movie"]["movie_id"] for x in r["recommendations"]] == [4, 3]
    assert r["recommendations"][0] == {"rank": 1, "movie": {"movie_id": 4, "title": "Big (1988)", "genres": [],
                                                            "year": 1988, "poster_url": "https://image.tmdb.org/t/p/w342/big.jpg"},
                                         "probability": 0.5, "is_next_movie": True}
    assert len(client.get("/users/20/recommendations", params={"model": "qwen", "k": 1}).json()["recommendations"]) == 1
    assert client.get("/users/10/recommendations", params={"model": "nope"}).status_code == 404
    assert client.get("/users/10/recommendations").json()["model"] == "qwen"    # the default when loaded


def test_compare_and_movie(client):
    c = client.get("/users/20/compare").json()
    assert [x["model"] for x in c] == ["sasrec", "qwen"] and [x["hit"] for x in c] == [True, True]
    assert client.get("/movies/3").json()["genres"] == ["Horror"]
    assert client.get("/movies/3").json()["poster_url"] is None             # no poster found: the UI draws one

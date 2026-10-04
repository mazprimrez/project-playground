"""The gateway end to end: genrec mounted under /genrec on a small synthetic table, and a project that fails."""
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from genrec import lookup  # noqa: E402

MOVIES = pd.DataFrame({"movie_id": [1, 2, 3], "title": ["Heat (1995)", "Fargo (1996)", "Alien (1979)"],
                       "genres": ["Action|Crime", "Crime", "Horror"], "year": [1995, 1996, 1979]})
USERS = pd.DataFrame({"user_id": [10], "history": [[1]], "next_movie_id": [3]})
RECS = lookup.to_rows("sasrec", [10], [[(3, 0.5), (2, 0.2)]])


@pytest.fixture()
def client(monkeypatch):
    def run(lookup_dir):
        monkeypatch.setenv("GENREC_LOOKUP_DIR", str(lookup_dir))
        from app import app
        return TestClient(app)
    return run


def test_genrec_mounted(client, tmp_path):
    lookup.write(tmp_path, MOVIES, USERS, RECS)
    with client(tmp_path) as c:
        assert c.get("/health").json() == {"status": "ok", "projects": {"genrec": "up"}}
        assert c.get("/").json()[0] | {"description": None} == {
            "name": "genrec", "title": "GenRec lookup API", "description": None, "status": "up", "docs": "/genrec/docs"}
        r = c.get("/genrec/users/10/recommendations").json()      # no model given: the best loaded one
        assert r["model"] == "sasrec" and r["hit"]
        assert c.get("/genrec/openapi.json").status_code == 200
        assert "/genrec/openapi.json" in c.get("/genrec/docs").text


def test_failed_project_is_isolated(client, tmp_path):
    with client(tmp_path / "missing") as c:                       # no tables: genrec's startup fails
        h = c.get("/health").json()
        assert h["status"] == "degraded" and h["projects"]["genrec"].startswith("startup failed")
        r = c.get("/genrec/users/10/recommendations")
        assert r.status_code == 503 and "genrec is unavailable" in r.json()["detail"]
        assert c.get("/").json()[0]["status"] == "down"

"""The server end to end: genrec mounted under /api/genrec on a small synthetic table, a project that fails, and the UI."""
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI  # noqa: E402
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
        assert c.get("/api/health").json() == {"status": "ok", "projects": {"genrec": "up"}}
        assert c.get("/api/").json()[0] | {"description": None} == {
            "name": "genrec", "title": "GenRec lookup API", "description": None, "status": "up",
            "docs": "/api/genrec/docs"}
        r = c.get("/api/genrec/users/10/recommendations").json()      # no model given: the best loaded one
        assert r["model"] == "sasrec" and r["hit"]
        assert c.get("/api/genrec/openapi.json").status_code == 200
        assert "/api/genrec/openapi.json" in c.get("/api/genrec/docs").text
        assert c.get("/api/genrec/users/99").status_code == 404       # API errors stay JSON, not the UI page


def test_failed_project_is_isolated(client, tmp_path):
    with client(tmp_path / "missing") as c:                           # no tables: genrec's startup fails
        h = c.get("/api/health").json()
        assert h["status"] == "degraded" and h["projects"]["genrec"].startswith("startup failed")
        r = c.get("/api/genrec/users/10/recommendations")
        assert r.status_code == 503 and "genrec is unavailable" in r.json()["detail"]
        assert c.get("/api/").json()[0]["status"] == "down"


def test_ui_serves_files_and_routes(tmp_path):
    from app import ui_app
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<div id=root></div>")
    (tmp_path / "assets" / "app.js").write_text("console.log(1)")
    ui = FastAPI()
    ui.mount("/", ui_app(tmp_path))
    c = TestClient(ui)
    assert c.get("/assets/app.js").text == "console.log(1)"
    assert c.get("/").text == c.get("/projects/genrec").text == "<div id=root></div>"   # browser routes -> index.html
    assert c.get("/assets/old-build.js").status_code == 404                               # missing files stay 404s

    missing = FastAPI()
    missing.mount("/", ui_app(tmp_path / "nope"))
    assert "npm run build" in TestClient(missing).get("/").json()["detail"]

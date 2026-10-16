"""The API end to end on tables built from the synthetic tour."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from eras import backtest as B, lookup  # noqa: E402


@pytest.fixture()
def client(tour, tmp_path, monkeypatch):
    lookup.write(tmp_path, lookup.build(tour, B.run(tour, warmup=10, prophet=False)))
    monkeypatch.setenv("ERAS_LOOKUP_DIR", str(tmp_path))
    from serving.app import app
    with TestClient(app) as c:
        yield c


def test_summary(client):
    s = client.get("/summary").json()
    assert s["shows"] == 20 and s["max_points"] == 56
    names = [e["entry"] for e in s["entries"]]
    assert "choice" in names and "prophet" not in names          # this backtest ran without prophet
    pts = [e["points_per_show"] for e in s["entries"]]
    assert pts == sorted(pts, reverse=True)


def test_shows_and_detail(client):
    shows = client.get("/shows").json()
    assert [s["show"] for s in shows] == list(range(10, 30))
    d = client.get("/shows/12").json()
    assert d["city"] == "City 4" and d["night"] == 1
    cats = {c["category"]: c for c in d["categories"]}
    assert len(cats) == 15 and cats["Surprise song (piano)"]["max_points"] == 2
    rep = cats["Reputation outfit"]
    assert rep["actual"] == "V1" and rep["picks"]["choice"] == {
        "pick": "V1", "probability": rep["picks"]["choice"]["probability"], "correct": True, "points": 4.0}
    total = sum(c["picks"]["choice"]["points"] for c in d["categories"])
    assert total == d["points"]["choice"]
    assert client.get("/shows/3").status_code == 404                 # warm-up shows aren't replayed

from fastapi.testclient import TestClient

from tinytalk import server


class FakeEngine:
    def __init__(self, loaded):
        self.loaded = loaded

    def load(self):
        self.loaded = True


def test_health_ok_after_loaded(monkeypatch):
    monkeypatch.setattr(server, "engine", FakeEngine(True))
    with TestClient(server.app) as client:
        res = client.get("/health")
    assert res.status_code == 200


def test_health_503_when_not_loaded(monkeypatch):
    engine = FakeEngine(False)
    monkeypatch.setattr(server, "engine", engine)
    monkeypatch.setattr(engine, "load", lambda: None)
    with TestClient(server.app) as client:
        engine.loaded = False
        res = client.get("/health")
    assert res.status_code == 503

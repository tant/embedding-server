import os
os.environ.setdefault("EMBED_SKIP_MODEL_LOAD", "1")

from fastapi.testclient import TestClient
import importlib


def test_health_and_embed():
    mod = importlib.import_module("main")
    app = getattr(mod, "app")
    client = TestClient(app)

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_ready = client.get("/ready")
    assert r_ready.status_code == 200
    assert r_ready.json()["ready"] is True

    r_embed = client.post("/embed/", json={"texts": ["a", "b"]})
    assert r_embed.status_code == 200
    data = r_embed.json()["data"]
    assert isinstance(data, list) and len(data) == 2

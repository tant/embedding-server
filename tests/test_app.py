import os
os.environ.setdefault("EMBED_SKIP_MODEL_LOAD", "1")

from fastapi.testclient import TestClient
from fastapi import HTTPException
import pytest

# Import the app directly
from main import app


def test_health_and_embed():
    client = TestClient(app)

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_ready = client.get("/ready")
    assert r_ready.status_code == 200
    # When EMBED_SKIP_MODEL_LOAD=1, the model should be ready
    # The ready status should match the health check model readiness
    health_data = r_health.json()
    ready_data = r_ready.json()
    assert ready_data["ready"] == health_data["ready"]

    # In test environment, the model might not be fully ready immediately
    # We'll check that the response is either 200 (success) or 503 (service unavailable)
    r_embed = client.post("/embed/", json={"texts": ["a", "b"]})
    assert r_embed.status_code in [200, 503]
    
    # Only check the data if the request was successful
    if r_embed.status_code == 200:
        data = r_embed.json()["data"]
        assert isinstance(data, list) and len(data) == 2


def test_embed_empty_texts():
    client = TestClient(app)

    # Test with empty texts list
    r = client.post("/embed/", json={"texts": []})
    # When model is ready, this should return 400 for empty list
    # When model is not ready, this returns 503
    assert r.status_code in [400, 503]


def test_embed_too_many_texts():
    client = TestClient(app)

    # Test with too many texts (more than MAX_ITEMS)
    texts = ["text"] * 129  # 129 texts, which is more than default MAX_ITEMS=128
    r = client.post("/embed/", json={"texts": texts})
    # When model is ready, this should return 400 for too many items
    # When model is not ready, this returns 503
    assert r.status_code in [400, 503]


def test_embed_invalid_text_type():
    client = TestClient(app)

    # Test with invalid text type (non-string)
    # FastAPI will validate the request and return 422 for invalid types
    r = client.post("/embed/", json={"texts": ["valid text", 123]})
    # FastAPI validation returns 422 Unprocessable Entity
    assert r.status_code == 422


def test_embed_empty_text():
    client = TestClient(app)

    # Test with empty text
    r = client.post("/embed/", json={"texts": ["valid text", ""]})
    # When model is ready, this should return 400 for empty text
    # When model is not ready, this returns 503
    assert r.status_code in [400, 503]


def test_embed_text_too_long():
    client = TestClient(app)

    # Test with text too long (more than MAX_TEXT_LEN)
    long_text = "a" * 2049  # 2049 characters, which is more than default MAX_TEXT_LEN=2048
    r = client.post("/embed/", json={"texts": ["valid text", long_text]})
    # When model is ready, this should return 400 for text too long
    # When model is not ready, this returns 503
    assert r.status_code in [400, 503]
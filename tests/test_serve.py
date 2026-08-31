import pytest
from fastapi.testclient import TestClient

from src.serve import app


@pytest.fixture
def client(isolated_dirs, monkeypatch):
    """TestClient with an isolated models/results dir (no trained model
    present) and synthetic data, auto-refresh disabled so the background
    scheduler doesn't fire mid-test."""
    monkeypatch.setenv("FRAUDMESH_SYNTHETIC", "1")
    monkeypatch.setenv("FRAUDMESH_AUTO_REFRESH", "false")
    with TestClient(app) as c:
        yield c


def test_health_after_startup_refresh(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["cache_populated"] is True
    assert body["model_trained"] is False  # no models/graphsage.pt in isolated dir
    assert body["n_cached_transactions"] > 0


def test_score_known_transaction_returns_untrained_warning(client):
    resp = client.post("/score", json={"transaction_id": 0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["transaction_id"] == 0
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["note"] is not None and "untrained" in body["note"]


def test_score_unknown_transaction_returns_404(client):
    resp = client.post("/score", json={"transaction_id": 10**9})
    assert resp.status_code == 404


def test_manual_refresh_repopulates_cache(client):
    resp = client.post("/refresh")
    assert resp.status_code == 200
    assert resp.json()["n_transactions"] > 0


def test_api_key_required_when_configured(isolated_dirs, monkeypatch):
    monkeypatch.setenv("FRAUDMESH_SYNTHETIC", "1")
    monkeypatch.setenv("FRAUDMESH_AUTO_REFRESH", "false")
    monkeypatch.setenv("FRAUDMESH_API_KEY", "test-secret")
    with TestClient(app) as c:
        # No key -> rejected
        resp = c.post("/score", json={"transaction_id": 0})
        assert resp.status_code == 401

        # Wrong key -> rejected
        resp = c.post("/score", json={"transaction_id": 0}, headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401

        # Correct key -> allowed
        resp = c.post("/score", json={"transaction_id": 0},
                       headers={"X-API-Key": "test-secret"})
        assert resp.status_code == 200

        # /health has no auth requirement
        assert c.get("/health").status_code == 200

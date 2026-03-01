"""Test API endpoints."""

import pytest
from starlette.testclient import TestClient

from stream.app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_home_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "STREAM" in resp.text


def test_illicit_page(client):
    resp = client.get("/illicit")
    assert resp.status_code == 200
    assert "ILLICIT" in resp.text


def test_fees_page(client):
    resp = client.get("/fees")
    assert resp.status_code == 200
    assert "FEE" in resp.text


def test_lightning_page(client):
    resp = client.get("/lightning")
    assert resp.status_code == 200
    assert "LIGHTNING" in resp.text


def test_onboarding_page(client):
    resp = client.get("/onboarding")
    assert resp.status_code == 200
    assert "ONBOARDING" in resp.text


def test_walkthrough_page(client):
    resp = client.get("/walkthrough")
    assert resp.status_code == 200
    assert "WALKTHROUGH" in resp.text


def test_models_page_removed(client):
    resp = client.get("/models")
    assert resp.status_code == 404


def test_pipeline_page(client):
    resp = client.get("/pipeline")
    assert resp.status_code == 200
    assert "PIPELINE" in resp.text


def test_random_sample(client):
    resp = client.get("/api/v1/illicit/random-sample")
    assert resp.status_code == 200
    # Should return comma-separated numbers
    text = resp.text
    parts = text.split(",")
    assert len(parts) == 166


def test_illicit_metrics(client):
    resp = client.get("/api/v1/illicit/metrics")
    assert resp.status_code == 200
    assert "PR-AUC" in resp.text


def test_onboarding_metrics(client):
    resp = client.get("/api/v1/onboarding/metrics")
    assert resp.status_code == 200
    assert "Logistic" in resp.text


def test_recent_alerts(client):
    resp = client.get("/api/v1/alerts/recent")
    assert resp.status_code == 200


def test_pipeline_recent_runs(client):
    resp = client.get("/api/v1/pipeline/recent-runs")
    assert resp.status_code == 200


def test_pipeline_freshness(client):
    resp = client.get("/api/v1/pipeline/freshness")
    assert resp.status_code == 200
    assert "FRESH" in resp.text

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_expected_response():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "SpecSentry API"}


@pytest.mark.parametrize("origin", ["http://localhost:5173", "http://127.0.0.1:5173"])
def test_health_allows_local_frontend_origins(origin):
    response = client.get("/api/health", headers={"Origin": origin})

    assert response.headers["access-control-allow-origin"] == origin


def test_health_omits_cors_header_for_other_origins():
    response = client.get("/api/health", headers={"Origin": "https://example.com"})

    assert "access-control-allow-origin" not in response.headers

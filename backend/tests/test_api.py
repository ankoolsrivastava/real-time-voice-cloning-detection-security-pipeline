from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info():
    response = client.get("/api/model")

    assert response.status_code == 200
    data = response.json()

    assert data["model_version"] == "voiceguard_v2_epoch8"
    assert data["status"] == "loaded"


def test_create_and_get_session():
    response = client.post("/api/sessions")

    assert response.status_code == 200

    data = response.json()
    session_id = data["session_id"]

    assert data["status"] == "active"

    response = client.get(f"/api/sessions/{session_id}")

    assert response.status_code == 200
    assert response.json()["session_id"] == session_id


def test_missing_session_returns_404():
    response = client.get("/api/sessions/does-not-exist")

    assert response.status_code == 404

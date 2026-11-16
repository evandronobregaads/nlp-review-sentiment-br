from fastapi.testclient import TestClient
from src.api import app, get_model

def test_health_no_artifact(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "not_found.joblib"))
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_available"] is False

def test_untrained_predict_returns_503(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "not_found.joblib"))
    get_model.cache_clear()
    try:
        response = TestClient(app).post("/predict", json={"text": "Entrega excelente"})
        assert response.status_code == 503
    finally:
        get_model.cache_clear()

def test_validates_empty_input():
    response = TestClient(app).post("/predict", json={"text": ""})
    assert response.status_code == 422

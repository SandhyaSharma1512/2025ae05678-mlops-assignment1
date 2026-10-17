from fastapi.testclient import TestClient
from heart_disease.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_predict_rejects_invalid_payload():
    response = client.post("/predict", json={"age": -1})
    assert response.status_code == 422

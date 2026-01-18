import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_single_sample_returns_expected_shape(client):
    payload = {"x": [0.1, 0.2, 0.3, 0.4]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text

    data = r.json()
    assert data["batch_size"] == 1
    assert data["t"] == 4
    assert isinstance(data["y"], list)
    assert len(data["y"]) == 4


def test_predict_batch_returns_batch_output(client):
    payload = {
        "x": [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
        ]
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text

    data = r.json()
    assert data["batch_size"] == 2
    assert data["t"] == 4
    assert len(data["y"]) == 2
    assert all(len(row) == 4 for row in data["y"])


def test_predict_rejects_empty_input(client):
    r = client.post("/predict", json={"x": []})
    assert r.status_code == 400


def test_predict_onnx(client):
    payload = {"x": [0.1] * 1024}
    r = client.post("/predict/onnx", json=payload)

    # ONNX may be unavailable locally; accept skip via 503
    if r.status_code == 503:
        pytest.skip("ONNX model not loaded")

    assert r.status_code == 200
    data = r.json()
    assert data["batch_size"] == 1
    assert data["t"] == 1024

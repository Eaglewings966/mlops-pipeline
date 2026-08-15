"""Tests for the FastAPI serving API."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.95, 0.05]])
    return model


@pytest.fixture
def test_client(mock_model):
    with patch("src.serving.api.model", mock_model), \
         patch("src.serving.api.model_version", "1"), \
         patch("src.serving.api.feature_columns", ["TransactionAmt", "C1", "V1"]):
        from fastapi.testclient import TestClient
        from src.serving.api import app
        yield TestClient(app)


def test_health_check(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_name" in data
    assert "timestamp" in data


def test_predict_returns_correct_schema(test_client):
    response = test_client.post("/predict", json={"TransactionAmt": 150.0})
    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert "is_fraud" in data
    assert "fraud_probability" in data
    assert "risk_level" in data
    assert "latency_ms" in data
    assert 0.0 <= data["fraud_probability"] <= 1.0


def test_predict_rejects_negative_amount(test_client):
    response = test_client.post("/predict", json={"TransactionAmt": -100.0})
    assert response.status_code == 422


def test_predict_batch_returns_correct_count(test_client):
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([
        [0.9, 0.1], [0.7, 0.3], [0.4, 0.6]
    ])
    with patch("src.serving.api.model", mock_model):
        response = test_client.post("/predict/batch", json={
            "transactions": [
                {"TransactionAmt": 100.0},
                {"TransactionAmt": 250.0},
                {"TransactionAmt": 50.0},
            ]
        })
    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] == 3
    assert len(data["predictions"]) == 3


def test_risk_level_assignment():
    from src.serving.api import get_risk_level

    assert get_risk_level(0.9) == "CRITICAL"
    assert get_risk_level(0.6) == "HIGH"
    assert get_risk_level(0.3) == "MEDIUM"
    assert get_risk_level(0.1) == "LOW"


def test_model_info_endpoint(test_client):
    response = test_client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "feature_count" in data

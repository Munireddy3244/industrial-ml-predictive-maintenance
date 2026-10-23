from pathlib import Path
import sys
import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_ml.api import app

class FakeModel:
    classes_ = np.array([0, 1])
    def predict_proba(self, X):
        return np.array([[0.65, 0.35]])

def test_health_and_predict():
    # Override the lifespan-loaded model with a deterministic test double.
    with TestClient(app) as client:
        app.state.model = FakeModel()
        app.state.metadata = {"model_version": "test", "threshold": 0.5}
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True

        response = client.post("/predict", json={
            "equipment_id": "TEST-2",
            "temperature_c": 80,
            "vibration_mm_s": 4,
            "speed_rpm": 1500,
            "operating_hours": 10000
        })
        assert response.status_code == 200
        assert response.json()["predicted_failure"] is False
        assert response.json()["model_version"] == "test"

def test_invalid_input_rejected():
    with TestClient(app) as client:
        app.state.model = FakeModel()
        app.state.metadata = {"model_version": "test", "threshold": 0.5}
        response = client.post("/predict", json={
            "equipment_id": "TEST-3",
            "temperature_c": 999,
            "vibration_mm_s": 4,
            "speed_rpm": 1500,
            "operating_hours": 10000
        })
        assert response.status_code == 422

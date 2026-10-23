from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from industrial_ml.model_service import predict_one, risk_band

class FakeModel:
    classes_ = np.array([0, 1])
    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])

def test_predict_one_returns_expected_shape():
    reading = {
        "equipment_id": "TEST-1",
        "temperature_c": 80,
        "vibration_mm_s": 4.2,
        "speed_rpm": 1500,
        "operating_hours": 12000,
    }
    result = predict_one(FakeModel(), reading, threshold=0.5)
    assert result["predicted_failure"] is True
    assert result["failure_probability"] == 0.8
    assert result["risk_band"] == "high"
    assert result["equipment_id"] == "TEST-1"

def test_risk_band_boundaries():
    assert risk_band(0.1) == "low"
    assert risk_band(0.5) == "moderate"
    assert risk_band(0.8) == "high"

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .settings import MODEL_PATH, METADATA_PATH, DEFAULT_THRESHOLD, MODEL_VERSION

logger = logging.getLogger(__name__)

FEATURES = ["temperature_c", "vibration_mm_s", "speed_rpm", "operating_hours"]

class ModelNotReadyError(RuntimeError):
    """Raised when the trained model artifact is missing or invalid."""

def load_model(model_path: Path = MODEL_PATH) -> Any:
    if not model_path.exists():
        raise ModelNotReadyError(
            f"Model not found at {model_path}. Run: python scripts/train.py"
        )
    # Only load joblib artifacts produced and controlled by this project.
    return joblib.load(model_path)

def load_metadata(metadata_path: Path = METADATA_PATH) -> dict:
    if metadata_path.exists():
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception("Could not read model metadata; using defaults.")
    return {"model_version": MODEL_VERSION, "threshold": DEFAULT_THRESHOLD}

def risk_band(probability: float) -> str:
    if probability >= 0.70:
        return "high"
    if probability >= 0.35:
        return "moderate"
    return "low"

def predict_one(model: Any, reading: dict, threshold: float) -> dict:
    row = pd.DataFrame([{
        "temperature_c": reading["temperature_c"],
        "vibration_mm_s": reading["vibration_mm_s"],
        "speed_rpm": reading["speed_rpm"],
        "operating_hours": reading["operating_hours"],
    }], columns=FEATURES)

    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)
    if 1 not in classes:
        raise ModelNotReadyError("Model does not contain failure class 1.")
    probability = float(probabilities[classes.index(1)])
    return {
        "equipment_id": reading["equipment_id"],
        "predicted_failure": probability >= threshold,
        "failure_probability": round(probability, 6),
        "risk_band": risk_band(probability),
        "threshold": threshold,
        "model_version": MODEL_VERSION,
        "disclaimer": (
            "Demonstration model trained on synthetic data. Not for real "
            "maintenance or safety decisions."
        ),
    }

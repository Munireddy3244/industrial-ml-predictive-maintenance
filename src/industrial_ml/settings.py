from pathlib import Path
import os

# Resolve project root from this file: <project>/src/industrial_ml/settings.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "generated"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"
DATASET_PATH = DATA_DIR / "industrial_sensor_data.csv"
MODEL_PATH = ARTIFACTS_DIR / "failure_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

DEFAULT_THRESHOLD = float(os.getenv("FAILURE_THRESHOLD", "0.50"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "demo-1.0.0")

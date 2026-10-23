from pathlib import Path
import sys
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split

from industrial_ml.settings import (
    DATASET_PATH, ARTIFACTS_DIR, MODEL_PATH, METADATA_PATH,
    DEFAULT_THRESHOLD, MODEL_VERSION
)

FEATURES = [
    "temperature_c",
    "vibration_mm_s",
    "speed_rpm",
    "operating_hours"
]
TARGET = "failure"

def main():
    if not DATASET_PATH.exists():
        raise SystemExit(
            f"Dataset not found: {DATASET_PATH}\n"
            "Run python scripts/generate_data.py first."
        )

    df = pd.read_csv(DATASET_PATH)
    required = FEATURES + [TARGET]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise SystemExit(f"Dataset is missing columns: {missing}")
    if df[required].isna().any().any():
        raise SystemExit("Missing values found. Add a documented imputation strategy.")
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise SystemExit("Target column 'failure' must contain only 0 and 1.")

    X = df[FEATURES]
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    probability = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    predicted = (probability >= DEFAULT_THRESHOLD).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probability)) if y_test.nunique() == 2 else None,
        "average_precision": float(average_precision_score(y_test, probability)),
        "threshold": DEFAULT_THRESHOLD,
        "confusion_matrix_labels_0_1": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, predicted, zero_division=0, output_dict=True
        ),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metadata = {
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "RandomForestClassifier",
        "features": FEATURES,
        "target": TARGET,
        "threshold": DEFAULT_THRESHOLD,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "dataset": DATASET_PATH.name,
        "metrics": metrics,
        "warning": "Synthetic dataset; not suitable for real operational decisions.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (ARTIFACTS_DIR / "evaluation_report.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(f"Model saved: {MODEL_PATH}")
    print(f"Metadata saved: {METADATA_PATH}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print("\nClassification report:")
    print(classification_report(y_test, predicted, zero_division=0))
    print("\nNOTE: evaluation is on synthetic data, not evidence of real-world performance.")

if __name__ == "__main__":
    main()

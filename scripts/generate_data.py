from pathlib import Path
import sys

# Permit running this script directly from the project root without installation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from industrial_ml.settings import DATA_DIR, DATASET_PATH

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    n = 5000

    # Synthetic sensor readings with broad, plausible-looking ranges.
    temperature = np.clip(rng.normal(68, 13, n), 25, 125)
    vibration = np.clip(rng.lognormal(mean=0.85, sigma=0.65, size=n), 0.05, 18)
    speed = np.clip(rng.normal(1500, 220, n), 500, 3000)
    hours = rng.uniform(100, 30000, n)
    load_pct = np.clip(rng.normal(68, 18, n), 10, 100)

    # Create a noisy synthetic risk relationship, not real failure physics.
    logit = (
        -5.0
        + 0.055 * (temperature - 65)
        + 0.38 * (vibration - 2.5)
        + 0.00006 * (hours - 10000)
        + 0.018 * (load_pct - 65)
        + 0.0005 * (speed - 1500)
    )
    probability = 1 / (1 + np.exp(-np.clip(logit, -20, 20)))
    failure = rng.binomial(1, probability)

    df = pd.DataFrame({
        "equipment_id": [f"EQ-{i:05d}" for i in range(n)],
        "temperature_c": temperature.round(2),
        "vibration_mm_s": vibration.round(3),
        "speed_rpm": speed.round(1),
        "operating_hours": hours.round(1),
        "load_pct": load_pct.round(1),
        "failure": failure.astype(int),
    })
    df.to_csv(DATASET_PATH, index=False)
    print(f"Saved {len(df)} rows to: {DATASET_PATH}")
    print("Synthetic positive rate:", round(float(df["failure"].mean()), 3))
    print("Reminder: synthetic data is for software demonstration only.")

if __name__ == "__main__":
    main()

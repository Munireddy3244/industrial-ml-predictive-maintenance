# Industrial ML Predictive Maintenance (Windows, no Docker)

A learning/reference project demonstrating an end-to-end ML lifecycle:
synthetic sensor data -> training/evaluation -> saved model -> FastAPI -> Streamlit -> tests.

**Important:** the included data is synthetic and the model is a demonstration only. Do not use it to make real maintenance, safety, warranty, or shutdown decisions. A real deployment needs validated historical data, domain review, calibration, monitoring, security review, and a controlled rollout.

## Architecture

- `scripts/generate_data.py`: creates a reproducible synthetic dataset.
- `scripts/train.py`: trains and evaluates a Random Forest model, writes model + metadata.
- `src/industrial_ml/`: shared schemas, model loading, prediction and API.
- `app/streamlit_app.py`: web UI calling the API.
- `tests/`: API and prediction tests.

## Requirements

- Windows 10/11, 64-bit
- Python 3.11 or 3.12 recommended
- PowerShell or Command Prompt
- Internet access only for initial package installation

## Install (PowerShell)

```powershell
cd C:\IndustrialML
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

If PowerShell blocks activation, use Command Prompt:
```bat
.venv\Scripts\activate.bat
```
Or skip activation and call `.venv\Scripts\python.exe` explicitly.

## Run from project root

1. Generate data:
```powershell
python scripts\generate_data.py
```

2. Train:
```powershell
python scripts\train.py
```

3. Run tests:
```powershell
pytest -q
```

4. Start the API in Terminal 1:
```powershell
uvicorn industrial_ml.api:app --host 127.0.0.1 --port 8000
```
Open `http://127.0.0.1:8000/docs` to inspect the API.

5. Start the UI in Terminal 2:
```powershell
streamlit run app\streamlit_app.py
```

The UI normally opens at `http://localhost:8501`. The API should remain running.

## API example

```powershell
$body = @{
  equipment_id = "PUMP-001"
  temperature_c = 92.0
  vibration_mm_s = 6.2
  speed_rpm = 1750
  operating_hours = 16000
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/predict" `
  -ContentType "application/json" `
  -Body $body
```

## What the outputs mean

- `failure_probability`: the model's estimated probability for the synthetic label, not a guaranteed real-world probability.
- `predicted_failure`: the selected threshold decision.
- `risk_band`: a simple UI category based on configured probability cutoffs, not a certified safety classification.
- `model_version`: metadata version used for traceability.

The threshold is a configurable demo setting. Select a real threshold based on the costs of false negatives and false positives, operational capacity, and validation results—not by arbitrary convention.

## Real-world checklist before production

1. Replace synthetic data with approved historical sensor data and confirmed failure labels.
2. Prevent leakage: split by time, equipment, site, or production batch where appropriate.
3. Evaluate recall, precision, PR-AUC, calibration, and false-negative costs on a representative holdout.
4. Review feature units, sensor quality, missing data, out-of-range values and drift with equipment specialists.
5. Add authentication/authorization, TLS, rate limits, secrets management, structured logs, and access auditing.
6. Deploy behind a reverse proxy or managed service; do not expose the development server directly to the internet.
7. Run in shadow mode first; human maintenance staff should review alerts.
8. Establish rollback, retraining approval, model registry, and incident response processes.

## Security notes

- API binds to loopback (`127.0.0.1`) by default.
- Do not expose `/docs` or the API publicly without authentication and a security review.
- Do not load joblib/pickle model files from untrusted sources; they can execute code during deserialization.
- Do not put secrets in source control.
- Avoid sending personal or sensitive information in logs.

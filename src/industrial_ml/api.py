import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .model_service import (
    load_model, load_metadata, predict_one, ModelNotReadyError
)
from .schemas import SensorReading, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("industrial_ml.api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model = load_model()
        app.state.metadata = load_metadata()
        logger.info("Model loaded successfully.")
    except ModelNotReadyError as exc:
        app.state.model = None
        app.state.metadata = load_metadata()
        logger.error("%s", exc)
    yield

app = FastAPI(
    title="Industrial Failure Risk API",
    version="1.0.0",
    description="Educational predictive-maintenance API. Synthetic-data demo only.",
    lifespan=lifespan,
)

@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error for %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

@app.get("/health")
def health(request: Request):
    ready = request.app.state.model is not None
    return {
        "status": "ready" if ready else "model_not_loaded",
        "model_loaded": ready,
        "model_version": request.app.state.metadata.get("model_version", "unknown"),
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading, request: Request):
    model = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Run scripts/train.py and restart the API."
        )
    metadata = request.app.state.metadata
    threshold = float(metadata.get("threshold", 0.50))
    try:
        result = predict_one(model, reading.model_dump(), threshold)
        result["model_version"] = metadata.get("model_version", "unknown")
        return result
    except ModelNotReadyError as exc:
        logger.exception("Prediction failed due to model configuration.")
        raise HTTPException(status_code=503, detail="Model is not ready.") from exc

"""FastAPI model-serving API with structured logs and Prometheus metrics."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, make_asgi_app

from heart_disease.predict import load_model, predict_one, MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("heart_disease_api")

REQUEST_COUNT = Counter("prediction_requests_total", "Number of prediction requests", ["endpoint", "status"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction request latency in seconds")

app = FastAPI(title="Heart Disease Risk Classifier", version="1.0.0",
              description="Educational demo only; not for clinical use.")
app.mount("/metrics", make_asgi_app())


class PatientFeatures(BaseModel):
    age: float = Field(..., ge=0, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: float = Field(..., ge=0, le=300)
    chol: float = Field(..., ge=0, le=800)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=0, le=300)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=-10, le=20)
    slope: int = Field(..., ge=0, le=2)
    ca: float = Field(..., ge=0, le=4)
    thal: float = Field(..., ge=0, le=7)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict")
def predict(features: PatientFeatures):
    start = time.perf_counter()
    try:
        model = load_model()
        result = predict_one(features.model_dump(), model=model)
        REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()
        logger.info("prediction_success prediction=%s confidence=%.4f", result["prediction"], result["confidence"])
        return {
            **result,
            "label": "disease_present" if result["prediction"] == 1 else "disease_absent",
            "disclaimer": "Educational demonstration only; not medical advice.",
        }
    except FileNotFoundError as exc:
        REQUEST_COUNT.labels(endpoint="/predict", status="model_missing").inc()
        logger.exception("model_artifact_missing")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        logger.exception("prediction_failed")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
    finally:
        PREDICTION_LATENCY.observe(time.perf_counter() - start)

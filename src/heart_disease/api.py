from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from fastapi import Request


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predict heart disease risk using machine learning",
    version="1.0.0"
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MODEL_PATH = Path("models/final_model.pkl")
model = joblib.load(MODEL_PATH)

class PatientData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(patient: PatientData):
    try:
        logger.info(f"Prediction request received: {patient.dict()}")
        df = pd.DataFrame([patient.model_dump()])
        prediction = int(model.predict(df)[0])
        confidence = (
            float(model.predict_proba(df).max())
            if hasattr(model, "predict_proba")
            else 1.0
        )
        logger.info(
            f"Prediction={prediction}, Confidence={confidence:.4f}"
        )
        return {
            "prediction": prediction,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(
        f"Request: {request.method} {request.url.path}"
    )

    response = await call_next(request)

    logger.info(
        f"Response Status: {response.status_code}"
    )

    return response
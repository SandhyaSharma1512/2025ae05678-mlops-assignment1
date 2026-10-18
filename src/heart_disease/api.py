from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0"
)

MODEL_PATH = Path("models/final_model.pkl")
PIPELINE_PATH = Path("models/preprocessing_pipeline.pkl")

model = joblib.load(MODEL_PATH)
preprocessing_pipeline = joblib.load(PIPELINE_PATH)


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
        df = pd.DataFrame([patient.model_dump()])
        prediction = int(model.predict(df)[0])
        confidence = (
            float(model.predict_proba(df).max())
            if hasattr(model, "predict_proba")
            else 1.0
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
"""Reusable prediction utilities."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "model.joblib"


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Model artifact missing at {path}. Train the model first.")
    return joblib.load(path)


def predict_one(payload: dict, model=None) -> dict:
    model = model or load_model()
    row = pd.DataFrame([payload])
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])
    return {"prediction": prediction, "confidence": probability}

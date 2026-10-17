import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from heart_disease.predict import predict_one


class TinyModel:
    def predict(self, X):
        return np.array([1])

    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])


def test_predict_one_returns_prediction_and_confidence():
    result = predict_one({"age": 50}, model=TinyModel())
    assert result["prediction"] == 1
    assert result["confidence"] == 0.8

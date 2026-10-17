"""Train, compare, evaluate, and log classification pipelines with MLflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "reports" / "model_metrics.json"
RANDOM_STATE = 42


def build_pipeline(X: pd.DataFrame, estimator) -> Pipeline:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def train(data_path: Path, experiment_name: str = "heart-disease-classification") -> dict:
    df = pd.read_csv(data_path)
    if "target" not in df:
        raise ValueError("Dataset must contain a binary 'target' column.")
    df = df.dropna(subset=["target"])
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)
    if y.nunique() != 2:
        raise ValueError(f"Expected binary target; found classes {sorted(y.unique().tolist())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "roc_auc": "roc_auc"}
    mlflow.set_experiment(experiment_name)
    results = []
    best_name, best_pipeline, best_auc = None, None, -1.0

    for name, estimator in candidates.items():
        pipeline = build_pipeline(X_train, estimator)
        with mlflow.start_run(run_name=name):
            cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, error_score="raise")
            cv_metrics = {f"cv_{metric}_mean": float(values.mean()) for metric, values in cv_results.items() if metric.startswith("test_")}
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            metrics = {
                "test_accuracy": float(accuracy_score(y_test, predictions)),
                "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
                "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
                "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
                **cv_metrics,
            }
            mlflow.log_params({"model": name, "random_state": RANDOM_STATE, "cv_folds": 5})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, artifact_path="pipeline")
            results.append({"model": name, **metrics})
            if metrics["cv_roc_auc_mean"] > best_auc:
                best_auc, best_name, best_pipeline = metrics["cv_roc_auc_mean"], name, pipeline

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    summary = {"selected_model": best_name, "selection_metric": "mean cross-validation ROC-AUC", "results": results}
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    with mlflow.start_run(run_name="selected_model_artifact"):
        mlflow.log_param("selected_model", best_name)
        mlflow.log_metric("best_cv_roc_auc", float(best_auc))
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(METRICS_PATH))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "heart_disease.csv")
    parser.add_argument("--experiment", default="heart-disease-classification")
    args = parser.parse_args()
    train(args.data, args.experiment)

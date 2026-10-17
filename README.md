# MLOps Assignment 01 — Heart Disease Classification

This is a runnable project scaffold for the **MLOps AIMLCZG523 Assignment 01**. It covers data acquisition/EDA, feature engineering and model comparison, MLflow experiment tracking, model packaging, tests, GitHub Actions CI, FastAPI + Docker serving, Kubernetes manifests, and monitoring/logging.

> **Important:** This project is a learning/demo classifier, not a medical device. Do not use predictions for clinical decisions. Review and personalize the analysis, model choices, screenshots, report, and deployment before submission. The assignment requires you to understand and demonstrate your own work.

## Project structure

```text
mlops_heart_disease_project/
├── data/                       # downloaded/generated CSVs (not committed by default)
├── models/                     # trained model artifact
├── reports/                    # EDA plots and final report starter
├── screenshots/                # add your own Docker/cloud/Kubernetes screenshots
├── notebooks/                 # EDA, training/MLflow, inference notebooks
├── src/heart_disease/
│   ├── __init__.py
│   ├── data.py                 # download, clean, split
│   ├── eda.py                  # plots and summary
│   ├── train.py                # CV, model comparison, MLflow logging
│   ├── predict.py              # reusable inference function
│   └── api.py                  # FastAPI /health and /predict
├── tests/
├── .github/workflows/ci.yml
├── k8s/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── report_template.md
```

## Jupyter notebooks

The `notebooks/` directory contains:
- `01_eda.ipynb`: data acquisition, cleaning checks, EDA visualizations.
- `02_model_training_mlflow.ipynb`: preprocessing, model comparison, evaluation and MLflow.
- `03_inference_api.ipynb`: saved-model inference and an optional API smoke test.

Install the requirements, then launch Jupyter from the project root:
```bash
pip install jupyter
jupyter notebook
```
Run notebooks in order. Notebook 2 saves the model artifact used by notebook 3.

## 1. Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 2. Download and prepare the dataset

The download script uses the `ucimlrepo` package to retrieve UCI Heart Disease dataset (ID 45).

```bash
python -m heart_disease.data --download
```

It writes `data/raw/heart_disease.csv` and a normalized `data/processed/heart_disease.csv`. The target is binarized as `0 = no disease`, `1 = disease present` (original target values greater than 0 map to 1). Check the downloaded schema and document any changes in your report.

If UCI access is unavailable, download the Heart Disease dataset from the UCI Machine Learning Repository manually and place the source CSV at `data/raw/heart_disease.csv`; adapt the loader if its column names differ.

## 3. EDA

```bash
python -m heart_disease.eda
```

Plots and summary statistics are written to `reports/eda/`. Include the generated plots and your interpretation in the report.

## 4. Train models and track experiments

```bash
python -m heart_disease.train --data data/processed/heart_disease.csv --experiment heart-disease-classification
```

The script compares Logistic Regression and Random Forest using stratified cross-validation, logs parameters/metrics/artifacts with MLflow, and saves the best pipeline to `models/model.joblib`. The default selection metric is ROC-AUC; explain whether this is appropriate for your use case. The assignment asks you to report accuracy, precision, recall and ROC-AUC.

View MLflow runs:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open the URL printed by MLflow (usually `http://127.0.0.1:5000`).

## 5. Run API locally

Train the model first, then:

```bash
uvicorn heart_disease.api:app --host 0.0.0.0 --port 8000
```

Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

Response includes a binary prediction and probability estimate. The API intentionally refuses to start if the trained artifact is missing; this avoids silently serving an untrained model.

## 6. Docker

Build from the project root after training:

```bash
docker build -t heart-disease-api .
docker run --rm -p 8000:8000 heart-disease-api
```

Test `/health` and `/predict` as above. `docker-compose.yml` is also provided. Make sure the model artifact exists before building so it can be copied into the image.

## 7. Tests and CI

```bash
pytest -q
```

The GitHub Actions workflow installs the package, runs lint/tests, downloads data and trains a model. CI fails when commands fail. Review workflow logs and adjust caching/runtime as needed.

## 8. Kubernetes (local Minikube / Docker Desktop Kubernetes)

Build the image in the environment your cluster uses, then apply:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods,svc
```

The Service is `NodePort` for local learning. For a cloud cluster, use the cloud provider's LoadBalancer/Ingress setup and document the actual URL. Kubernetes image availability differs between Minikube, Docker Desktop, and cloud clusters; follow your cluster's image-loading workflow.

## 9. Monitoring and logging

The API logs requests and prediction latency. `/metrics` exposes Prometheus metrics. A sample Prometheus scrape configuration is in `monitoring/prometheus.yml`. Add Prometheus/Grafana screenshots from your own running environment to `screenshots/`.

## 10. Submission checklist

- [ ] Personalize the EDA and model-selection discussion; record actual results.
- [ ] Capture screenshots from your own CI run, Docker container, API, and deployment.
- [ ] Fill out `report_template.md` and create the required ~10-page PDF/DOCX report.
- [ ] Add your own architecture diagram and pipeline/video recording.
- [ ] Add repository URL and deployed API URL, if public.
- [ ] Verify clean setup by following this README from a fresh environment.

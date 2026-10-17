# MLOps Assignment 01 — Final Report

**Course:** MLOps AIMLCZG523  
**Dataset:** UCI Heart Disease  
**Student:** _Fill in_  
**Repository URL:** _Fill in_  
**Deployment URL / access instructions:** _Fill in_

## 1. Executive summary
Describe the objective, approach, key measured results, limitations, and final recommendation. Do not invent metrics; use your actual `reports/model_metrics.json`.

## 2. Problem and dataset
- Source and retrieval date:
- Dataset size and feature descriptions:
- Target definition (explain original target values > 0 mapped to disease present):
- Missing values and duplicates:
- Dataset limitations and potential bias:

## 3. EDA
Insert `reports/eda/class_balance.png`, `correlation_heatmap.png`, and `feature_histograms.png`.
Discuss class balance, distributions, correlations, missingness, and any observations.

## 4. Feature engineering and modelling
Describe imputation, scaling, categorical encoding, train/test split, cross-validation strategy, and why Logistic Regression and Random Forest were selected.

| Model | CV ROC-AUC | Test accuracy | Test precision | Test recall | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | Fill from run | | | | |
| Random Forest | Fill from run | | | | |

Explain the selected metric and trade-offs. For a health-related screening task, discuss the consequences of false negatives and why threshold selection requires domain expertise.

## 5. Experiment tracking
Include MLflow screenshot(s), experiment name, parameters and logged metrics/artifacts. Explain how runs are compared and reproduced.

## 6. Architecture and reproducibility
Insert an architecture diagram. Explain repository structure, pinned/ranged dependencies, saved preprocessing pipeline, and steps for a clean setup.

## 7. Automated tests and CI/CD
Describe unit tests, linting, training job, artifact upload, and evidence from a successful GitHub Actions run. Include a screenshot.

## 8. Containerization and API
Document Docker build/run commands, `/health`, `/predict`, `/metrics`, example request/response, validation, logs, and container test evidence.

## 9. Deployment
Document target environment (Minikube, Docker Desktop Kubernetes, or cloud), manifests, service exposure, replicas/probes/resources, endpoint verification, and deployment screenshots.

## 10. Monitoring, risks, and next steps
Discuss request/latency metrics, log visibility, data/model drift, alerting, privacy, security, and the fact that this educational model is not clinically validated.

## Appendix
- Reproduction commands
- CI workflow link
- Video link
- Repository link
- Screenshot inventory

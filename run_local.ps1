$ErrorActionPreference = "Stop"
python -m heart_disease.data --download
python -m heart_disease.eda
python -m heart_disease.train
Write-Host "Model trained. Start API with: uvicorn heart_disease.api:app --host 0.0.0.0 --port 8000"

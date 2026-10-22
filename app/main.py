import os
import sys
from pathlib import Path
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure the app directory is in sys.path so 'predictor' can be imported anywhere
app_dir = Path(__file__).resolve().parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

try:
    from app.predictor import load_artifacts, predict
except ImportError:
    from predictor import load_artifacts, predict

app = FastAPI(title="Smart Manufacturing ML Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_artifacts = None


def get_artifacts():
    global _artifacts
    if _artifacts is None:
        _artifacts = load_artifacts()
    return _artifacts


@app.get("/")
def root():
    return {"service": "Smart Manufacturing ML Backend", "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
@app.get("/api/metadata")
def metadata():
    try:
        artifacts = get_artifacts()
        return artifacts["meta"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load artifacts: {e}")


@app.post("/predict")
@app.post("/api/predict")
def predict_endpoint(values: Dict[str, Any]):
    try:
        artifacts = get_artifacts()
        result = predict(artifacts, values)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

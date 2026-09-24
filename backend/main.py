"""
FastAPI backend for the Predictive Risk Monitoring System.
Serves PM2.5 concentration (µg/m³) predictions. NOT official AQI.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import numpy as np
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

import tensorflow as tf

from data_utils import (
    load_and_clean, dataset_summary, MODELS_DIR, DATA_PATH,
    INPUT_HOURS, FORECAST_HORIZON, scale, inverse_scale, load_scaler
)

app = FastAPI(
    title="Predictive Risk Monitoring System – Urban AQI",
    description=(
        "Historical PM2.5 concentration forecasting demo using Beijing data. "
        "This system predicts PM2.5 (µg/m³), NOT official AQI values."
    ),
    version="1.0.0",
)

# CORS for local Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------- lazy-loaded singletons ---------------
_state = {}


def _get_df():
    if "df" not in _state:
        if not DATA_PATH.exists():
            raise HTTPException(503, f"Dataset not found at {DATA_PATH}. Download from Kaggle and place CSV in data/ folder.")
        _state["df"] = load_and_clean()
    return _state["df"]


def _get_models():
    if "cnn" not in _state:
        cnn_path = MODELS_DIR / "cnn_model.keras"
        lstm_path = MODELS_DIR / "lstm_model.keras"
        if not cnn_path.exists() or not lstm_path.exists():
            raise HTTPException(503, "Models not found. Run training first: python train.py")
        _state["cnn"] = tf.keras.models.load_model(str(cnn_path))
        _state["lstm"] = tf.keras.models.load_model(str(lstm_path))
    return _state["cnn"], _state["lstm"]


def _get_scaler():
    if "scaler" not in _state:
        try:
            _state["scaler"] = load_scaler()
        except FileNotFoundError:
            raise HTTPException(503, "Scaler not found. Run training first: python train.py")
    return _state["scaler"]


def _get_metrics():
    if "metrics" not in _state:
        path = MODELS_DIR / "metrics.json"
        if not path.exists():
            raise HTTPException(503, "Metrics not found. Run training first: python train.py")
        with open(path) as f:
            _state["metrics"] = json.load(f)
    return _state["metrics"]


def _get_test_preds():
    if "test_preds" not in _state:
        path = MODELS_DIR / "test_predictions.json"
        if not path.exists():
            raise HTTPException(503, "Test predictions not found. Run training first: python train.py")
        with open(path) as f:
            _state["test_preds"] = json.load(f)
    return _state["test_preds"]


# --------------- endpoints ---------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "dataset_found": DATA_PATH.exists(),
        "models_found": (MODELS_DIR / "cnn_model.keras").exists() and (MODELS_DIR / "lstm_model.keras").exists(),
    }


@app.get("/summary")
def summary():
    df = _get_df()
    metrics = _get_metrics()
    info = dataset_summary(df)
    info["model_metrics"] = metrics
    return info


@app.get("/history")
def history(limit: int = 2000):
    """Return a subset of historical PM2.5 readings for charts."""
    df = _get_df()
    sub = df[["timestamp", "pm2.5"]].dropna(subset=["pm2.5"])
    # Evenly sample if too many
    if len(sub) > limit:
        step = len(sub) // limit
        sub = sub.iloc[::step].head(limit)
    return {
        "timestamps": sub["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
        "pm25": sub["pm2.5"].tolist(),
    }


@app.get("/test-predictions")
def test_predictions():
    return _get_test_preds()


# --------------- Prediction request ---------------

class PredictRequest(BaseModel):
    pm25_values: List[float]

    @field_validator("pm25_values")
    @classmethod
    def check_length(cls, v):
        if len(v) != INPUT_HOURS:
            raise ValueError(f"Exactly {INPUT_HOURS} hourly PM2.5 readings required, got {len(v)}")
        for i, val in enumerate(v):
            if val < 0:
                raise ValueError(f"PM2.5 value at index {i} is negative ({val}). Values must be >= 0.")
        return v


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Accept 24 hourly PM2.5 readings and return CNN & LSTM
    predictions for the value 24 hours after the last reading.
    """
    scaler = _get_scaler()
    cnn_model, lstm_model = _get_models()

    raw = np.array(req.pm25_values, dtype=np.float32)
    scaled = scale(raw, scaler).reshape(1, INPUT_HOURS, 1)

    cnn_pred_scaled = cnn_model.predict(scaled, verbose=0).flatten()[0]
    lstm_pred_scaled = lstm_model.predict(scaled, verbose=0).flatten()[0]

    cnn_pred = float(inverse_scale(np.array([cnn_pred_scaled]), scaler)[0])
    lstm_pred = float(inverse_scale(np.array([lstm_pred_scaled]), scaler)[0])

    return {
        "forecast_description": f"Predicted PM2.5 concentration {FORECAST_HORIZON} hours after the last input reading",
        "unit": "µg/m³",
        "cnn_prediction": round(cnn_pred, 2),
        "lstm_prediction": round(lstm_pred, 2),
        "note": "These are PM2.5 concentration predictions, NOT official AQI values.",
    }

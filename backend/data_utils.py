"""
Data loading, cleaning, windowing, and splitting utilities.
All processing uses PM2.5 concentration (µg/m³), NOT official AQI.
"""

import numpy as np
import pandas as pd
import os, json, pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "PRSA_data_2010.1.1-2014.12.31.csv"
MODELS_DIR = BASE_DIR / "models"

INPUT_HOURS = 24
FORECAST_HORIZON = 24  # predict 24 hours ahead of last input
SEED = 42


def load_and_clean() -> pd.DataFrame:
    """Load CSV, build timestamps, sort, deduplicate, flag missing pm2.5."""
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(
        df[["year", "month", "day", "hour"]].rename(
            columns={"year": "year", "month": "month", "day": "day", "hour": "hour"}
        )
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)
    # pm2.5 may be 'NA' string or NaN
    df["pm2.5"] = pd.to_numeric(df["pm2.5"], errors="coerce")
    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    """Return basic dataset statistics."""
    return {
        "total_rows": int(len(df)),
        "start_date": str(df["timestamp"].min()),
        "end_date": str(df["timestamp"].max()),
        "missing_pm25": int(df["pm2.5"].isna().sum()),
        "valid_pm25": int(df["pm2.5"].notna().sum()),
    }


def chronological_split(df: pd.DataFrame):
    """Split into ~70/15/15 chronologically. Returns DataFrames."""
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def fit_scaler(train_df: pd.DataFrame) -> dict:
    """Compute min/max from training set only. Save to disk."""
    valid = train_df["pm2.5"].dropna()
    params = {"min": float(valid.min()), "max": float(valid.max())}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "scaler_params.json", "w") as f:
        json.dump(params, f)
    return params


def load_scaler() -> dict:
    with open(MODELS_DIR / "scaler_params.json", "r") as f:
        return json.load(f)


def scale(values: np.ndarray, params: dict) -> np.ndarray:
    """Min-max scale to [0, 1]."""
    denom = params["max"] - params["min"]
    if denom == 0:
        return np.zeros_like(values)
    return (values - params["min"]) / denom


def inverse_scale(values: np.ndarray, params: dict) -> np.ndarray:
    return values * (params["max"] - params["min"]) + params["min"]


def build_windows(df: pd.DataFrame, scaler_params: dict):
    """
    Build valid (X, y) windows from a dataframe.
    A window is valid ONLY if:
      - All 24 input rows + 1 target row have non-NaN pm2.5
      - The timestamps form a consecutive hourly sequence (no gaps)
      - Target timestamp == last-input timestamp + 24 hours
    """
    timestamps = df["timestamp"].values  # numpy datetime64
    pm25 = df["pm2.5"].values
    one_hour = np.timedelta64(1, "h")

    X_list, y_list, ts_input_list, ts_target_list = [], [], [], []
    total_span = INPUT_HOURS + FORECAST_HORIZON  # 24 input + 24 gap = need row at +48? No.
    # We need 24 consecutive hourly inputs, then the target at +24h from last input.
    # So last input is at index i+23, target is 24 hours after that.
    # We scan the dataframe for sequences.

    n = len(df)
    i = 0
    while i <= n - INPUT_HOURS:
        # Check if the 24 input readings are consecutive and valid
        input_slice = slice(i, i + INPUT_HOURS)
        input_ts = timestamps[input_slice]
        input_vals = pm25[input_slice]

        # All inputs must be valid
        if np.any(np.isnan(input_vals)):
            i += 1
            continue

        # Check hourly spacing in inputs
        diffs = np.diff(input_ts)
        if not np.all(diffs == one_hour):
            i += 1
            continue

        # Target timestamp: exactly 24 hours after last input
        target_ts = input_ts[-1] + np.timedelta64(FORECAST_HORIZON, "h")

        # Find target in dataframe
        target_mask = timestamps == target_ts
        if not np.any(target_mask):
            i += 1
            continue

        target_idx = np.where(target_mask)[0][0]
        target_val = pm25[target_idx]

        if np.isnan(target_val):
            i += 1
            continue

        # Verify: target is exactly 24h after last input
        assert (target_ts - input_ts[-1]) == np.timedelta64(FORECAST_HORIZON, "h")

        # Scale
        x_scaled = scale(input_vals, scaler_params)
        y_scaled = scale(np.array([target_val]), scaler_params)

        X_list.append(x_scaled)
        y_list.append(y_scaled[0])
        ts_input_list.append(pd.Timestamp(input_ts[-1]))
        ts_target_list.append(pd.Timestamp(target_ts))

        i += 1

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, ts_input_list, ts_target_list

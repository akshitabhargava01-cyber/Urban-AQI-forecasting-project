"""
Training script for the Predictive Risk Monitoring System.
Trains persistence baseline, 1D CNN, and LSTM models on PM2.5 data.
All values are PM2.5 concentration (µg/m³), NOT official AQI.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import json
from pathlib import Path

np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

from tensorflow import keras
from tensorflow.keras import layers
from data_utils import (
    load_and_clean, chronological_split, fit_scaler,
    build_windows, inverse_scale, MODELS_DIR, INPUT_HOURS, SEED
)

MAX_EPOCHS = 5
BATCH_SIZE = 64


def evaluate(y_true, y_pred):
    """MAE and RMSE in original PM2.5 units."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"mae": round(mae, 2), "rmse": round(rmse, 2)}


def build_cnn(input_length):
    model = keras.Sequential([
        layers.Input(shape=(input_length, 1)),
        layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ], name="cnn")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def build_lstm(input_length):
    model = keras.Sequential([
        layers.Input(shape=(input_length, 1)),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ], name="lstm")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main():
    print("=" * 60)
    print("Predictive Risk Monitoring System - Training")
    print("Predicting PM2.5 concentration (µg/m³), NOT official AQI")
    print("=" * 60)

    # 1. Load and clean
    print("\n[1/6] Loading and cleaning data...")
    df = load_and_clean()
    print(f"  Total rows: {len(df)}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Missing PM2.5: {df['pm2.5'].isna().sum()}")

    # 2. Chronological split
    print("\n[2/6] Splitting data chronologically (70/15/15)...")
    train_df, val_df, test_df = chronological_split(df)
    print(f"  Train: {len(train_df)} rows ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"  Val:   {len(val_df)} rows ({val_df['timestamp'].min()} to {val_df['timestamp'].max()})")
    print(f"  Test:  {len(test_df)} rows ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")

    # 3. Fit scaler on training data only
    print("\n[3/6] Fitting scaler on training data...")
    scaler_params = fit_scaler(train_df)
    print(f"  PM2.5 min: {scaler_params['min']}, max: {scaler_params['max']}")

    # 4. Build windows (separately for each split)
    print("\n[4/6] Building valid windows...")
    X_train, y_train, _, _ = build_windows(train_df, scaler_params)
    X_val, y_val, _, _ = build_windows(val_df, scaler_params)
    X_test, y_test, ts_inputs_test, ts_targets_test = build_windows(test_df, scaler_params)
    print(f"  Train windows: {len(X_train)}")
    print(f"  Val windows:   {len(X_val)}")
    print(f"  Test windows:  {len(X_test)}")

    if len(X_train) == 0:
        print("ERROR: No valid training windows. Check data.")
        return

    # Reshape for Conv1D / LSTM: (samples, timesteps, features)
    X_train_3d = X_train.reshape(-1, INPUT_HOURS, 1)
    X_val_3d = X_val.reshape(-1, INPUT_HOURS, 1)
    X_test_3d = X_test.reshape(-1, INPUT_HOURS, 1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    # 5. Persistence baseline
    print("\n[5/6] Evaluating persistence baseline...")
    # Baseline: predict = last input value (already scaled), inverse-scale both
    baseline_pred_scaled = X_test[:, -1]  # last of 24 inputs
    baseline_pred = inverse_scale(baseline_pred_scaled, scaler_params)
    y_test_orig = inverse_scale(y_test, scaler_params)
    baseline_metrics = evaluate(y_test_orig, baseline_pred)
    all_metrics["persistence"] = baseline_metrics
    print(f"  Persistence MAE:  {baseline_metrics['mae']} µg/m³")
    print(f"  Persistence RMSE: {baseline_metrics['rmse']} µg/m³")

    # 6. Train CNN
    print("\n[6/6] Training CNN...")
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )
    cnn = build_cnn(INPUT_HOURS)
    cnn.fit(
        X_train_3d, y_train,
        validation_data=(X_val_3d, y_val),
        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[early_stop], verbose=1
    )
    cnn.save(str(MODELS_DIR / "cnn_model.keras"))
    cnn_pred_scaled = cnn.predict(X_test_3d, verbose=0).flatten()
    cnn_pred = inverse_scale(cnn_pred_scaled, scaler_params)
    cnn_metrics = evaluate(y_test_orig, cnn_pred)
    all_metrics["cnn"] = cnn_metrics
    print(f"  CNN MAE:  {cnn_metrics['mae']} µg/m³")
    print(f"  CNN RMSE: {cnn_metrics['rmse']} µg/m³")

    # 7. Train LSTM
    print("\nTraining LSTM...")
    lstm = build_lstm(INPUT_HOURS)
    lstm.fit(
        X_train_3d, y_train,
        validation_data=(X_val_3d, y_val),
        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[early_stop], verbose=1
    )
    lstm.save(str(MODELS_DIR / "lstm_model.keras"))
    lstm_pred_scaled = lstm.predict(X_test_3d, verbose=0).flatten()
    lstm_pred = inverse_scale(lstm_pred_scaled, scaler_params)
    lstm_metrics = evaluate(y_test_orig, lstm_pred)
    all_metrics["lstm"] = lstm_metrics
    print(f"  LSTM MAE:  {lstm_metrics['mae']} µg/m³")
    print(f"  LSTM RMSE: {lstm_metrics['rmse']} µg/m³")

    # Save metrics
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Save test predictions for frontend charts
    test_results = {
        "timestamps": [str(t) for t in ts_targets_test],
        "actual": y_test_orig.tolist(),
        "persistence": baseline_pred.tolist(),
        "cnn": cnn_pred.tolist(),
        "lstm": lstm_pred.tolist(),
    }
    with open(MODELS_DIR / "test_predictions.json", "w") as f:
        json.dump(test_results, f)

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - Test Set Results (PM2.5 µg/m³)")
    print("=" * 60)
    print(f"{'Model':<15} {'MAE':>8} {'RMSE':>8}")
    print("-" * 31)
    for name, m in all_metrics.items():
        print(f"{name:<15} {m['mae']:>8.2f} {m['rmse']:>8.2f}")

    best = min(all_metrics, key=lambda k: all_metrics[k]["mae"])
    print(f"\nLowest test MAE: {best}")
    if best == "persistence":
        print("Note: The persistence baseline outperformed the neural models.")
        print("This can happen with limited epochs or when PM2.5 has strong autocorrelation.")
    print("\nAll models and metrics saved to:", MODELS_DIR)


if __name__ == "__main__":
    main()

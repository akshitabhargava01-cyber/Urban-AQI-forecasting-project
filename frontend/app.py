"""
Streamlit frontend for the Predictive Risk Monitoring System.
Calls the FastAPI backend for all data and predictions.
Displays PM2.5 concentration (µg/m³), NOT official AQI.
"""

import streamlit as st
import requests
import pandas as pd
import json

BACKEND_URL = "http://127.0.0.1:8001"

st.set_page_config(page_title="Urban AQI - PM2.5 Forecast", layout="wide")


def api_get(endpoint: str):
    try:
        r = requests.get(f"{BACKEND_URL}{endpoint}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error("❌ Cannot reach the FastAPI backend. Make sure it is running: `uvicorn main:app --reload` in the backend/ folder.")
        return None
    except requests.HTTPError as e:
        st.error(f"⚠️ Backend error: {e.response.text}")
        return None


def api_post(endpoint: str, data: dict):
    try:
        r = requests.post(f"{BACKEND_URL}{endpoint}", json=data, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error("❌ Cannot reach the FastAPI backend.")
        return None
    except requests.HTTPError as e:
        st.error(f"⚠️ Backend error: {e.response.text}")
        return None


# ============ Header ============
st.title("🌍 Predictive Risk Monitoring System – Urban AQI")
st.markdown("""
**Historical PM2.5 Concentration Forecasting Demo**

This dashboard uses historical Beijing air quality data (2010–2014) to demonstrate
24-hour-ahead PM2.5 concentration (µg/m³) prediction using baseline, CNN, and LSTM models.
""")

st.warning("""
⚠️ **Important Notice:** This project predicts **PM2.5 concentration (µg/m³)**, which is
**NOT** the same as an official Air Quality Index (AQI). It uses historical data and is
**NOT** a live air-quality monitoring system. The title "Urban AQI" is a project name only.
""")

# ============ Health Check ============
health = api_get("/health")
if health is None:
    st.stop()

if not health.get("dataset_found"):
    st.error("Dataset not found. Place `PRSA_data_2010.1.1-2014.12.31.csv` in the `data/` folder.")
    st.stop()

if not health.get("models_found"):
    st.warning("Models not yet trained. Run `python train.py` in the `backend/` folder first.")
    st.stop()

# ============ Summary ============
st.header("📊 Dataset & Model Summary")
summary = api_get("/summary")
if summary:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{summary['total_rows']:,}")
    col2.metric("Date Range", f"{summary['start_date'][:10]} to {summary['end_date'][:10]}")
    col3.metric("Missing PM2.5", f"{summary['missing_pm25']:,}")
    col4.metric("Valid PM2.5", f"{summary['valid_pm25']:,}")

    # Model metrics comparison
    st.subheader("Model Test Metrics (PM2.5 µg/m³)")
    metrics = summary.get("model_metrics", {})
    if metrics:
        metrics_df = pd.DataFrame(metrics).T
        metrics_df.index.name = "Model"
        metrics_df.columns = [c.upper() for c in metrics_df.columns]
        st.dataframe(metrics_df.style.highlight_min(axis=0, color="#90EE90"), use_container_width=True)

        st.caption("Lower is better. Green highlights the best (lowest) value in each column.")

        # Bar chart
        st.bar_chart(metrics_df)

# ============ Historical Chart ============
st.header("📈 Historical PM2.5 Readings")
history = api_get("/history?limit=2000")
if history:
    hist_df = pd.DataFrame({
        "Timestamp": pd.to_datetime(history["timestamps"]),
        "PM2.5 (µg/m³)": history["pm25"]
    }).set_index("Timestamp")
    st.line_chart(hist_df, use_container_width=True)
    st.caption("Sampled historical PM2.5 concentration readings (up to 2000 points).")

# ============ Test Predictions Chart ============
st.header("🎯 Actual vs Predicted (Test Set)")
test_preds = api_get("/test-predictions")
if test_preds:
    pred_df = pd.DataFrame({
        "Timestamp": pd.to_datetime(test_preds["timestamps"]),
        "Actual": test_preds["actual"],
        "Persistence": test_preds["persistence"],
        "CNN": test_preds["cnn"],
        "LSTM": test_preds["lstm"],
    }).set_index("Timestamp")

    # Show only a window for readability
    max_show = 200
    if len(pred_df) > max_show:
        st.info(f"Showing first {max_show} of {len(pred_df)} test predictions for readability.")
        show_df = pred_df.head(max_show)
    else:
        show_df = pred_df

    st.line_chart(show_df, use_container_width=True)
    st.caption("All values in PM2.5 concentration (µg/m³). Chart shows actual vs model predictions on the test set.")

# ============ Predict Section ============
st.header("🔮 Make a Prediction")
st.markdown("""
Provide **24 consecutive hourly PM2.5 readings** (µg/m³).
The models will predict the PM2.5 concentration **24 hours after the last reading**.
""")

tab1, tab2 = st.tabs(["📋 Use sample from dataset", "✍️ Enter manually"])

with tab1:
    if test_preds and len(test_preds.get("actual", [])) > 0:
        # Get a sample sequence from history
        hist_for_sample = api_get("/history?limit=5000")
        if hist_for_sample and len(hist_for_sample["pm25"]) >= 24:
            sample_values = hist_for_sample["pm25"][:24]
            st.write("**Sample 24-hour sequence (first 24 valid readings from dataset):**")
            st.code(json.dumps(sample_values, indent=2))

            if st.button("🚀 Predict using this sample", key="sample_predict"):
                result = api_post("/predict", {"pm25_values": sample_values})
                if result:
                    st.success(f"🎯 **{result['forecast_description']}**")
                    c1, c2 = st.columns(2)
                    c1.metric("CNN Prediction", f"{result['cnn_prediction']} µg/m³")
                    c2.metric("LSTM Prediction", f"{result['lstm_prediction']} µg/m³")
                    st.info(result["note"])

with tab2:
    user_input = st.text_area(
        "Enter 24 PM2.5 values (comma-separated, µg/m³):",
        placeholder="e.g. 120, 130, 125, 140, ...",
        height=100
    )
    if st.button("🚀 Predict", key="manual_predict"):
        if user_input.strip():
            try:
                values = [float(x.strip()) for x in user_input.split(",") if x.strip()]
                if len(values) != 24:
                    st.error(f"Expected 24 values, got {len(values)}.")
                else:
                    result = api_post("/predict", {"pm25_values": values})
                    if result:
                        st.success(f"🎯 **{result['forecast_description']}**")
                        c1, c2 = st.columns(2)
                        c1.metric("CNN Prediction", f"{result['cnn_prediction']} µg/m³")
                        c2.metric("LSTM Prediction", f"{result['lstm_prediction']} µg/m³")
                        st.info(result["note"])
            except ValueError:
                st.error("Invalid input. Enter numeric values separated by commas.")
        else:
            st.warning("Please enter 24 PM2.5 values.")

# ============ Footer ============
st.divider()
st.caption("""
**Predictive Risk Monitoring System – Urban AQI** | Historical forecasting demo
| Data: Beijing PM2.5 (2010–2014) | Predicts PM2.5 concentration (µg/m³), not official AQI
""")

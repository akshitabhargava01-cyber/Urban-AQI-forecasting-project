# Predictive Risk Monitoring System – Urban AQI

## What This Project Does

A **historical PM2.5 concentration forecasting demo** using Beijing air quality data (2010–2014). Given 24 consecutive hourly PM2.5 readings, the system predicts the **PM2.5 concentration (µg/m³) exactly 24 hours after the last input reading**.

Three approaches are compared:
- **Persistence baseline** – predicts the last known value will persist.
- **1D CNN** – a small convolutional neural network.
- **LSTM** – a small recurrent neural network.

### What This Project Does NOT Do

- ❌ Does **not** compute or predict official AQI values. PM2.5 concentration ≠ AQI.
- ❌ Does **not** provide live or real-time air quality monitoring.
- ❌ Does **not** use external APIs, cloud services, or databases.

> **Note:** The title "Urban AQI" is a project name only. All predictions, metrics, and charts show **PM2.5 concentration in µg/m³**.

---

## Dataset

**Source:** [PM2.5 Air Pollution Dataset (Kaggle)](https://www.kaggle.com/datasets/ineubytes/pm25-airpolution-dataset)

**Expected file:** `data/PRSA_data_2010.1.1-2014.12.31.csv`

### Download Instructions

1. Go to the Kaggle link above.
2. Download the dataset (you may need a free Kaggle account).
3. Extract the CSV file.
4. Place it in the `data/` folder so the path is:
   ```
   data/PRSA_data_2010.1.1-2014.12.31.csv
   ```

> The CSV file is **not committed** to the repository due to size and Kaggle license. You must download it manually.

---

## Project Structure

```
Predictive Risk Monitoring System – Urban AQI/
├── frontend/
│   └── app.py              # Streamlit dashboard (calls FastAPI backend)
├── backend/
│   ├── main.py             # FastAPI application with all endpoints
│   ├── train.py            # Model training script
│   └── data_utils.py       # Data loading, cleaning, windowing utilities
├── data/
│   ├── .gitkeep
│   └── PRSA_data_2010.1.1-2014.12.31.csv  (download from Kaggle)
├── models/
│   └── .gitkeep            # Trained models saved here by train.py
├── requirements.txt
├── README.md
└── .gitignore
```

Both **frontend** and **backend** are in the **same single repository**.

---

## Setup and Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Install Dependencies

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
# source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

---

## Step 1: Train the Models

```bash
cd backend
python train.py
```

This will:
1. Load and clean the dataset from `data/`.
2. Split chronologically into 70% training, 15% validation, 15% test.
3. Fit min-max scaling on training data only.
4. Build valid 24-hour input windows with verified timestamps.
5. Evaluate the persistence baseline.
6. Train the CNN and LSTM models (max 5 epochs with early stopping).
7. Save models, scaler parameters, metrics, and test predictions to `models/`.

**Training takes approximately 2–5 minutes** depending on hardware.

---

## Step 2: Start the FastAPI Backend

Open **Terminal 1**:

```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

The API will be available at:
- **API:** http://127.0.0.1:8001
- **Interactive docs (Swagger):** http://127.0.0.1:8001/docs
- **ReDoc:** http://127.0.0.1:8001/redoc

---

## Step 3: Start the Streamlit Frontend

Open **Terminal 2**:

```bash
cd frontend
streamlit run app.py --server.port 8501
```

The dashboard will open at: **http://localhost:8501**

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Backend status, dataset/model availability |
| GET | `/summary` | Dataset stats, date range, missing values, model metrics |
| GET | `/history?limit=2000` | Historical PM2.5 readings for charts |
| GET | `/test-predictions` | Actual vs predicted values on test set |
| POST | `/predict` | Accept 24 hourly PM2.5 readings, return CNN & LSTM predictions |

### POST /predict Example

```json
{
  "pm25_values": [120, 130, 125, 140, 135, 128, 132, 145, 150, 138,
                  127, 133, 141, 136, 129, 142, 148, 137, 126, 134,
                  139, 143, 131, 144]
}
```

Response:
```json
{
  "forecast_description": "Predicted PM2.5 concentration 24 hours after the last input reading",
  "unit": "µg/m³",
  "cnn_prediction": 135.42,
  "lstm_prediction": 138.17,
  "note": "These are PM2.5 concentration predictions, NOT official AQI values."
}
```

---

## How Frontend Communicates with Backend

The Streamlit frontend (`frontend/app.py`) makes HTTP requests to the FastAPI backend running on `http://127.0.0.1:8001`. It uses the Python `requests` library to call the REST API endpoints. The backend has CORS enabled for local development. **The frontend does not process data directly** — all data loading, model inference, and metrics come from the backend.

---

## Technical Details

### Data Processing
- Timestamps constructed from `year`, `month`, `day`, `hour` columns.
- Sorted chronologically, duplicates removed.
- PM2.5 values parsed; `NA` strings treated as missing.

### Data Split
- **Training:** First ~70% of data chronologically.
- **Validation:** Next ~15%.
- **Test:** Final ~15%.
- Windows are built separately per split (no cross-boundary leakage).

### Windowing
- Each valid window: 24 consecutive hourly PM2.5 readings (all non-NaN, hourly spacing verified).
- Target: PM2.5 value exactly 24 hours after the last input timestamp.
- Invalid windows (missing values, timestamp gaps) are excluded.

### Scaling
- Min-max scaling fitted on **training data only**.
- Same parameters applied to validation, test, and inference.

### Models
- **Persistence:** Last input value = prediction. No training needed.
- **CNN:** Conv1D(32, kernel=3) → GlobalAvgPool → Dense(16) → Dense(1).
- **LSTM:** LSTM(32) → Dense(16) → Dense(1).
- Both trained with Adam optimizer, MSE loss, early stopping (patience=2), max 5 epochs.

### Evaluation Metrics
- **MAE** (Mean Absolute Error) in µg/m³
- **RMSE** (Root Mean Squared Error) in µg/m³
- Computed on the test set in original (un-scaled) units.

### Forecasting Target
- **Input:** 24 consecutive hourly PM2.5 readings.
- **Output:** 1 PM2.5 reading exactly 24 hours after the last input.
- Example: If last input is Monday 12:00, prediction is for Tuesday 12:00.

---

## Limitations

- Uses **PM2.5 concentration only** as input (no weather features, wind, etc.).
- Historical data from one city (Beijing) only.
- Small models with limited training (max 5 epochs) for quick builds.
- The persistence baseline may outperform neural models with limited epochs.
- Not suitable for real-world deployment without additional validation.

---

## PM2.5 ≠ AQI

**Important:** PM2.5 concentration (µg/m³) is a raw pollutant measurement. Official AQI is a composite index calculated from multiple pollutants using regulatory formulas that vary by country. This project predicts **PM2.5 concentration only** and does not compute AQI.

---

## License

This project is for educational and demonstration purposes.
Dataset: [PM2.5 Air Pollution Dataset](https://www.kaggle.com/datasets/ineubytes/pm25-airpolution-dataset) – see Kaggle for dataset license terms.

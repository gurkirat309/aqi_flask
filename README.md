# AQI Dashboard + Prediction Flask App

A complete Flask application to visualize historical AQI from CSV, fetch live AQI from OpenWeather, and predict next-day and 7-day AQI per city using RandomForest models.

## Features
- Dashboard with latest CSV AQI and live AQI per city
- Per-city and global RandomForest models
- Fallback to global model when per-city model is missing
- JSON prediction API
- Chart.js visualization for 7-day forecast
- Robust OpenWeather requests with retries and in-memory caching
- Optional SMTP email alerts when predicted AQI exceeds a threshold

## Install
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train models
Per-city training across all cities in the CSV:
```bash
python train.py --data data/city_day.csv --save_dir models
```
Train a single city:
```bash
python train_single_city.py --data data/city_day.csv --city Bengaluru --save_dir models --n_lags 3
```
Train global model:
```bash
python train_global.py --data data/city_day.csv --save_dir models --n_lags 7
```

## Run app
```bash
python app.py
```
App runs at http://localhost:5000

## Environment variables
- `OPENWEATHER_API_KEY`: to enable live AQI fetch (optional)
- `OPENWEATHER_CACHE_SECONDS`: cache TTL in seconds (default 900)
- `ALERT_AQI_THRESHOLD`: alert threshold (default 200)
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`, `ALERT_TO`: for email alerts (optional)

## Global fallback behavior
- If `models/<City>/rf_model.joblib` is not found, the app automatically attempts to use `models/global_rf_model.joblib` with a label-encoded city.

## Troubleshooting
- OpenWeather timeouts: network issues happen. The app uses retries and short timeouts; you can disable live fetch by not setting `OPENWEATHER_API_KEY`.
- Sparse city history: per-city trainer will skip if not enough rows; use the global model.

## Tests
Run unit tests:
```bash
pytest -q
```


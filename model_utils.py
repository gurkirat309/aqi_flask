import numpy as np
import pandas as pd
from typing import List


def create_features(df: pd.DataFrame, n_lags: int = 7) -> pd.DataFrame:
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])  # safe parse
        df.sort_values("Date", inplace=True)
    # Generate lag features on AQI
    for i in range(1, n_lags + 1):
        df[f"AQI_lag_{i}"] = df["AQI"].shift(i)
    # Rolling mean
    df["AQI_roll_mean_3"] = df["AQI"].rolling(window=min(3, max(1, len(df)))).mean().shift(1)
    # Calendar features
    df["dayofweek"] = df["Date"].dt.dayofweek.astype(int)
    df["month"] = df["Date"].dt.month.astype(int)
    return df


def _feature_columns(n_lags: int) -> List[str]:
    lags = [f"AQI_lag_{i}" for i in range(1, n_lags + 1)]
    return lags + ["AQI_roll_mean_3", "dayofweek", "month"]


def iterative_forecast(model, last_known_df: pd.DataFrame, days: int = 7, n_lags: int = 7) -> List[float]:
    df = last_known_df.copy()
    df = create_features(df, n_lags=n_lags)
    feats = _feature_columns(n_lags)

    preds: List[float] = []
    # Work on a small frame carrying only required columns to avoid inplace pitfalls
    work_df = df[["Date", "AQI"]].copy()
    work_df["Date"] = pd.to_datetime(work_df["Date"])  # ensure dtype

    for d in range(days):
        temp = create_features(work_df.rename(columns={"AQI": "AQI"}), n_lags=n_lags)
        X = temp[feats].tail(1)
        X = X.fillna(method="bfill").fillna(method="ffill").fillna(0.0)
        yhat = float(model.predict(X)[0])
        preds.append(yhat)
        # append new row with predicted AQI and next day date
        next_date = work_df["Date"].iloc[-1] + pd.Timedelta(days=1)
        work_df = pd.concat([
            work_df,
            pd.DataFrame({"Date": [next_date], "AQI": [yhat]})
        ], ignore_index=True)
    return preds

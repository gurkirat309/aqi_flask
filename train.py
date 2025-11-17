import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from model_utils import create_features


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")


def train_city(df_city: pd.DataFrame, n_lags: int = 7):
    df_feat = create_features(df_city, n_lags=n_lags)
    feat_cols = [f"AQI_lag_{i}" for i in range(1, n_lags + 1)] + ["AQI_roll_mean_3", "dayofweek", "month"]
    df_feat = df_feat.dropna(subset=feat_cols + ["AQI"])  # ensure usable
    if len(df_feat) < max(5, n_lags + 3):
        return None, None, None
    X = df_feat[feat_cols].values
    y = df_feat["AQI"].values
    model = RandomForestRegressor(n_estimators=200, random_state=42)

    n_samples = len(X)
    n_splits = min(3, max(2, n_samples // 5))
    cv_scores = None
    if n_samples > n_splits:
        try:
            tss = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = cross_val_score(model, X, y, cv=tss, scoring="neg_mean_absolute_error")
        except Exception as e:
            logging.warning(f"CV failed due to {e}. Proceeding to train on all data.")
    else:
        logging.info("Not enough rows for CV; training on all data.")

    model.fit(X, y)
    return model, feat_cols, cv_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--city", required=False)
    args = parser.parse_args()

    data_path = Path(args.data)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])

    cities = [args.city] if args.city else sorted(df["City"].unique().tolist())

    trained = []
    skipped = []

    for city in cities:
        dff = df[df["City"] == city].sort_values("Date")
        model, feats, cv_scores = train_city(dff)
        if model is None:
            logging.info(f"Skipping city {city}: not enough usable rows")
            skipped.append(city)
            continue
        city_dir = save_dir / city.replace(" ", "_")
        city_dir.mkdir(parents=True, exist_ok=True)
        pkg = {
            "model": model,
            "features": feats,
            "n_lags": sum(1 for f in feats if f.startswith("AQI_lag_")),
        }
        joblib.dump(pkg, city_dir / "rf_model.joblib")
        trained.append(city)
        if cv_scores is not None:
            logging.info(f"{city} CV MAE: {np.mean(-cv_scores):.2f} (+/- {np.std(-cv_scores):.2f})")

    logging.info(f"Training complete. Trained: {len(trained)}; Skipped: {len(skipped)}")
    if skipped:
        logging.info("Skipped cities: " + ", ".join(skipped))


if __name__ == "__main__":
    main()

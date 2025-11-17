import argparse
from pathlib import Path
import logging

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from model_utils import create_features

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--city', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--n_lags', type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df['Date'] = pd.to_datetime(df['Date'])

    dff = df[df['City'] == args.city].sort_values('Date')
    if dff.empty:
        raise SystemExit(f"City {args.city} not found in data")

    feat_df = create_features(dff, n_lags=args.n_lags)
    feat_cols = [f"AQI_lag_{i}" for i in range(1, args.n_lags + 1)] + ['AQI_roll_mean_3','dayofweek','month']
    feat_df = feat_df.dropna(subset=feat_cols + ['AQI'])
    if len(feat_df) < max(5, args.n_lags + 3):
        raise SystemExit("Not enough data to train per configuration")

    X = feat_df[feat_cols].values
    y = feat_df['AQI'].values

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    city_dir = Path(args.save_dir) / args.city.replace(' ', '_')
    city_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'features': feat_cols, 'n_lags': args.n_lags}, city_dir / 'rf_model.joblib')
    logging.info(f"Saved model for {args.city} -> {city_dir / 'rf_model.joblib'}")


if __name__ == '__main__':
    main()

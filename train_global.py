import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from model_utils import create_features

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--n_lags', type=int, default=7)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df['Date'] = pd.to_datetime(df['Date'])

    frames = []
    for city, dff in df.groupby('City'):
        tmp = create_features(dff.sort_values('Date'), n_lags=args.n_lags)
        tmp['City'] = city
        frames.append(tmp)
    full = pd.concat(frames, ignore_index=True)

    feat_cols = [f"AQI_lag_{i}" for i in range(1, args.n_lags + 1)] + ['AQI_roll_mean_3','dayofweek','month']
    full = full.dropna(subset=feat_cols + ['AQI'])

    le = LabelEncoder()
    full['City_enc'] = le.fit_transform(full['City'])

    X = np.c_[full[feat_cols].values, full['City_enc'].values]
    y = full['AQI'].values

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y)

    pkg = {
        'model': model,
        'label_encoder': le,
        'features': feat_cols,
        'n_lags': args.n_lags,
    }
    out_path = Path(args.save_dir) / 'global_rf_model.joblib'
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(pkg, out_path)
    logging.info(f"Saved global model -> {out_path}")


if __name__ == '__main__':
    main()

from pathlib import Path
import tempfile

import joblib
import subprocess
import sys


def run_py(cmd):
    return subprocess.run([sys.executable] + cmd, check=True)


def test_train_single_city_creates_model():
    root = Path(__file__).resolve().parents[1]
    data = root / 'data' / 'city_day.csv'
    with tempfile.TemporaryDirectory() as td:
        run_py(['train_single_city.py', '--data', str(data), '--city', 'Bengaluru', '--save_dir', td, '--n_lags', '3'])
        out = Path(td) / 'Bengaluru' / 'rf_model.joblib'
        assert out.exists()
        pkg = joblib.load(out)
        assert 'model' in pkg and 'features' in pkg


def test_train_creates_models_for_all():
    root = Path(__file__).resolve().parents[1]
    data = root / 'data' / 'city_day.csv'
    with tempfile.TemporaryDirectory() as td:
        run_py(['train.py', '--data', str(data), '--save_dir', td])
        # at least one city should be trained
        paths = list(Path(td).glob('*/rf_model.joblib'))
        assert len(paths) >= 1

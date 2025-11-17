from pathlib import Path
from unittest import mock
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import app as aqi_app  # noqa: E402


@pytest.fixture
def client():
    aqi_app.app.config.update({'TESTING': True})
    with aqi_app.app.test_client() as client:
        yield client


def test_dashboard_ok(client):
    with mock.patch.object(aqi_app, 'fetch_live_aqi', return_value=None):
        r = client.get('/')
    assert r.status_code == 200


def test_predict_form_ok(client):
    r = client.get('/predict')
    assert r.status_code == 200


def test_api_predict_bengaluru(client, tmp_path):
    # Train a quick per-city model into temp models dir and swap MODELS_DIR
    from subprocess import run
    import sys as _sys
    data = Path(__file__).resolve().parents[1] / 'data' / 'city_day.csv'
    tmp_models = tmp_path / 'models'
    tmp_models.mkdir(parents=True, exist_ok=True)
    run([_sys.executable, 'train_single_city.py', '--data', str(data), '--city', 'Bengaluru', '--save_dir', str(tmp_models), '--n_lags', '3'], check=True)

    old = aqi_app.MODELS_DIR
    aqi_app.MODELS_DIR = tmp_models
    try:
        r = client.get('/api/predict/Bengaluru')
        assert r.status_code == 200
        js = r.get_json()
        assert js['city'] == 'Bengaluru'
        assert isinstance(js['next_day'], float)
        assert isinstance(js['trend_7'], list)
        assert len(js['trend_7']) == 7
    finally:
        aqi_app.MODELS_DIR = old

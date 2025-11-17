import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request, redirect, url_for, jsonify, abort
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from model_utils import create_features, iterative_forecast

# Minimal .env loader (no external dependency)
def _load_env_file(path: Path = Path('.env')):
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception as e:
        logging.warning(f"Failed to load .env: {e}")

_load_env_file()
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

DATA_PATH = Path('data/city_day.csv')
MODELS_DIR = Path('models')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
OPENWEATHER_CACHE_SECONDS = int(os.getenv('OPENWEATHER_CACHE_SECONDS', '900'))
ALERT_AQI_THRESHOLD = int(os.getenv('ALERT_AQI_THRESHOLD', '200'))

_df = pd.read_csv(DATA_PATH)
_df['Date'] = pd.to_datetime(_df['Date'])
logging.info(f"Loaded data rows: {_df.shape[0]} from {DATA_PATH}")

# Simple in-memory cache
_cache: Dict[str, Any] = {}


def _requests_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
    s.mount('https://', HTTPAdapter(max_retries=retries))
    s.mount('http://', HTTPAdapter(max_retries=retries))
    return s


SESSION = _requests_session()


def cache_get(key: str) -> Optional[Any]:
    v = _cache.get(key)
    if not v:
        return None
    if time.time() - v['ts'] > v['ttl']:
        _cache.pop(key, None)
        return None
    return v['val']


def cache_set(key: str, val: Any, ttl: int):
    _cache[key] = {'val': val, 'ts': time.time(), 'ttl': ttl}


def compute_aqi_pm25(pm25: float) -> float:
    # Approximate EPA breakpoints
    brks = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for c_lo, c_hi, i_lo, i_hi in brks:
        if c_lo <= pm25 <= c_hi:
            return (i_hi - i_lo) / (c_hi - c_lo) * (pm25 - c_lo) + i_lo
    return float(pm25)  # fallback


def aqi_category_and_advice(aqi: float):
    a = float(aqi)
    if a <= 50:
        return 'Good', 'Air quality is good. Enjoy outdoor activities.'
    if a <= 100:
        return 'Satisfactory', 'Air is acceptable. People unusually sensitive should limit prolonged outdoor exertion.'
    if a <= 200:
        return 'Moderate', 'Consider reducing prolonged or heavy outdoor exertion; sensitive groups should take precautions.'
    if a <= 300:
        return 'Poor', 'Limit outdoor activities; wear a mask if you must go out.'
    if a <= 400:
        return 'Very Poor', 'Avoid outdoor activities; stay indoors and use air purification if available.'
    return 'Severe', 'Stay indoors. Avoid all outdoor exertion. Follow local health advisories.'


def geocode_city(city: str) -> Optional[Dict[str, float]]:
    if not OPENWEATHER_API_KEY:
        return None
    key = f"geocode:{city}"
    cached = cache_get(key)
    if cached:
        return cached
    try:
        resp = SESSION.get(
            'https://api.openweathermap.org/geo/1.0/direct',
            params={'q': city, 'limit': 1, 'appid': OPENWEATHER_API_KEY},
            timeout=5,
        )
        resp.raise_for_status()
        arr = resp.json()
        if not arr:
            return None
        loc = {'lat': float(arr[0]['lat']), 'lon': float(arr[0]['lon'])}
        cache_set(key, loc, OPENWEATHER_CACHE_SECONDS)
        return loc
    except Exception as e:
        logging.warning(f"Geocode failed for {city}: {e}")
        return None


def fetch_live_aqi(city: str) -> Optional[Dict[str, Any]]:
    if not OPENWEATHER_API_KEY:
        return None
    key = f"aqi:{city}"
    cached = cache_get(key)
    if cached:
        return cached
    loc = geocode_city(city)
    if not loc:
        return None
    try:
        resp = SESSION.get(
            'https://api.openweathermap.org/data/2.5/air_pollution',
            params={'lat': loc['lat'], 'lon': loc['lon'], 'appid': OPENWEATHER_API_KEY},
            timeout=5,
        )
        resp.raise_for_status()
        js = resp.json()
        if 'list' not in js or not js['list']:
            return None
        comp = js['list'][0]['components']
        pm25 = float(comp.get('pm2_5', 0))
        aqi_num = compute_aqi_pm25(pm25)
        aqi_idx = int(js['list'][0].get('main', {}).get('aqi', 0))
        result = {'pm2_5': pm25, 'aqi_estimate': float(aqi_num), 'ow_index': aqi_idx}
        cache_set(key, result, OPENWEATHER_CACHE_SECONDS)
        return result
    except Exception as e:
        logging.warning(f"Air fetch failed for {city}: {e}")
        return None


def load_city_model(city: str):
    city_path = MODELS_DIR / city.replace(' ', '_') / 'rf_model.joblib'
    if city_path.exists():
        return joblib.load(city_path)
    glob_path = MODELS_DIR / 'global_rf_model.joblib'
    if glob_path.exists():
        return joblib.load(glob_path)
    return None


def send_email_alert(subject: str, body: str):
    import smtplib
    from email.mime.text import MIMEText

    host = os.getenv('SMTP_HOST')
    port = int(os.getenv('SMTP_PORT', '0') or 0)
    user = os.getenv('SMTP_USER')
    pw = os.getenv('SMTP_PASS')
    to_addr = os.getenv('ALERT_TO')
    if not (host and port and user and pw and to_addr):
        logging.warning('SMTP not configured; skipping email alert')
        return
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = user
        msg['To'] = to_addr
        with smtplib.SMTP(host, port, timeout=10) as s:
            s.starttls()
            s.login(user, pw)
            s.send_message(msg)
    except Exception as e:
        logging.warning(f"Email sending failed: {e}")


def send_email_to(to_addr: str, subject: str, body: str):
    import smtplib
    from email.mime.text import MIMEText
    host = os.getenv('SMTP_HOST')
    port = int(os.getenv('SMTP_PORT', '0') or 0)
    user = os.getenv('SMTP_USER')
    pw = os.getenv('SMTP_PASS')
    if not (host and user and pw and to_addr):
        msg = 'SMTP not fully configured or recipient missing.'
        logging.warning(msg)
        return False, msg
    # Build message once
    msg_obj = MIMEText(body)
    msg_obj['Subject'] = subject
    msg_obj['From'] = user
    msg_obj['To'] = to_addr
    # Attempt STARTTLS if port suggests 587
    try:
        use_port = port or 587
        with smtplib.SMTP(host, use_port, timeout=10) as s:
            s.starttls()
            s.login(user, pw)
            s.send_message(msg_obj)
        return True, None
    except Exception as e1:
        logging.warning(f"Email send via STARTTLS failed: {e1}")
        # Fallback to SSL 465
        try:
            with smtplib.SMTP_SSL(host, 465, timeout=10) as s:
                s.login(user, pw)
                s.send_message(msg_obj)
            return True, None
        except Exception as e2:
            logging.warning(f"Email send via SSL failed: {e2}")
            return False, str(e2)


@app.route('/')
def dashboard():
    cities = set(_df['City'].unique().tolist())
    extra = os.getenv('DASHBOARD_EXTRA_CITIES', '')
    if extra:
        for c in [x.strip() for x in extra.split(',') if x.strip()]:
            cities.add(c)
    cities = sorted(cities)
    latest = _df.sort_values('Date').groupby('City').tail(1).set_index('City')
    rows = []
    for city in cities:
        csv_aqi = float(latest.loc[city, 'AQI']) if city in latest.index else None
        live = fetch_live_aqi(city)
        # choose best available AQI for advice (prefer live)
        ref_aqi = float(live['aqi_estimate']) if live and 'aqi_estimate' in live else (csv_aqi if csv_aqi is not None else None)
        cat, advice = (aqi_category_and_advice(ref_aqi) if ref_aqi is not None else (None, None))
        rows.append({'city': city, 'csv_aqi': csv_aqi, 'live': live, 'category': cat, 'advice': advice})
    try:
        return render_template('dashboard.html', rows=rows)
    except Exception:
        return jsonify({'cities': rows})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    cities = set(_df['City'].unique().tolist())
    extra = os.getenv('DASHBOARD_EXTRA_CITIES', '')
    if extra:
        for c in [x.strip() for x in extra.split(',') if x.strip()]:
            cities.add(c)
    cities = sorted(cities)
    if request.method == 'POST':
        city = request.form.get('city')
        if not city:
            return redirect(url_for('predict'))
        return redirect(url_for('predict_city', city=city))
    return render_template('predict.html', cities=cities)


@app.route('/predict/<city>', methods=['GET', 'POST'])
def predict_city(city: str = None):  # type: ignore
    city = city or request.view_args.get('city')
    dff = _df[_df['City'] == city].sort_values('Date')
    if dff.empty:
        abort(404)
    pkg = load_city_model(city)
    if not pkg:
        abort(404)

    n_lags = int(pkg.get('n_lags', 7))
    feats = pkg.get('features')
    model = pkg.get('model')

    if 'label_encoder' in pkg:  # global model
        le = pkg['label_encoder']
        tmp = create_features(dff, n_lags=n_lags).dropna()
        if tmp.empty:
            abort(404)
        # build feature row(s)
        X_last = tmp[feats].tail(1).values
        city_enc = le.transform([city])[0]
        X_last = np.c_[X_last, [city_enc]]
        # simple next-day
        next_day = float(model.predict(X_last)[0])
        # iterative 7-day
        # Build a shim model that appends city code to features
        class _GlobWrap:
            def __init__(self, base_model, city_code):
                self.base_model = base_model
                self.city_code = city_code
            def predict(self, X):
                X2 = np.c_[X, np.full((len(X), 1), self.city_code)]
                return self.base_model.predict(X2)
        wrapper = _GlobWrap(model, city_enc)
        trend = iterative_forecast(wrapper, dff[['Date', 'AQI']], days=7, n_lags=n_lags)
        src = 'global'
    else:
        next_day = iterative_forecast(pkg['model'], dff[['Date', 'AQI']], days=1, n_lags=n_lags)[0]
        trend = iterative_forecast(pkg['model'], dff[['Date', 'AQI']], days=7, n_lags=n_lags)
        src = 'per-city'

    if next_day >= ALERT_AQI_THRESHOLD:
        try:
            send_email_alert(
                subject=f"AQI Alert for {city}: {next_day:.0f}",
                body=f"Predicted next-day AQI for {city} is {next_day:.1f}",
            )
        except Exception:
            pass

    # Build labels for next 7 days
    last_date = pd.to_datetime(dff['Date'].max())
    labels = [(last_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
    category, advice = aqi_category_and_advice(next_day)
    email_sent = False
    email_to = None
    email_error = None
    if request.method == 'POST':
        email_to = request.form.get('email')
        if email_to and '@' in email_to:
            subj = f"AQI Advice for {city}: {next_day:.0f} ({category})"
            body = (
                f"City: {city}\n"
                f"Predicted next-day AQI: {next_day:.1f}\n"
                f"Category: {category}\n"
                f"Suggestion: {advice}\n"
                f"7-day forecast: {', '.join(f'{x:.0f}' for x in trend)}\n"
            )
            email_sent, email_error = send_email_to(email_to, subj, body)
    try:
        return render_template('trend.html', city=city, next_day=next_day, trend=trend, source=src, labels=labels, category=category, advice=advice, email_sent=email_sent, email_to=email_to, email_error=email_error)
    except Exception:
        return jsonify({'city': city, 'next_day': float(next_day), 'trend_7': [float(x) for x in trend], 'labels': labels, 'source': src, 'category': category, 'advice': advice, 'email_sent': email_sent, 'email_to': email_to, 'email_error': email_error})


@app.route('/api/predict/<city>')
def api_predict_city(city):
    dff = _df[_df['City'] == city].sort_values('Date')
    if dff.empty:
        abort(404)
    pkg = load_city_model(city)
    if not pkg:
        abort(404)
    n_lags = int(pkg.get('n_lags', 7))
    if 'label_encoder' in pkg:
        le = pkg['label_encoder']
        feats = pkg['features']
        model = pkg['model']
        tmp = create_features(dff, n_lags=n_lags).dropna()
        if tmp.empty:
            abort(404)
        X_last = tmp[feats].tail(1).values
        city_enc = le.transform([city])[0]
        X_last = np.c_[X_last, [city_enc]]
        next_day = float(model.predict(X_last)[0])
        class _GlobWrap:
            def __init__(self, base_model, city_code):
                self.base_model = base_model
                self.city_code = city_code
            def predict(self, X):
                X2 = np.c_[X, np.full((len(X), 1), self.city_code)]
                return self.base_model.predict(X2)
        wrapper = _GlobWrap(model, city_enc)
        trend = iterative_forecast(wrapper, dff[['Date','AQI']], days=7, n_lags=n_lags)
        src = 'global'
    else:
        model = pkg['model']
        next_day = iterative_forecast(model, dff[['Date','AQI']], days=1, n_lags=n_lags)[0]
        trend = iterative_forecast(model, dff[['Date','AQI']], days=7, n_lags=n_lags)
        src = 'per-city'
    if next_day >= ALERT_AQI_THRESHOLD:
        try:
            send_email_alert(
                subject=f"AQI Alert for {city}: {next_day:.0f}",
                body=f"Predicted next-day AQI for {city} is {next_day:.1f}",
            )
        except Exception:
            pass
    category, advice = aqi_category_and_advice(next_day)
    return jsonify({'city': city, 'next_day': float(next_day), 'trend_7': [float(x) for x in trend], 'source': src, 'category': category, 'advice': advice})


if __name__ == '__main__':
    logging.info(f"Models present: {sum(1 for _ in MODELS_DIR.glob('**/rf_model.joblib'))}")
    logging.info(f"OpenWeather configured: {bool(OPENWEATHER_API_KEY)}")
    app.run(debug=True, host='0.0.0.0', port=5000)

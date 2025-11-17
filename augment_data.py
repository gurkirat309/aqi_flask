import pandas as pd
import numpy as np
from pathlib import Path

# Synthesize 30 days for new cities and append to data/city_day.csv
DATA = Path('data/city_day.csv')
NEW_CITIES = [
    'Chennai','Hyderabad','Kolkata','Pune','Ahmedabad','Jaipur','Lucknow'
]

# Use same date range as existing sample (2025-10-01..2025-10-30)
dates = pd.date_range('2025-10-01', '2025-10-30', freq='D')

def approx_aqi_from_pm25(pm25: float) -> float:
    # simple piecewise based on EPA-like mapping
    brks = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
    ]
    for c_lo, c_hi, i_lo, i_hi in brks:
        if c_lo <= pm25 <= c_hi:
            return (i_hi - i_lo) / (c_hi - c_lo) * (pm25 - c_lo) + i_lo
    return pm25

# Base ranges per city (rough heuristic)
ranges = {
    'Chennai': (35, 80),
    'Hyderabad': (40, 90),
    'Kolkata': (55, 120),
    'Pune': (30, 75),
    'Ahmedabad': (60, 140),
    'Jaipur': (50, 110),
    'Lucknow': (65, 150),
}

rows = []
for city in NEW_CITIES:
    lo, hi = ranges[city]
    pm25_series = np.clip(np.linspace(lo, hi, len(dates)) + np.random.normal(0, 3, len(dates)), 10, 220)
    pm10_series = pm25_series * 2 + np.random.normal(0, 5, len(dates))
    for d, pm25, pm10 in zip(dates, pm25_series, pm10_series):
        aqi = round(approx_aqi_from_pm25(float(pm25)))
        bucket = (
            'Good' if aqi <= 50 else
            'Satisfactory' if aqi <= 100 else
            'Moderate' if aqi <= 200 else
            'Poor'
        )
        rows.append({
            'City': city,
            'Date': d.strftime('%Y-%m-%d'),
            'PM2.5': round(float(pm25)),
            'PM10': round(float(pm10)),
            'NO': 10, 'NO2': 20, 'NOx': 25, 'NH3': 5, 'CO': 0.8,
            'SO2': 10, 'O3': 20, 'Benzene': 2, 'Toluene': 7, 'Xylene': 1,
            'AQI': aqi, 'AQI_Bucket': bucket,
        })

new_df = pd.DataFrame(rows)
orig = pd.read_csv(DATA)
merged = pd.concat([orig, new_df], ignore_index=True)
merged.to_csv(DATA, index=False)
print(f"Appended {len(new_df)} rows for {len(NEW_CITIES)} cities. Total now: {len(merged)} rows")

"""
Empirical Indian Grid Benchmark: Multi-Year Delhi Macro Grid Telemetry & ERA5 Solar Data
Constructs an authentic empirical Indian utility benchmark dataset (Jan 2021 - Jun 2024) by joining:
1. Delhi State Load Despatch Centre (SLDC) / Grid-India SCADA telemetry and surface meteorological
   observations (coderkarishma/Delhi-Electricity-Demand-Prediction GitHub repository).
   - Columns: Power demand (MW), surface temperature (°C), relative humidity (%), wind speed (km/h).
2. ECMWF ERA5 Reanalysis Historical Weather Archive (Open-Meteo API) for Delhi coordinates (28.61°N, 77.21°E):
   - Column: direct_radiation (solar irradiance, W/m²).

Engineering & Methodology Notes:
- Strict Time Alignment: Resamples to a continuous 1-hour time grid and interpolates short gaps (<= 3 hours).
  Rows with larger gaps are left as NaNs and dropped during feature creation so lag features strictly reflect
  true physical time differences (no row-count shifting across missing gaps).
- Macro Demand Proxy: Delhi city-wide demand (1,500 to 8,600 MW) is scaled by 1.10 into an equivalent
  1,685 to 9,422 kW test envelope (8,000 kW continuous capacity, 7,200 kW supervisory SCADA alarm).
  As noted in the project limitations, aggregate city-wide curves are smoother than single distribution feeders.
"""

import os
import json
import urllib.request
import pandas as pd
import numpy as np

DELHI_COORDS = (28.61, 77.21)
START_DATE = "2021-01-01"
END_DATE = "2024-06-30"

WEATHER_API_URL = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={DELHI_COORDS[0]}&longitude={DELHI_COORDS[1]}&"
    f"start_date={START_DATE}&end_date={END_DATE}&"
    f"hourly=direct_radiation"
)

def build_indian_grid_benchmark(output_path=None):
    data_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(data_dir, "delhi_demand_raw.csv")
    solar_cache_path = os.path.join(data_dir, "delhi_solar_cache.json")
    if output_path is None:
        output_path = os.path.join(data_dir, "indian_grid_benchmark.csv")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Official Delhi SLDC raw telemetry not found at {raw_path}. "
            "Please ensure data/delhi_demand_raw.csv is present."
        )

    print(f"[INFO] Loading raw Delhi SLDC telemetry from {raw_path}...")
    df_raw = pd.read_csv(raw_path)
    df_raw['timestamp'] = pd.to_datetime(df_raw['datetime'])
    df_raw = df_raw.sort_values('timestamp').drop_duplicates('timestamp').set_index('timestamp')

    # Step 1: Resample to strict continuous hourly grid (2021-01-01 to 2024-06-30)
    print(f"[INFO] Aligning to continuous hourly time grid ({START_DATE} to {END_DATE})...")
    df_hourly = df_raw[['Power demand', 'temp', 'rhum', 'wspd']].resample('1h').mean()
    df_window = df_hourly.loc[f"{START_DATE} 00:00:00":f"{END_DATE} 23:59:59"].copy()

    # Step 2: Interpolate short gaps (<= 3 hours), leaving larger gaps as NaN
    df_window = df_window.interpolate(method='time', limit=3).reset_index()
    print(f"[INFO] Total hourly slots: {len(df_window):,} | Valid load records: {df_window['Power demand'].notna().sum():,}")

    # Step 3: Fetch or load empirical solar irradiance from Open-Meteo ERA5
    solar_data = None
    if os.path.exists(solar_cache_path):
        try:
            with open(solar_cache_path, 'r') as f:
                solar_data = json.load(f)
            print(f"[INFO] Loaded solar irradiance from local cache: {solar_cache_path}")
        except Exception as e:
            print(f"[WARNING] Cache read failed: {e}")
            solar_data = None

    if solar_data is None:
        print(f"[INFO] Fetching ERA5 historical solar irradiance for Delhi ({DELHI_COORDS[0]}°N, {DELHI_COORDS[1]}°E)...")
        try:
            req = urllib.request.Request(WEATHER_API_URL, headers={'User-Agent': 'SmartEnergyGridBenchmarker/2.0'})
            with urllib.request.urlopen(req, timeout=35) as resp:
                solar_data = json.loads(resp.read().decode('utf-8'))['hourly']
            with open(solar_cache_path, 'w') as f:
                json.dump(solar_data, f)
            print(f"[SUCCESS] Retrieved and cached solar irradiance ({len(solar_data['time'])} records).")
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch ERA5 solar irradiance from Open-Meteo API ({e}) and no local cache found. "
                "Ensure internet connectivity or ensure data/delhi_solar_cache.json exists."
            ) from e

    df_solar = pd.DataFrame({
        'timestamp': pd.to_datetime(solar_data['time']),
        'solar_irradiance_wm2': np.round(solar_data['direct_radiation'], 1)
    })

    # Step 4: Merge telemetry and solar irradiance on exact timestamp
    df_merged = pd.merge(df_window, df_solar, on='timestamp', how='inner')

    # Step 5: Calendar annotations & Indian Gazetted Holidays (2021-2024)
    df_merged['is_weekend'] = (df_merged['timestamp'].dt.dayofweek >= 5).astype(int)
    indian_holidays = pd.to_datetime([
        '2021-01-26', '2021-03-29', '2021-08-15', '2021-10-02', '2021-10-15', '2021-11-04', '2021-12-25',
        '2022-01-26', '2022-03-18', '2022-08-15', '2022-10-02', '2022-10-05', '2022-10-24', '2022-12-25',
        '2023-01-26', '2023-03-08', '2023-08-15', '2023-10-02', '2023-10-24', '2023-11-12', '2023-12-25',
        '2024-01-26', '2024-03-25', '2024-04-11', '2024-04-17', '2024-05-23', '2024-06-17'
    ])
    df_merged['is_holiday'] = df_merged['timestamp'].dt.normalize().isin(indian_holidays).astype(int)

    # Step 6: Zone metadata & Benchmark scaling
    df_merged['zone_id'] = 'Delhi_Macro_Grid'
    df_merged['population'] = 140000
    df_merged['is_mnc_zone'] = 1
    df_merged['temperature_c'] = np.round(df_merged['temp'], 2)
    df_merged['humidity_pct'] = np.round(df_merged['rhum'], 1)
    df_merged['wind_speed_kmh'] = np.round(df_merged['wspd'], 2)
    df_merged['delhi_raw_mw'] = np.round(df_merged['Power demand'], 2)
    df_merged['load_kwh'] = np.round(df_merged['Power demand'] * 1.10, 2)
    df_merged['transformer_capacity'] = 8000.0

    cols = [
        'timestamp', 'zone_id', 'population', 'is_mnc_zone', 'transformer_capacity',
        'temperature_c', 'humidity_pct', 'solar_irradiance_wm2', 'wind_speed_kmh',
        'is_weekend', 'is_holiday', 'delhi_raw_mw', 'load_kwh'
    ]
    df_final = df_merged[cols].copy()
    df_final.to_csv(output_path, index=False)
    print(f"[SUCCESS] Created Multi-Year Indian Grid Benchmark: {output_path} ({len(df_final):,} rows)")
    return df_final

if __name__ == '__main__':
    build_indian_grid_benchmark()

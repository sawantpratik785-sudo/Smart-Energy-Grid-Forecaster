"""
Time-Series Feature Engineering Module for Energy Grid Forecasting
Constructs time-lagged features, rolling window metrics, cyclic time encodings,
and meteorological interaction features.
"""

import numpy as np
import pandas as pd

def create_time_series_features(df, target_col='load_kwh', is_inference=False):
    """
    Transforms raw hourly smart grid data into rich time-series features.
    
    Parameters:
        df (pd.DataFrame): Raw DataFrame containing timestamp and weather features.
        target_col (str): Column name of target variable (kwh load).
        is_inference (bool): If True, handles single/short horizon prediction windows.
        
    Returns:
        pd.DataFrame: Feature matrix (X) and target vector (y) if target_col present.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 1. Calendar & Temporal Features
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['dayofyear'] = df['timestamp'].dt.dayofyear
    df['quarter'] = df['timestamp'].dt.quarter

    # 2. Cyclic Time Encodings (sine/cosine transformations)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12.0)

    # 3. Weather Interaction & Non-linear Features
    df['temp_squared'] = df['temperature_c'] ** 2
    df['temp_humidity_idx'] = df['temperature_c'] * (df['humidity_pct'] / 100.0)
    df['is_extreme_temp'] = ((df['temperature_c'] > 32) | (df['temperature_c'] < 6)).astype(int)

    # 4. Lagged Target Features
    if target_col in df.columns:
        df['load_lag_1'] = df[target_col].shift(1)
        df['load_lag_2'] = df[target_col].shift(2)
        df['load_lag_24'] = df[target_col].shift(24)
        df['load_lag_168'] = df[target_col].shift(168)

        # 5. Rolling Window Statistics
        df['rolling_mean_6'] = df[target_col].shift(1).rolling(window=6).mean()
        df['rolling_mean_24'] = df[target_col].shift(1).rolling(window=24).mean()
        df['rolling_std_24'] = df[target_col].shift(1).rolling(window=24).std()

    # Drop NaNs created by lag shifting if not inference mode
    if not is_inference:
        df = df.dropna().reset_index(drop=True)

    return df

FEATURE_COLUMNS = [
    'temperature_c', 'humidity_pct', 'solar_irradiance_wm2', 'wind_speed_kmh',
    'is_weekend', 'is_holiday', 'hour', 'dayofweek', 'month', 'dayofyear',
    'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
    'temp_squared', 'temp_humidity_idx', 'is_extreme_temp',
    'load_lag_1', 'load_lag_2', 'load_lag_24', 'load_lag_168',
    'rolling_mean_6', 'rolling_mean_24', 'rolling_std_24'
]

if __name__ == '__main__':
    import os
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'energy_grid_hourly.csv')
    df_raw = pd.read_csv(data_path)
    df_feat = create_time_series_features(df_raw)
    print(f"[SUCCESS] Processed {len(df_feat)} rows with {len(FEATURE_COLUMNS)} feature columns.")

"""
Smart Energy Grid Peak-Demand Dataset Generator
Generates a realistic 1-year hourly smart grid dataset (8,760 observations per zone)
incorporating meteorological parameters, demographic features (Population, MNC zones).
"""

import os
import numpy as np
import pandas as pd

def generate_energy_data(num_days=365, seed=42):
    np.random.seed(seed)
    start_date = pd.Timestamp('2025-01-01 00:00:00')
    hours = num_days * 24
    dates = pd.date_range(start=start_date, periods=hours, freq='h')

    # Time components
    day_of_year = dates.dayofyear.values
    hour_of_day = dates.hour.values
    day_of_week = dates.dayofweek.values
    month = dates.month.values
    is_weekend = (day_of_week >= 5).astype(int)

    # Holidays list
    holidays = [
        '2025-01-01', '2025-01-26', '2025-08-15', '2025-10-02',
        '2025-10-24', '2025-11-01', '2025-12-25'
    ]
    holiday_dates = set(pd.to_datetime(holidays).date)
    is_holiday = np.array([1 if d.date() in holiday_dates else 0 for d in dates])

    # Weather
    seasonal_temp = 25 + 12 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
    diurnal_temp = 6 * np.sin(2 * np.pi * (hour_of_day - 9) / 24)
    temp_noise = np.random.normal(0, 1.8, hours)
    temperature = np.round(seasonal_temp + diurnal_temp + temp_noise, 2)

    humidity = np.clip(np.round(75 - 1.2 * (temperature - 20) + np.random.normal(0, 4.0, hours), 2), 20.0, 98.0)
    
    solar_base = np.maximum(0, np.sin(np.pi * (hour_of_day - 6) / 12))
    solar_base[hour_of_day < 6] = 0
    solar_base[hour_of_day > 18] = 0
    solar_irradiance = np.clip(np.round(solar_base * (400 + 400 * np.sin(2 * np.pi * day_of_year / 365)) + np.random.normal(0, 25, hours), 2), 0, 1000)
    wind_speed = np.clip(np.round(12 + 6 * np.sin(2 * np.pi * day_of_year / 180) + np.random.normal(0, 3.5, hours), 2), 2.0, 60.0)

    cooling_demand = np.maximum(0, temperature - 24) ** 1.5 * 12.0
    heating_demand = np.maximum(0, 15 - temperature) ** 1.3 * 10.0
    hvac_load_base = cooling_demand + heating_demand

    zones = [
        {"id": "Sector_A_Residential", "pop": 45000, "is_mnc": 0, "base": 800, "cap": 2500, "vol": 1.0},
        {"id": "Sector_B_Commercial", "pop": 12000, "is_mnc": 1, "base": 1500, "cap": 3500, "vol": 1.5},
        {"id": "Sector_C_Industrial", "pop": 5000, "is_mnc": 1, "base": 2200, "cap": 4500, "vol": 2.0}
    ]

    all_dfs = []
    for z in zones:
        # Load Curve variations based on zone
        if z["is_mnc"]:
            # MNCs peak during day (9 AM - 6 PM)
            diurnal = 800 * np.exp(-((hour_of_day - 14) ** 2) / 20.0)
            weekend_drop = 1.0 - 0.4 * is_weekend
        else:
            # Residential peaks in morning and evening
            diurnal = 300 * np.exp(-((hour_of_day - 8) ** 2) / 6.0) + 600 * np.exp(-((hour_of_day - 20) ** 2) / 8.0)
            weekend_drop = 1.0 + 0.1 * is_weekend # slightly more power on weekends at home

        holiday_drop = 1.0 - 0.3 * is_holiday if z["is_mnc"] else 1.0 + 0.05 * is_holiday
        
        # Population multiplier
        pop_mult = z["pop"] / 10000.0

        raw_load = (z["base"] + diurnal * pop_mult + hvac_load_base * pop_mult * z["vol"]) * weekend_drop * holiday_drop
        
        load_noise = np.random.normal(0, 45.0 * z["vol"], hours)
        spike_mask = np.random.binomial(1, 0.015, hours)
        spikes = spike_mask * np.random.uniform(500, 1500, hours) * z["vol"]

        load_kwh = np.round(raw_load + load_noise + spikes, 2)

        df = pd.DataFrame({
            'zone_id': z["id"],
            'timestamp': dates,
            'population': z["pop"],
            'is_mnc_zone': z["is_mnc"],
            'transformer_capacity': z["cap"],
            'temperature_c': temperature,
            'humidity_pct': humidity,
            'solar_irradiance_wm2': solar_irradiance,
            'wind_speed_kmh': wind_speed,
            'is_weekend': is_weekend,
            'is_holiday': is_holiday,
            'load_kwh': load_kwh
        })
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'energy_grid_hourly.csv')
    
    df = generate_energy_data(num_days=365)
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Generated dataset: {len(df)} rows saved to {csv_path}")

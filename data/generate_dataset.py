"""
Smart Energy Grid Peak-Demand Dataset Generator
Generates a realistic 1-year hourly smart grid dataset (8,760 observations)
incorporating meteorological parameters, calendar patterns, and power demand (kWh).
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

    # Holidays list (approximate major public holidays)
    holidays = [
        '2025-01-01', '2025-01-26', '2025-08-15', '2025-10-02',
        '2025-10-24', '2025-11-01', '2025-12-25'
    ]
    holiday_dates = set(pd.to_datetime(holidays).date)
    is_holiday = np.array([1 if d.date() in holiday_dates else 0 for d in dates])

    # Temperature profile (°C): Seasonal sine wave + daily diurnal variation + noise
    # Seasonal: peak in summer (July, month 7), lowest in winter (January)
    seasonal_temp = 22 + 12 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
    diurnal_temp = 5 * np.sin(2 * np.pi * (hour_of_day - 9) / 24)
    temp_noise = np.random.normal(0, 1.8, hours)
    temperature = seasonal_temp + diurnal_temp + temp_noise
    temperature = np.round(temperature, 2)

    # Humidity (%): Negatively correlated with temperature
    humidity = 70 - 0.9 * (temperature - 20) + np.random.normal(0, 4.0, hours)
    humidity = np.clip(np.round(humidity, 2), 20.0, 98.0)

    # Solar Irradiance (W/m²): 0 at night, sine profile during day
    solar_base = np.maximum(0, np.sin(np.pi * (hour_of_day - 6) / 12))
    solar_base[hour_of_day < 6] = 0
    solar_base[hour_of_day > 18] = 0
    solar_irradiance = solar_base * (400 + 400 * np.sin(2 * np.pi * day_of_year / 365))
    solar_irradiance = np.clip(np.round(solar_irradiance + np.random.normal(0, 25, hours), 2), 0, 1000)

    # Wind Speed (km/h)
    wind_speed = np.round(15 + 6 * np.sin(2 * np.pi * day_of_year / 180) + np.random.normal(0, 3.5, hours), 2)
    wind_speed = np.clip(wind_speed, 2.0, 60.0)

    # Energy Demand Baseline & Modulation (kWh)
    base_load = 24000.0

    # Diurnal load curve: Morning peak (8-10 AM), Evening peak (6-10 PM), Night dip (2-5 AM)
    diurnal_load = (
        3500 * np.exp(-((hour_of_day - 9) ** 2) / 6.0) +   # Morning peak
        5500 * np.exp(-((hour_of_day - 20) ** 2) / 8.0) -  # Evening peak
        4000 * np.exp(-((hour_of_day - 4) ** 2) / 5.0)     # Night dip
    )

    # HVAC (Heating / Air Conditioning) demand penalty:
    # High load above 24°C (Air Conditioning), high load below 12°C (Heating)
    cooling_demand = np.maximum(0, temperature - 24) ** 1.45 * 320.0
    heating_demand = np.maximum(0, 12 - temperature) ** 1.35 * 280.0
    hvac_load = cooling_demand + heating_demand

    # Weekend drop (15% reduction in industrial consumption)
    weekend_factor = 1.0 - 0.15 * is_weekend

    # Holiday drop (20% reduction)
    holiday_factor = 1.0 - 0.20 * is_holiday

    # Combined load calculation
    raw_load = (base_load + diurnal_load + hvac_load) * weekend_factor * holiday_factor

    # Add realistic load noise and heatwave spikes
    load_noise = np.random.normal(0, 650.0, hours)
    
    # Occasional extreme peak heatwave / cold snap spikes (2% probability)
    spike_mask = np.random.binomial(1, 0.02, hours)
    spikes = spike_mask * np.random.uniform(3000, 7000, hours)

    load_kwh = np.round(raw_load + load_noise + spikes, 2)

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'temperature_c': temperature,
        'humidity_pct': humidity,
        'solar_irradiance_wm2': solar_irradiance,
        'wind_speed_kmh': wind_speed,
        'is_weekend': is_weekend,
        'is_holiday': is_holiday,
        'load_kwh': load_kwh
    })

    return df

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'energy_grid_hourly.csv')
    
    df = generate_energy_data(num_days=365)
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Generated dataset: {len(df)} rows saved to {csv_path}")
    print(f"Summary Statistics:\n{df.describe().T[['mean', 'std', 'min', '50%', 'max']]}")

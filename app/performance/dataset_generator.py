"""
Realistic Synthetic Air Quality & Weather Dataset Generator for AirSpark.
Generates multi-source correlated spatio-temporal data with controlled anomalies,
missing values, and duplicate records for benchmarking and testing data-quality pipelines.
"""
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

from app.utils.logging import get_logger
from app.utils.paths import ensure_dir, resolve_path

logger = get_logger("AirSpark.DatasetGenerator")

# Standard Reference Indian/Global Cities with Geocoordinates
SAMPLE_CITIES = [
    {"city": "Delhi", "state": "Delhi", "lat": 28.6139, "lon": 77.2090, "base_aqi": 180, "type": "Traffic"},
    {"city": "Mumbai", "state": "Maharashtra", "lat": 19.0760, "lon": 72.8777, "base_aqi": 110, "type": "Coastal"},
    {"city": "Bengaluru", "state": "Karnataka", "lat": 12.9716, "lon": 77.5946, "base_aqi": 75, "type": "Residential"},
    {"city": "Kolkata", "state": "West Bengal", "lat": 22.5726, "lon": 88.3639, "base_aqi": 150, "type": "Industrial"},
    {"city": "Hyderabad", "state": "Telangana", "lat": 17.3850, "lon": 78.4867, "base_aqi": 95, "type": "Commercial"},
    {"city": "Chennai", "state": "Tamil Nadu", "lat": 13.0827, "lon": 80.2707, "base_aqi": 80, "type": "Coastal"},
    {"city": "Ahmedabad", "state": "Gujarat", "lat": 23.0225, "lon": 72.5714, "base_aqi": 140, "type": "Industrial"},
    {"city": "Pune", "state": "Maharashtra", "lat": 18.5204, "lon": 73.8567, "base_aqi": 85, "type": "Residential"},
    {"city": "Jaipur", "state": "Rajasthan", "lat": 26.9124, "lon": 75.7873, "base_aqi": 130, "type": "Commercial"},
    {"city": "Lucknow", "state": "Uttar Pradesh", "lat": 26.8467, "lon": 80.9462, "base_aqi": 170, "type": "Traffic"},
]


class SyntheticDataGenerator:
    """
    Generates multi-source synthetic datasets with controlled data quality artifacts.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def generate_stations(self, num_stations: int = 10) -> pd.DataFrame:
        """Generate station metadata."""
        stations = []
        for i in range(num_stations):
            city_info = SAMPLE_CITIES[i % len(SAMPLE_CITIES)]
            # Add slight jitter to coordinates for multiple stations in the same city
            jitter_lat = np.random.normal(0, 0.05)
            jitter_lon = np.random.normal(0, 0.05)
            station_id = f"STN_{i+1:03d}"
            stations.append({
                "station_id": station_id,
                "station_name": f"{city_info['city']} Central Sensor #{i+1}",
                "city": city_info["city"],
                "state": city_info["state"],
                "latitude": round(city_info["lat"] + jitter_lat, 6),
                "longitude": round(city_info["lon"] + jitter_lon, 6),
                "elevation": round(float(np.random.uniform(50, 600)), 1),
                "station_type": city_info["type"],
            })
        return pd.DataFrame(stations)

    def generate_datasets(
        self,
        num_stations: int = 5,
        days: int = 7,
        frequency_minutes: int = 60,
        inject_quality_issues: bool = True,
        start_date: Optional[datetime] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate (air_quality_df, weather_df, stations_df).
        Correlates weather conditions with pollutants realistically.
        """
        stations_df = self.generate_stations(num_stations)
        start_dt = start_date or datetime(2026, 1, 1, 0, 0, 0)
        total_steps = int((days * 24 * 60) / frequency_minutes)

        aq_records: List[Dict[str, Any]] = []
        weather_records: List[Dict[str, Any]] = []

        for _, station in stations_df.iterrows():
            station_id = station["station_id"]
            stn_type = station["station_type"]

            # Type multipliers
            type_mult = {
                "Traffic": {"pm": 1.4, "no2": 1.6, "co": 1.5, "so2": 1.0, "o3": 0.8},
                "Industrial": {"pm": 1.5, "no2": 1.3, "co": 1.2, "so2": 2.0, "o3": 0.9},
                "Residential": {"pm": 0.8, "no2": 0.7, "co": 0.7, "so2": 0.6, "o3": 1.1},
                "Commercial": {"pm": 1.0, "no2": 1.1, "co": 1.0, "so2": 0.9, "o3": 1.0},
                "Coastal": {"pm": 0.7, "no2": 0.8, "co": 0.8, "so2": 0.7, "o3": 1.2},
            }.get(stn_type, {"pm": 1.0, "no2": 1.0, "co": 1.0, "so2": 1.0, "o3": 1.0})

            # Base weather state for the station
            base_temp = np.random.uniform(18.0, 32.0)
            base_pressure = np.random.uniform(1005.0, 1018.0)

            for step in range(total_steps):
                current_time = start_dt + timedelta(minutes=step * frequency_minutes)
                hour = current_time.hour
                doy = current_time.timetuple().tm_yday
                time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

                # Weather simulation
                # Diurnal temperature cycle: peaks around 14:00 (2 PM)
                temp_diurnal = 6.0 * math.sin((hour - 8) * math.pi / 12)
                temp = float(np.clip(base_temp + temp_diurnal + np.random.normal(0, 1.0), -5.0, 48.0))
                
                # Humidity is inversely related to temperature
                humidity = float(np.clip(75.0 - (temp - 20) * 1.8 + np.random.normal(0, 4.0), 15.0, 98.0))
                
                # Wind speed varies with random gusts
                wind_speed = float(np.clip(3.5 + 2.0 * math.sin(hour * math.pi / 12) + np.random.exponential(1.5), 0.2, 35.0))
                wind_dir = float((np.random.normal(180, 60) + hour * 5) % 360)
                pressure = float(base_pressure - 0.1 * (temp - 25) + np.random.normal(0, 0.5))
                
                # Occasional rainfall
                has_rain = np.random.random() < 0.05
                rainfall = float(np.random.exponential(8.0)) if has_rain else 0.0

                # Pollutant physics simulation
                # Dispersion factor: high wind or rain significantly reduces particulates
                dispersion = 1.0 / (1.0 + 0.08 * wind_speed + (0.3 * rainfall if rainfall > 0 else 0))
                
                # Traffic peaks at 8-10 AM and 6-9 PM
                traffic_peak = 1.8 if (8 <= hour <= 10 or 18 <= hour <= 21) else 0.9
                
                # Inversion in late night / early morning increases surface concentration
                inversion_factor = 1.3 if (hour <= 6 or hour >= 22) else 1.0

                # Pollutant values
                base_pm25 = (38.0 * type_mult["pm"] * traffic_peak * inversion_factor * dispersion) + np.random.normal(0, 4.0)
                pm2_5 = float(np.clip(base_pm25, 2.0, 450.0))
                
                pm10 = float(np.clip(pm2_5 * np.random.uniform(1.4, 2.1) + np.random.normal(0, 6.0), 5.0, 650.0))
                no2 = float(np.clip((25.0 * type_mult["no2"] * traffic_peak * dispersion) + np.random.normal(0, 3.0), 1.0, 200.0))
                so2 = float(np.clip((12.0 * type_mult["so2"] * dispersion) + np.random.normal(0, 2.0), 0.5, 120.0))
                co = float(np.clip((1.2 * type_mult["co"] * traffic_peak * dispersion) + np.random.normal(0, 0.15), 0.1, 15.0))
                
                # Ozone forms photochemically in sunlight (peaks 12-16) with high temp and low NO
                sunlight = max(0.0, math.sin((hour - 6) * math.pi / 12)) if 6 <= hour <= 18 else 0.0
                o3 = float(np.clip((18.0 + 45.0 * sunlight * (temp / 30.0)) * type_mult["o3"] + np.random.normal(0, 3.0), 2.0, 160.0))

                aq_records.append({
                    "station_id": station_id,
                    "timestamp": time_str,
                    "pm2_5": round(pm2_5, 2),
                    "pm10": round(pm10, 2),
                    "no2": round(no2, 2),
                    "so2": round(so2, 2),
                    "co": round(co, 2),
                    "o3": round(o3, 2),
                })

                weather_records.append({
                    "station_id": station_id,
                    "timestamp": time_str,
                    "temperature": round(temp, 2),
                    "humidity": round(humidity, 2),
                    "wind_speed": round(wind_speed, 2),
                    "wind_direction": round(wind_dir, 1),
                    "pressure": round(pressure, 2),
                    "rainfall": round(rainfall, 2),
                })

        aq_df = pd.DataFrame(aq_records)
        weather_df = pd.DataFrame(weather_records)

        if inject_quality_issues:
            aq_df = self._inject_anomalies_and_noise(aq_df)
            weather_df = self._inject_weather_noise(weather_df)

        return aq_df, weather_df, stations_df

    def _inject_anomalies_and_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject missing values, duplicate rows, and statistical outliers."""
        n = len(df)
        if n == 0:
            return df

        df = df.copy()

        # 1. Inject missing values (approx 3% in PM2.5, NO2, etc.)
        for col in ["pm2_5", "pm10", "no2", "so2", "co", "o3"]:
            mask = np.random.random(n) < 0.03
            df.loc[mask, col] = np.nan

        # 2. Inject extreme statistical outliers / sensor glitches (approx 1%)
        outlier_indices = np.random.choice(n, size=max(1, int(n * 0.01)), replace=False)
        for idx in outlier_indices:
            anomaly_type = random.choice(["extreme_high", "negative", "stuck_sensor"])
            if anomaly_type == "extreme_high":
                df.at[idx, "pm2_5"] = float(np.random.uniform(1200.0, 5000.0))
            elif anomaly_type == "negative":
                df.at[idx, "no2"] = float(np.random.uniform(-50.0, -1.0))
            elif anomaly_type == "stuck_sensor":
                df.at[idx, "co"] = 99.99

        # 3. Inject duplicate records (approx 1.5%)
        dup_indices = np.random.choice(n, size=max(1, int(n * 0.015)), replace=False)
        duplicates = df.iloc[dup_indices].copy()
        # Optionally add slight noise to timestamp format in some duplicates to test normalization
        df = pd.concat([df, duplicates], ignore_index=True)

        return df

    def _inject_weather_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject missing values and occasional extreme readings in weather data."""
        n = len(df)
        if n == 0:
            return df

        df = df.copy()
        # Missing values (approx 2%)
        for col in ["temperature", "humidity", "wind_speed", "pressure", "rainfall"]:
            mask = np.random.random(n) < 0.02
            df.loc[mask, col] = np.nan

        # Duplicate records (approx 1%)
        dup_indices = np.random.choice(n, size=max(1, int(n * 0.01)), replace=False)
        duplicates = df.iloc[dup_indices].copy()
        df = pd.concat([df, duplicates], ignore_index=True)
        return df

    def save_datasets(
        self,
        aq_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        stations_df: pd.DataFrame,
        output_dir: str = "data/raw",
    ) -> Dict[str, str]:
        """Save generated dataframes to CSV files."""
        out_path = ensure_dir(output_dir)
        aq_path = out_path / "air_quality.csv"
        weather_path = out_path / "weather.csv"
        stations_path = out_path / "stations.csv"

        aq_df.to_csv(aq_path, index=False)
        weather_df.to_csv(weather_path, index=False)
        stations_df.to_csv(stations_path, index=False)

        logger.info(f"Saved datasets to {output_dir}: AQ={len(aq_df)} rows, Weather={len(weather_df)} rows, Stations={len(stations_df)} rows")
        return {
            "air_quality": str(aq_path),
            "weather": str(weather_path),
            "stations": str(stations_path),
        }

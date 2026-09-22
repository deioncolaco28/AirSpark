# AirSpark Data Dictionary

This document provides complete schema definitions, descriptions, physical units, data types, and valid bounds for all datasets ingested and produced by AirSpark.

---

## 1. Raw Air Quality Dataset (`data/raw/air_quality.csv`)

| Field Name | Type | Physical Unit | Description | Valid Physical Range |
| :--- | :--- | :--- | :--- | :--- |
| `station_id` | String | Identifier | Unique alphanumeric station identifier (e.g. `STN_001`) | Non-null |
| `timestamp` | String / Timestamp | ISO-8601 | Observation timestamp in UTC | Valid date string |
| `pm2_5` | Double | $\mu\text{g/m}^3$ | Fine Particulate Matter ($<2.5\ \mu\text{m}$) | $0.0 - 1000.0$ |
| `pm10` | Double | $\mu\text{g/m}^3$ | Coarse Particulate Matter ($<10\ \mu\text{m}$) | $0.0 - 1500.0$ |
| `no2` | Double | $\text{ppb}$ | Nitrogen Dioxide concentration | $0.0 - 2000.0$ |
| `so2` | Double | $\text{ppb}$ | Sulfur Dioxide concentration | $0.0 - 2000.0$ |
| `co` | Double | $\text{ppm}$ | Carbon Monoxide concentration | $0.0 - 100.0$ |
| `o3` | Double | $\text{ppb}$ | Ground-level Ozone concentration | $0.0 - 500.0$ |

---

## 2. Raw Weather Dataset (`data/raw/weather.csv`)

| Field Name | Type | Physical Unit | Description | Valid Physical Range |
| :--- | :--- | :--- | :--- | :--- |
| `station_id` | String | Identifier | Unique station identifier linking to Air Quality | Non-null |
| `timestamp` | String / Timestamp | ISO-8601 | Timestamp of meteorological recording | Valid date string |
| `temperature` | Double | $^{\circ}\text{C}$ | Ambient ambient dry-bulb temperature | $-50.0 - 60.0$ |
| `humidity` | Double | $\%$ | Relative Humidity | $0.0 - 100.0$ |
| `wind_speed` | Double | $\text{m/s}$ | Surface wind speed | $0.0 - 150.0$ |
| `wind_direction` | Double | Degrees ($^{\circ}$) | Wind direction angle ($0^{\circ}-360^{\circ}$) | $0.0 - 360.0$ |
| `pressure` | Double | $\text{hPa}$ | Atmospheric barometric pressure | $800.0 - 1100.0$ |
| `rainfall` | Double | $\text{mm}$ | Precipitation / Rainfall accumulation | $0.0 - 500.0$ |

---

## 3. Station Metadata (`data/raw/stations.csv`)

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `station_id` | String | Unique station identifier (Primary Key) |
| `station_name` | String | Descriptive monitoring station name |
| `city` | String | Associated municipality or city |
| `state` | String | State / Regional province |
| `latitude` | Double | WGS84 Geographic Latitude (Decimal degrees) |
| `longitude` | Double | WGS84 Geographic Longitude (Decimal degrees) |
| `elevation` | Double | Station elevation above sea level in meters |
| `station_type` | String | Classification: `Traffic`, `Industrial`, `Residential`, `Commercial`, `Coastal` |

---

## 4. Processed Integrated Dataset (`data/processed/integrated_aqi.parquet`)

Includes all columns from above plus computed analytics:

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `sub_index_pm2_5` | Double | Piecewise linear sub-index for $PM_{2.5}$ |
| `sub_index_pm10` | Double | Piecewise linear sub-index for $PM_{10}$ |
| `sub_index_no2` | Double | Piecewise linear sub-index for $NO_2$ |
| `sub_index_so2` | Double | Piecewise linear sub-index for $SO_2$ |
| `sub_index_co` | Double | Piecewise linear sub-index for $CO$ |
| `sub_index_o3` | Double | Piecewise linear sub-index for $O_3$ |
| `aqi` | Double | Overall computed Air Quality Index ($\max(I_p)$) |
| `dominant_pollutant` | String | Pollutant responsible for highest sub-index |
| `aqi_category` | String | Categorization: `Good`, `Moderate`, `Unhealthy for Sensitive Groups`, `Unhealthy`, `Very Unhealthy`, `Hazardous` |
| `pm_ratio` | Double | Ratio of $PM_{2.5} / PM_{10}$ |
| `anomaly_classification` | String | `NORMAL`, `ACUTE_POLLUTION_EVENT`, or `SENSOR_MALFUNCTION_SUSPECT` |
| `is_anomaly` | Boolean | True if classified as any non-normal reading |
| `year`, `month`, `hour` | Integer | Partition and temporal feature keys |

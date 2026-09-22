# AirSpark Data Dictionary & Schema Definitions

This document provides complete schema definitions, descriptions, physical units, data types, valid bounds, and academic data provenance classifications for all datasets ingested and produced by AirSpark across both the **Controlled Development Dataset** and the **India-Scale Experimental Dataset**.

---

## 1. Academic Data Provenance Classification

AirSpark categorizes all analytical data into three explicit tiers:

| Data Category | Component Fields / Assets | Provenance Source & Nature |
| :--- | :--- | :--- |
| **1. Authentic Source-Derived Metadata** | `station_id`, `station_name`, `city`, `state`, `latitude`, `longitude`, `elevation`, `station_type` | Sourced from official CPCB / CAAQMS monitoring station registries (44 representative stations across 18 States/UTs and 31 cities). |
| **2. Controlled Synthetic Observations** | `pm2_5`, `pm10`, `no2`, `so2`, `co`, `o3`, `temperature`, `humidity`, `wind_speed`, `wind_direction`, `pressure`, `rainfall` | Physics-informed numerical simulation incorporating regional baselines, diurnal boundary layer cycles, photochemical ozone formation, and realistic sensor anomalies (~32,155 AQ records, ~31,996 weather records). |
| **3. Derived Analytics** | `aqi`, `extrapolated_aqi`, `sub_index_*`, `dominant_pollutant`, `hotspot_score`, `hotspot_level`, `prediction_error`, ML models, streaming alerts | Distributed analytics calculated dynamically by AirSpark Spark processing engine. |

> **Plug-and-Play Capability:** The schema definitions below are strictly aligned with public continuous CAAQMS data formats, enabling real observation files to be substituted without modifying pipeline logic.

---

## 2. Raw Air Quality Dataset (`data/raw/air_quality.csv` & `data/raw_india/air_quality.csv`)

| Field Name | Type | Physical Unit | Description | Valid Physical Range | Nature |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `station_id` | String | Identifier | Unique alphanumeric station identifier (e.g. `STN_001` or `IND_DL_001`) | Non-null | Authentic Metadata |
| `timestamp` | String / Timestamp | ISO-8601 | Observation timestamp in UTC / Local Normalized Time | Valid date string | Temporal Index |
| `pm2_5` | Double | $\mu\text{g/m}^3$ | Fine Particulate Matter ($<2.5\ \mu\text{m}$) | $0.0 - 1000.0$ | Controlled Simulated Observation |
| `pm10` | Double | $\mu\text{g/m}^3$ | Coarse Particulate Matter ($<10\ \mu\text{m}$) | $0.0 - 1500.0$ | Controlled Simulated Observation |
| `no2` | Double | $\text{ppb}$ | Nitrogen Dioxide concentration | $0.0 - 2000.0$ | Controlled Simulated Observation |
| `so2` | Double | $\text{ppb}$ | Sulfur Dioxide concentration | $0.0 - 2000.0$ | Controlled Simulated Observation |
| `co` | Double | $\text{ppm}$ | Carbon Monoxide concentration | $0.0 - 100.0$ | Controlled Simulated Observation |
| `o3` | Double | $\text{ppb}$ | Ground-level Ozone concentration | $0.0 - 500.0$ | Controlled Simulated Observation |

---

## 3. Raw Weather Dataset (`data/raw/weather.csv` & `data/raw_india/weather.csv`)

| Field Name | Type | Physical Unit | Description | Valid Physical Range | Nature |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `station_id` | String | Identifier | Unique station identifier linking to Air Quality observations | Non-null | Authentic Metadata |
| `timestamp` | String / Timestamp | ISO-8601 | Timestamp of meteorological recording | Valid date string | Temporal Index |
| `temperature` | Double | $^{\circ}\text{C}$ | Ambient dry-bulb temperature | $-50.0 - 60.0$ | Controlled Simulated Observation |
| `humidity` | Double | $\%$ | Relative Humidity | $0.0 - 100.0$ | Controlled Simulated Observation |
| `wind_speed` | Double | $\text{m/s}$ | Surface wind speed | $0.0 - 150.0$ | Controlled Simulated Observation |
| `wind_direction` | Double | Degrees ($^{\circ}$) | Wind direction angle ($0^{\circ}-360^{\circ}$) | $0.0 - 360.0$ | Controlled Simulated Observation |
| `pressure` | Double | $\text{hPa}$ | Atmospheric barometric pressure | $800.0 - 1100.0$ | Controlled Simulated Observation |
| `rainfall` | Double | $\text{mm}$ | Precipitation / Rainfall accumulation | $0.0 - 500.0$ | Controlled Simulated Observation |

---

## 4. Station Metadata (`data/raw/stations.csv` & `data/raw_india/stations.csv`)

| Field Name | Type | Description | Nature |
| :--- | :--- | :--- | :--- |
| `station_id` | String | Unique station identifier (Primary Key, e.g. `IND_DL_001`, `IND_MH_001`) | Authentic CPCB Metadata |
| `station_name` | String | Official descriptive monitoring station name | Authentic CPCB Metadata |
| `city` | String | Associated municipality or city (e.g. `Delhi`, `Mumbai`, `Bengaluru`) | Authentic CPCB Metadata |
| `state` | String | State / Union Territory (e.g. `Delhi`, `Maharashtra`, `Karnataka`) | Authentic CPCB Metadata |
| `latitude` | Double | WGS84 Geographic Latitude (Decimal degrees, bounds $[6.0, 38.0]$ for India) | Authentic CPCB Metadata |
| `longitude` | Double | WGS84 Geographic Longitude (Decimal degrees, bounds $[68.0, 98.0]$ for India) | Authentic CPCB Metadata |
| `elevation` | Double | Station elevation above sea level in meters | Authentic CPCB Metadata |
| `station_type` | String | Classification: `Traffic`, `Industrial`, `Residential`, `Commercial`, `Coastal` | Authentic CPCB Metadata |

---

## 5. Processed Integrated Dataset (`data/processed/integrated_aqi.parquet` & `data/processed_india/integrated_aqi.parquet`)

Includes all source fields plus distributed computed analytical attributes:

| Field Name | Type | Description | Nature |
| :--- | :--- | :--- | :--- |
| `sub_index_pm2_5` | Double | Piecewise linear sub-index for $PM_{2.5}$ per EPA / CPCB breakpoints | Derived Analytics |
| `sub_index_pm10` | Double | Piecewise linear sub-index for $PM_{10}$ | Derived Analytics |
| `sub_index_no2` | Double | Piecewise linear sub-index for $NO_2$ | Derived Analytics |
| `sub_index_so2` | Double | Piecewise linear sub-index for $SO_2$ | Derived Analytics |
| `sub_index_co` | Double | Piecewise linear sub-index for $CO$ | Derived Analytics |
| `sub_index_o3` | Double | Piecewise linear sub-index for $O_3$ | Derived Analytics |
| `aqi` | Double | Overall computed standard-compliant Air Quality Index ($\max(I_p)$, bounded $[0, 500]$) | Derived Analytics |
| `extrapolated_aqi` | Double | Piecewise research extrapolated AQI for acute exceedance tracking without upper truncation | Derived Analytics |
| `is_aqi_out_of_range` | Boolean | True if concentration exceeds standard maximum breakpoint ceiling | Derived Analytics |
| `aqi_exceedance_flag` | Boolean | Exceedance flag preserved for scientific audit | Derived Analytics |
| `dominant_pollutant` | String | Driving pollutant responsible for highest sub-index | Derived Analytics |
| `aqi_category` | String | Categorization per standard: `Good`, `Satisfactory`, `Moderate`, `Poor`, `Very Poor`, `Severe` (CPCB) or EPA categories | Derived Analytics |
| `pm_ratio` | Double | Ratio of $PM_{2.5} / PM_{10}$ | Derived Analytics |
| `anomaly_classification` | String | `NORMAL`, `ACUTE_POLLUTION_EVENT`, or `SENSOR_MALFUNCTION_SUSPECT` | Derived Analytics |
| `is_anomaly` | Boolean | True if classified as non-normal reading | Derived Analytics |
| `year`, `month` | Integer | Spark Parquet lake partition keys | Analytical Partition |

---

## 6. Provenance & Compliance (`data/raw_india/provenance.json`)
- **Authority:** Central Pollution Control Board (CPCB) & Ministry of Environment, Forest and Climate Change (MoEFCC).
- **Standards:** National Ambient Air Quality Standards (NAAQS) & National Air Quality Index (NAQI).
- **Coordinate Reference System:** WGS84 (EPSG:4326).
- **License:** Open Government Data License - India (OGDL) / Academic & Research Framework.


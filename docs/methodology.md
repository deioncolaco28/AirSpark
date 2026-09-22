# AirSpark Research Methodology & Academic Framework

## 1. Problem Statement & Research Gap Alignment

Existing literature in atmospheric monitoring often addresses air-quality prediction, IoT sensing, distributed data processing, sensor data quality, real-time alert systems, or spatial-temporal analysis in isolation. 

**AirSpark** provides an integrated Big Data Analytics (BDA) framework combining all 10 core dimensions:

| Literature Research Gap | AirSpark System Component | Empirical Metric / Verification |
| :--- | :--- | :--- |
| **1. Heterogeneous Data Fusion** | Multi-source ingestion (`app/ingestion/` & `joins.py`) | 100% matched join cardinality across air quality, weather, and station metadata |
| **2. Multi-Tier Nationwide Scale** | Dual Dataset Tiers (`app/ingestion/india_dataset.py`) | Controlled 10-station dev dataset + 44-station India-scale experimental dataset (CPCB metadata + controlled simulation) |
| **3. Data Quality & Sensor Anomalies** | Robust DQ Pipeline (`app/quality/`) | Data retention %, audit score, missing value imputation, duplicate elimination |
| **4. Scalable Distributed Processing** | Apache Spark & PySpark Engine (`app/processing/`) | Linear throughput scaling across Small, Medium, and Large datasets in local Spark mode |
| **5. Comprehensive Batch Processing** | End-to-end ETL & Parquet Storage | Sub-minute execution on tens of thousands of records with partition pruning |
| **6. Real-Time Stream Processing** | Spark Structured Streaming (`app/streaming/`) | Sub-second micro-batch processing latency & sliding window aggregations |
| **7. Standard-Compliant AQI Analysis** | Configurable Breakpoint Engine (`app/aqi/`) | Boundary-accurate US EPA & CPCB sub-index calculations and dominant pollutant detection |
| **8. Hierarchical Spatial & Hotspots** | Location Explorer & Hotspot Severity Index (`app/analytics/`) | Location-aware filtering (Country ➔ State ➔ City ➔ Station) and persistent hotspot categorization |
| **9. Temporal & Diurnal Analytics** | Time-series Aggregation Engine (`temporal.py`) | Location-filtered diurnal 24-hr profiles, daily rolling averages, and seasonal shifts |
| **10. Interactive Visual Analytics** | Streamlit & Plotly MapLibre Dashboard (`dashboard/`) | Real-time interactive KPIs, maps, correlation matrices, and alert monitoring |

---

## 2. Academic Data Provenance & Experimental Tiers

AirSpark enforces a rigorous, transparent data provenance model:

1. **Authentic Source-Derived Metadata:**
   - Station identity, official names, states, cities, coordinates (WGS84), elevations, and station classifications are derived directly from official CPCB / CAAQMS monitoring station registries across 18 States/UTs and 31 cities (44 representative stations).
2. **Controlled Synthetic Observations:**
   - Multi-pollutant (PM2.5, PM10, NO2, SO2, CO, O3) and meteorological time-series (temperature, humidity, wind, pressure, rainfall) are generated via physics-informed numerical models (diurnal boundary layer dynamics, photochemical reactions, dispersion, and regional baselines) with controlled sensor quality artifacts (~2.8% missing, ~1.5% duplicates, <0.5% domain limits).
3. **Derived Analytics:**
   - Standard-compliant calculated AQI ($[0, 500]$), pollutant sub-indices, Hotspot Severity Indices (HSI), temporal/spatial aggregations, Spark ML regression forecasts, and real-time streaming alerts.

> **Plug-and-Play Architecture:** The ingestion and analytics schema is structurally identical to official CPCB CAAQMS hourly data releases. As continuous monitoring datasets become available, raw observation CSVs can be swapped without modifying ETL scripts, calculation engines, or downstream machine learning pipelines.

---

## 3. Mathematical Formulations

### 2.1 Piecewise Linear AQI Sub-Index Calculation
For any given pollutant concentration $C$, the sub-index $I$ is calculated as:

$$I = \frac{I_{high} - I_{low}}{BP_{high} - BP_{low}} \times (C - BP_{low}) + I_{low}$$

Where:
- $C$: Truncated pollutant concentration
- $BP_{high}, BP_{low}$: Breakpoint concentration brackets encompassing $C$
- $I_{high}, I_{low}$: AQI index brackets corresponding to $BP_{high}, BP_{low}$

The overall Air Quality Index is determined by the maximum sub-index:
$$AQI = \max_{p \in \mathcal{P}} (I_p)$$
Where $\mathcal{P} = \{PM_{2.5}, PM_{10}, NO_2, SO_2, CO, O_3\}$.

Standard regulatory AQI is constrained to $[0, 500]$. Concentrations exceeding the maximum breakpoint are preserved via non-destructive exceedance flags (`is_aqi_out_of_range = True`) and research extrapolated AQI ($I_{extrapolated}$).

### 2.2 Hotspot Severity Index (HSI)
To quantify persistent spatial pollution exposure:

$$HSI = (\text{UnhealthyHoursPct} \times 0.6) + \left(\frac{\overline{AQI}}{500} \times 100 \times 0.4\right)$$

- $\text{UnhealthyHoursPct}$: Percentage of recorded hours with $AQI \ge 150$ (or $\ge 200$ under CPCB)
- $\overline{AQI}$: Station mean AQI
- Categorization:
  - $HSI \ge 60$: **Critical Hotspot**
  - $35 \le HSI < 60$: **Moderate Hotspot**
  - $15 \le HSI < 35$: **Low Hotspot**
  - $HSI < 15$: **Clean / Compliant**

### 2.3 Hierarchical Location Filtering Resolution
To enable seamless exploration across Indian geography:
$$D_{filtered} = \sigma_{\text{state}=s \land \text{city}=c \land \text{station\_id}=st}(D_{integrated})$$
All summary KPIs, daily trends, diurnal hourly curves, and spatial maps dynamically recalculate over the subset $D_{filtered}$.

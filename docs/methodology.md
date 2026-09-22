# AirSpark Research Methodology & Academic Framework

## 1. Problem Statement & Research Gap Alignment

Existing literature in atmospheric monitoring often addresses air-quality prediction, IoT sensing, distributed data processing, sensor data quality, real-time alert systems, or spatial-temporal analysis in isolation. 

**AirSpark** provides an integrated Big Data Analytics (BDA) framework combining all 10 core dimensions:

| Literature Research Gap | AirSpark System Component | Empirical Metric / Verification |
| :--- | :--- | :--- |
| **1. Heterogeneous Data Fusion** | Multi-source ingestion (`app/ingestion/` & `joins.py`) | 100% matched join cardinality across air quality, weather, and station metadata |
| **2. Data Quality & Sensor Anomalies** | Robust DQ Pipeline (`app/quality/`) | Data retention %, audit score, missing value imputation, duplicate elimination |
| **3. Scalable Distributed Processing** | Apache Spark & PySpark Engine (`app/processing/`) | Linear throughput scaling across Small, Medium, and Large datasets |
| **4. Comprehensive Batch Processing** | End-to-end ETL & Parquet Storage | Sub-minute execution on millions of records with partition pruning |
| **5. Real-Time Stream Processing** | Spark Structured Streaming (`app/streaming/`) | Sub-second micro-batch processing latency & sliding window aggregations |
| **6. Standard-Compliant AQI Analysis** | Configurable Breakpoint Engine (`app/aqi/`) | Boundary-accurate US EPA & CPCB sub-index calculations and dominant pollutant detection |
| **7. Spatial Analytics & Hotspots** | Spatial Aggregator & Hotspot Severity Index (`app/analytics/`) | Geographic rankings, station dispersion, and persistent hotspot categorization |
| **8. Temporal & Diurnal Analytics** | Time-series Aggregation Engine (`temporal.py`) | Diurnal 24-hr profiles, daily rolling averages, and seasonal shifts |
| **9. Interactive Visual Analytics** | Streamlit & Plotly Dashboard (`dashboard/`) | Real-time interactive KPIs, maps, correlation matrices, and alert monitoring |
| **10. Systematic Performance Evaluation**| Benchmark Runner (`app/performance/`) | Measured processing time, throughput (records/sec), and partition scalability |

---

## 2. Mathematical Formulations

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

### 2.2 Hotspot Severity Index (HSI)
To quantify persistent spatial pollution exposure:

$$HSI = (\text{UnhealthyHoursPct} \times 0.6) + \left(\frac{\overline{AQI}}{500} \times 100 \times 0.4\right)$$

- $\text{UnhealthyHoursPct}$: Percentage of recorded hours with $AQI \ge 150$
- $\overline{AQI}$: Station mean AQI
- Categorization:
  - $HSI \ge 60$: **Critical Hotspot**
  - $35 \le HSI < 60$: **Moderate Hotspot**
  - $15 \le HSI < 35$: **Low Hotspot**
  - $HSI < 15$: **Clean / Compliant**

### 2.3 Explainable Anomaly Classification
- **Jump Ratio**: $R_t = \frac{C_t}{C_{t-1}}$
- **Classification Logic**:
  - If $C_t > 150 \ \mu\text{g/m}^3 \land R_t > 3.0 \land (\text{Secondary pollutants not elevated}) \implies$ **Sensor Malfunction Suspect**
  - If $C_t > 150 \ \mu\text{g/m}^3 \land (\text{Multi-pollutant co-elevation}) \implies$ **Acute Pollution Event**
  - Otherwise $\implies$ **Normal Reading**

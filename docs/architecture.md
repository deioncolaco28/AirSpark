# AirSpark Architectural Blueprint

AirSpark is an integrated Big Data Analytics (BDA) framework designed to ingest, clean, integrate, compute, analyze, and visualize high-velocity, multi-source spatio-temporal air quality and meteorological data using Apache Spark and PySpark.

---

## 1. System Architecture Overview

```mermaid
flowchart TD
    subgraph Data Sources
        DS1[Air Quality Sensors]
        DS2[Meteorological Stations]
        DS3[Station Geo-Metadata]
        DS4[Streaming Simulated Feed]
    end

    subgraph Ingestion Layer
        IN1[File Adapter: CSV / Parquet]
        IN2[Structured Streaming Source]
    end

    subgraph Data Quality Engine
        DQ1[Schema Validation]
        DQ2[Deduplication: station_id + timestamp]
        DQ3[Timestamp Normalization]
        DQ4[Domain Limits & Physical Capping]
        DQ5[Temporal Window Missing Value Imputation]
        DQ6[Statistical IQR Outlier Flagging]
    end

    subgraph Distributed Processing & Analytics Layer
        SP1[Spark Session & Cluster Manager]
        SP2[Temporal & Spatial Joins]
        SP3[Distributed AQI Engine: US EPA / CPCB]
        SP4[Temporal Analytics: Hourly, Daily, Monthly, Seasonal]
        SP5[Spatial Analytics & Hotspot Severity Scoring]
        SP6[Weather Correlation Matrix Engine]
        SP7[Explainable Anomaly Detection Layer]
        SP8[Spark ML Regression Forecasting]
    end

    subgraph Storage & Sinks
        ST1[(Partitioned Parquet Lake: year/month)]
        ST2[Streaming Memory / File Sinks]
        ST3[JSON Metadata & Reports]
    end

    subgraph Presentation & Evaluation
        UI1[Streamlit Interactive Dashboard]
        UI2[Plotly Geospatial Map Views]
        UI3[Real-time Threshold Alert Feed]
        UI4[Scalability Benchmark Evaluation]
    end

    DS1 --> IN1
    DS2 --> IN1
    DS3 --> IN1
    DS4 --> IN2

    IN1 --> DQ1
    IN2 --> DQ1
    DQ1 --> DQ2 --> DQ3 --> DQ4 --> DQ5 --> DQ6

    DQ6 --> SP1
    SP1 --> SP2
    SP2 --> SP3
    SP3 --> SP4
    SP3 --> SP5
    SP3 --> SP6
    SP3 --> SP7
    SP3 --> SP8

    SP4 --> ST1
    SP5 --> ST1
    SP6 --> ST3
    SP7 --> ST1
    IN2 --> ST2

    ST1 --> UI1
    ST3 --> UI1
    ST2 --> UI3
    SP8 --> UI1
    SP1 --> UI4
```

---

## 2. Core Architectural Components

### 2.1 Ingestion Layer
- **Batch Ingestion Engine**: Dynamically ingests raw CSV and Parquet files into Spark DataFrames with explicit `StructType` schemas to avoid runtime schema inference overhead.
- **Structured Streaming Source**: Utilizes Spark's `readStream` micro-batch mechanism to monitor incoming sensor data with configurable trigger frequencies.

### 2.2 Data Quality & ETL Pipeline
- **Validation**: Enforces mandatory primary keys (`station_id`, `timestamp`) and numeric bounds.
- **Deduplication**: Eliminates duplicate observations based on business keys.
- **Normalization**: Handles diverse timestamp formats (ISO-8601, RFC-3339, standard SQL) into UTC Spark `TimestampType`.
- **Imputation**: Two-stage windowed temporal forward-fill followed by station-level and global median fallbacks.
- **Outlier Flagging**: Employs non-destructive statistical IQR and Z-Score flagging, ensuring legitimate extreme pollution events are preserved for scientific analysis.

### 2.3 Distributed AQI Calculation Engine
- Evaluates individual pollutant sub-indices for $PM_{2.5}, PM_{10}, NO_2, SO_2, CO, O_3$ using piecewise linear interpolation against official standards (US EPA and Indian CPCB).
- Computes overall AQI as the upper supremum ($AQI = \max(I_p)$) and identifies the instantaneous dominant pollutant.

### 2.4 Spatio-Temporal Analytics Layer
- **Temporal**: Diurnal 24-hour cycle aggregation, daily summary statistics, seasonal variation analysis, and rolling-window moving averages.
- **Spatial**: Station ranking, geographic aggregation, and **Hotspot Severity Scoring** based on persistent exposure duration and peak pollution loads.
- **Correlations**: Spark DataFrame Pearson correlation matrix between meteorological conditions and atmospheric pollutants.
- **Explainable Anomalies**: Evaluates temporal jump ratios and multi-pollutant cross-correlations to classify anomalies into sensor malfunctions vs acute atmospheric pollution surges.

### 2.5 Storage & Partitioning Strategy
- Writes processed datasets to Apache Parquet with Snappy compression.
- Partitions large-scale time series datasets hierarchically by `year` and `month` to optimize query pruning during spatial-temporal slicing.

"""
Unit tests for the Data Quality module (Imputation, Normalization, Deduplication, Outliers).
"""
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from app.quality.normalization import Normalizer
from app.quality.missing_values import MissingValueHandler
from app.quality.outliers import OutlierDetector
from app.quality.quality_pipeline import DataQualityPipeline


def test_timestamp_normalization(spark_session: SparkSession):
    data = [
        ("STN_001", "2026-01-01 10:00:00"),
        ("STN_001", "2026-01-01T11:00:00"),
        ("STN_001", "2026-01-01"),
        ("STN_001", "invalid_timestamp"),
    ]
    df = spark_session.createDataFrame(data, ["station_id", "timestamp"])
    norm_df = Normalizer.normalize_timestamps(df, "timestamp")

    valid_rows = norm_df.filter(F.col("timestamp").isNotNull()).count()
    invalid_rows = norm_df.filter(F.col("timestamp").isNull()).count()

    assert valid_rows == 3
    assert invalid_rows == 1


def test_missing_value_imputation(spark_session: SparkSession):
    data = [
        ("STN_001", "2026-01-01 00:00:00", 20.0),
        ("STN_001", "2026-01-01 01:00:00", None),   # Should be forward-filled with 20.0
        ("STN_001", "2026-01-01 02:00:00", 30.0),
    ]
    df = spark_session.createDataFrame(data, ["station_id", "timestamp", "pm2_5"])
    df = Normalizer.normalize_timestamps(df, "timestamp")
    
    imputed_df = MissingValueHandler.impute_missing(df, numeric_cols=["pm2_5"])
    results = imputed_df.orderBy("timestamp").collect()

    assert results[1]["pm2_5"] == 20.0
    assert results[2]["pm2_5"] == 30.0


def test_domain_limits_capping(spark_session: SparkSession):
    data = [
        ("STN_001", "2026-01-01 00:00:00", -15.0),    # Negative value
        ("STN_001", "2026-01-01 01:00:00", 2500.0),   # Extreme impossible value
        ("STN_001", "2026-01-01 02:00:00", 45.0),     # Normal value
    ]
    df = spark_session.createDataFrame(data, ["station_id", "timestamp", "pm2_5"])
    
    domain_limits = {"pm2_5": [0.0, 1000.0]}
    capped_df, violations = OutlierDetector.audit_and_handle_domain_limits(df, domain_limits, action="cap")

    rows = capped_df.orderBy("timestamp").collect()
    assert rows[0]["pm2_5"] == 0.0
    assert rows[1]["pm2_5"] == 1000.0
    assert rows[2]["pm2_5"] == 45.0
    assert violations["pm2_5"] == 2


def test_full_quality_pipeline_deduplication(spark_session: SparkSession):
    data = [
        ("STN_001", "2026-01-01 00:00:00", 25.0, 50.0, 10.0, 5.0, 0.5, 20.0),
        ("STN_001", "2026-01-01 00:00:00", 25.0, 50.0, 10.0, 5.0, 0.5, 20.0),  # Duplicate
        ("STN_001", "2026-01-01 01:00:00", 30.0, 60.0, 12.0, 6.0, 0.6, 22.0),
    ]
    df = spark_session.createDataFrame(data, ["station_id", "timestamp", "pm2_5", "pm10", "no2", "so2", "co", "o3"])
    
    dq = DataQualityPipeline()
    cleaned_df, report = dq.process_air_quality(df)

    assert report["initial_records"] == 3
    assert report["duplicate_records"] == 1
    assert report["final_records"] == 2


def test_statistical_outlier_detection_and_counting(spark_session: SparkSession):
    # Construct 20 records where 2 records are clear statistical outliers
    data = []
    for i in range(18):
        data.append(("STN_001", f"2026-01-01 {i:02d}:00:00", 25.0 + (i % 3)))
    # Add 2 extreme statistical outliers
    data.append(("STN_001", "2026-01-01 18:00:00", 350.0))
    data.append(("STN_001", "2026-01-01 19:00:00", 400.0))
    
    df = spark_session.createDataFrame(data, ["station_id", "timestamp", "pm2_5"])
    flagged_df, outlier_report = OutlierDetector.detect_iqr_outliers(df, ["pm2_5"], factor=1.5)
    
    # Verify flags exist
    assert "is_pm2_5_outlier" in flagged_df.columns
    assert "is_any_outlier" in flagged_df.columns
    
    # Verify count is real and non-zero (CRITICAL FIX FOR ISSUE #1)
    assert outlier_report["total_outlier_rows"] >= 2
    assert outlier_report["pm2_5"] >= 2
    assert flagged_df.filter(F.col("is_any_outlier") == True).count() >= 2


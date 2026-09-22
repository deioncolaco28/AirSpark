"""
Unit tests for Temporal and Spatial analytics engines.
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from app.analytics.temporal import TemporalAnalytics
from app.analytics.spatial import SpatialAnalytics
from app.analytics.hotspots import HotspotAnalyzer


def test_hourly_and_daily_temporal_analytics(spark_session: SparkSession):
    data = [
        ("STN_001", "2026-01-01 08:00:00", 60.0, 100.0),
        ("STN_001", "2026-01-01 08:30:00", 80.0, 120.0),
        ("STN_001", "2026-01-01 14:00:00", 40.0, 70.0),
        ("STN_001", "2026-01-02 08:00:00", 50.0, 90.0),
    ]
    df = spark_session.createDataFrame(data, ["station_id", "timestamp", "pm2_5", "aqi"])
    df = df.withColumn("timestamp", F.to_timestamp("timestamp"))

    # Hourly diurnal test
    hourly_df = TemporalAnalytics.compute_hourly_profile(df)
    hourly_rows = hourly_df.collect()
    assert len(hourly_rows) == 2  # Hours 8 and 14

    # Daily aggregation test
    daily_df = TemporalAnalytics.compute_daily_analytics(df)
    daily_rows = daily_df.collect()
    assert len(daily_rows) == 2  # Jan 1 and Jan 2


def test_spatial_summary_and_hotspots(spark_session: SparkSession):
    data = [
        ("STN_001", "Central Delhi", "Delhi", 28.61, 77.20, "2026-01-01 00:00:00", 180.0),
        ("STN_001", "Central Delhi", "Delhi", 28.61, 77.20, "2026-01-01 01:00:00", 220.0),
        ("STN_002", "South Bangalore", "Bengaluru", 12.97, 77.59, "2026-01-01 00:00:00", 45.0),
        ("STN_002", "South Bangalore", "Bengaluru", 12.97, 77.59, "2026-01-01 01:00:00", 55.0),
    ]
    df = spark_session.createDataFrame(data, ["station_id", "station_name", "city", "latitude", "longitude", "timestamp", "aqi"])
    df = df.withColumn("timestamp", F.to_timestamp("timestamp"))

    station_summary = SpatialAnalytics.compute_station_summary(df)
    station_rows = station_summary.collect()
    assert len(station_rows) == 2
    # STN_001 should rank higher (mean_aqi = 200.0)
    assert station_rows[0]["station_id"] == "STN_001"
    assert station_rows[0]["mean_aqi"] == 200.0

    hotspots = HotspotAnalyzer.identify_hotspots(df, unhealthy_threshold=150.0)
    hotspot_rows = hotspots.collect()
    assert hotspot_rows[0]["station_id"] == "STN_001"
    assert hotspot_rows[0]["unhealthy_pct"] == 100.0
    assert hotspot_rows[0]["hotspot_score"] > 60.0

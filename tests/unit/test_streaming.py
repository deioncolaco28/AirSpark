"""
Unit tests for Spark Structured Streaming and Alert Engine.
Validates real-time threshold detection, PM2.5 spike alerts, rapid surge detection, and window aggregations.
"""
from datetime import datetime, timezone
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from app.streaming.alerts import AlertEvaluator
from app.streaming.pipeline import StreamingPipeline


def test_alert_engine_detections():
    evaluator = AlertEvaluator()
    
    # 1. Unhealthy AQI Alert (AQI >= 150)
    rec_unhealthy = {
        "station_id": "STN_001",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aqi": 165.0,
        "pm2_5": 85.0
    }
    alert = evaluator.evaluate_record(rec_unhealthy)
    assert alert is not None
    assert alert["alert_type"] == "UNHEALTHY_AQI_ALERT"
    
    # 2. Hazardous AQI Alert (AQI >= 300)
    rec_hazard = {
        "station_id": "STN_001",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aqi": 320.0,
        "pm2_5": 280.0
    }
    alert_h = evaluator.evaluate_record(rec_hazard)
    assert alert_h is not None
    assert alert_h["alert_type"] == "HAZARDOUS_AQI_ALERT"
    
    # 3. Rapid Surge Alert (+50% increase)
    # First record
    evaluator.evaluate_record({
        "station_id": "STN_002",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aqi": 50.0,
        "pm2_5": 20.0
    })
    # Subsequent record with sudden jump from 20 -> 50 (+150%)
    alert_surge = evaluator.evaluate_record({
        "station_id": "STN_002",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aqi": 80.0,
        "pm2_5": 50.0
    })
    assert alert_surge is not None
    assert alert_surge["alert_type"] == "RAPID_CHANGE_ALERT"


def test_streaming_window_aggregation(spark_session: SparkSession):
    data = [
        ("STN_001", "2026-01-01 00:00:00", 25.0, 50.0, 10.0, 5.0, 0.5, 20.0, 50.0),
        ("STN_001", "2026-01-01 00:04:00", 35.0, 70.0, 12.0, 6.0, 0.6, 22.0, 70.0),
        ("STN_001", "2026-01-01 00:08:00", 45.0, 90.0, 15.0, 8.0, 0.8, 25.0, 90.0),
    ]
    df = spark_session.createDataFrame(
        data, 
        ["station_id", "timestamp", "pm2_5", "pm10", "no2", "so2", "co", "o3", "aqi"]
    )
    df = df.withColumn("timestamp", F.to_timestamp("timestamp"))
    
    pipeline = StreamingPipeline(spark_session)
    windowed_df = pipeline.apply_window_aggregation(df, window_duration="10 minutes", slide_duration="5 minutes")
    
    res = windowed_df.collect()
    assert len(res) > 0
    assert "avg_aqi" in windowed_df.columns
    assert "max_pm2_5" in windowed_df.columns
    assert "record_count" in windowed_df.columns

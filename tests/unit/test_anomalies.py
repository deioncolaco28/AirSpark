"""
Unit tests for the explainable anomaly detection and classification engine.
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from app.analytics.anomalies import AnomalyDetector


def test_anomaly_classification(spark_session: SparkSession):
    data = [
        # Normal baseline
        ("STN_001", "2026-01-01 00:00:00", 30.0, 50.0, 15.0, 0.8),
        # Correlated acute pollution event (both PM2.5 and PM10 surge together)
        ("STN_001", "2026-01-01 01:00:00", 180.0, 320.0, 60.0, 2.5),
        # Back to normal baseline
        ("STN_001", "2026-01-01 02:00:00", 25.0, 45.0, 12.0, 0.6),
        # Sensor malfunction (isolated massive 14x surge in PM2.5 from 25.0 to 350.0 with no corresponding PM10 / NO2 elevation)
        ("STN_001", "2026-01-01 03:00:00", 350.0, 45.0, 10.0, 0.5),
    ]
    df = spark_session.createDataFrame(data, ["station_id", "timestamp", "pm2_5", "pm10", "no2", "co"])
    df = df.withColumn("timestamp", F.to_timestamp("timestamp"))

    result_df = AnomalyDetector.detect_and_classify_anomalies(df, pm25_spike_threshold=150.0, jump_ratio_threshold=3.0)
    rows = result_df.orderBy("timestamp").collect()

    assert rows[0]["anomaly_classification"] == "NORMAL"
    assert rows[1]["anomaly_classification"] == "ACUTE_POLLUTION_EVENT"
    assert rows[2]["anomaly_classification"] == "NORMAL"
    assert rows[3]["anomaly_classification"] == "SENSOR_MALFUNCTION_SUSPECT"

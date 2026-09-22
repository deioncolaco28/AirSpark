"""
Integration tests for the full batch data processing pipeline.
"""
import tempfile
from pathlib import Path
from pyspark.sql import SparkSession
from app.quality.quality_pipeline import DataQualityPipeline
from app.processing.joins import DataIntegrator
from app.processing.transformations import FeatureTransformer
from app.aqi.calculator import AQICalculator
from app.analytics.temporal import TemporalAnalytics
from app.analytics.spatial import SpatialAnalytics
from app.processing.partitioning import ParquetStorageManager


def test_end_to_end_batch_pipeline_flow(spark_session: SparkSession):
    # 1. Setup mock raw data
    aq_data = [
        ("STN_001", "2026-01-01 00:00:00", 25.0, 50.0, 15.0, 5.0, 0.5, 20.0),
        ("STN_001", "2026-01-01 01:00:00", 35.0, 70.0, 20.0, 8.0, 0.8, 25.0),
        ("STN_002", "2026-01-01 00:00:00", 120.0, 200.0, 45.0, 15.0, 1.8, 40.0),
    ]
    weather_data = [
        ("STN_001", "2026-01-01 00:00:00", 22.0, 60.0, 3.5, 180.0, 1012.0, 0.0),
        ("STN_001", "2026-01-01 01:00:00", 21.5, 62.0, 4.0, 185.0, 1011.5, 0.0),
        ("STN_002", "2026-01-01 00:00:00", 18.0, 75.0, 1.5, 90.0, 1015.0, 0.0),
    ]
    station_data = [
        ("STN_001", "Sensor Alpha", "Delhi", "Delhi", 28.61, 77.20, 215.0, "Traffic"),
        ("STN_002", "Sensor Beta", "Mumbai", "Maharashtra", 19.07, 72.87, 10.0, "Coastal"),
    ]

    aq_df = spark_session.createDataFrame(aq_data, ["station_id", "timestamp", "pm2_5", "pm10", "no2", "so2", "co", "o3"])
    weather_df = spark_session.createDataFrame(weather_data, ["station_id", "timestamp", "temperature", "humidity", "wind_speed", "wind_direction", "pressure", "rainfall"])
    station_df = spark_session.createDataFrame(station_data, ["station_id", "station_name", "city", "state", "latitude", "longitude", "elevation", "station_type"])

    # 2. Data Quality
    dq = DataQualityPipeline()
    c_aq, _ = dq.process_air_quality(aq_df)
    c_w, _ = dq.process_weather(weather_df)
    c_st, _ = dq.process_stations(station_df)

    # 3. Join & Feature Engineering
    integrated_df, stats = DataIntegrator.integrate_sources(c_aq, c_w, c_st)
    transformed_df = FeatureTransformer.add_temporal_features(integrated_df)

    # 4. AQI Calculation
    calc = AQICalculator()
    aqi_df = calc.calculate_aqi_df(transformed_df)

    # 5. Analytics
    daily_df = TemporalAnalytics.compute_daily_analytics(aqi_df)
    station_summary_df = SpatialAnalytics.compute_station_summary(aqi_df)

    assert aqi_df.count() == 3
    assert daily_df.count() == 1
    assert station_summary_df.count() == 2

    # 6. Test Parquet export into temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "test_output.parquet"
        ParquetStorageManager.write_parquet(aqi_df, str(out_path))
        
        # Verify written Parquet can be read back
        read_df = ParquetStorageManager.read_parquet(spark_session, out_path)
        assert read_df.count() == 3

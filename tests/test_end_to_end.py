"""
Full End-to-End System Tests for AirSpark.
Validates synthetic generation, batch processing, bounded streaming, and ML prediction.
"""
import tempfile
from pathlib import Path
from pyspark.sql import SparkSession

from app.performance.dataset_generator import SyntheticDataGenerator
from app.ingestion.batch_ingestion import BatchIngestionEngine
from app.quality.quality_pipeline import DataQualityPipeline
from app.processing.joins import DataIntegrator
from app.aqi.calculator import AQICalculator
from app.streaming.pipeline import StreamingPipeline
from app.ml.prediction import AQIPredictionModel


def test_complete_end_to_end_system(spark_session: SparkSession):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Generate Small Synthetic Dataset
        gen = SyntheticDataGenerator(seed=123)
        aq_df, w_df, st_df = gen.generate_datasets(num_stations=3, days=3, frequency_minutes=60, inject_quality_issues=True)
        paths = gen.save_datasets(aq_df, w_df, st_df, output_dir=str(tmp_path / "raw"))

        # 2. Ingest
        engine = BatchIngestionEngine(spark_session)
        sources = engine.ingest_all(
            aq_path=paths["air_quality"],
            weather_path=paths["weather"],
            stations_path=paths["stations"]
        )
        assert sources["air_quality"].count() > 0

        # 3. Data Quality & Integration
        dq = DataQualityPipeline()
        c_aq, aq_rep = dq.process_air_quality(sources["air_quality"])
        c_w, _ = dq.process_weather(sources["weather"])
        c_st, _ = dq.process_stations(sources["stations"])

        assert aq_rep["quality_score"] > 50.0

        integrated_df, _ = DataIntegrator.integrate_sources(c_aq, c_w, c_st)
        assert integrated_df.count() > 0

        # 4. AQI Calculation
        calc = AQICalculator()
        aqi_df = calc.calculate_aqi_df(integrated_df)
        assert "aqi" in aqi_df.columns
        assert "aqi_category" in aqi_df.columns

        # 5. ML Prediction Module Verification
        ml_model = AQIPredictionModel()
        ml_metrics, predictions = ml_model.train_and_evaluate(aqi_df)
        assert "rmse" in ml_metrics["linear_regression"]
        assert predictions.count() > 0


def test_bounded_structured_streaming(spark_session: SparkSession):
    pipeline = StreamingPipeline(spark_session)
    result_df = pipeline.run_bounded_stream(
        duration_seconds=5,
        rows_per_second=10,
        num_stations=3,
        query_name="e2e_stream_test"
    )
    assert "aqi" in result_df.columns
    assert "station_id" in result_df.columns
    assert result_df.count() >= 5

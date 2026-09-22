"""
Unit tests for schema validation and integrity checks.
"""
from pyspark.sql import SparkSession
from app.quality.validation import SchemaValidator
from app.processing.schemas import (
    REQUIRED_AIR_QUALITY_COLS,
    REQUIRED_STATION_COLS,
    AIR_QUALITY_SCHEMA,
)


def test_schema_validator_valid_dataset(spark_session: SparkSession):
    data = [("STN_001", "2026-01-01 00:00:00", 25.0, 50.0, 15.0, 5.0, 0.5, 20.0)]
    df = spark_session.createDataFrame(data, AIR_QUALITY_SCHEMA)

    is_valid, missing = SchemaValidator.validate_columns(df, REQUIRED_AIR_QUALITY_COLS, "Air Quality")
    assert is_valid is True
    assert len(missing) == 0


def test_schema_validator_missing_columns(spark_session: SparkSession):
    data = [(25.0, 50.0)]
    df = spark_session.createDataFrame(data, ["pm2_5", "pm10"])

    is_valid, missing = SchemaValidator.validate_columns(df, REQUIRED_AIR_QUALITY_COLS, "Air Quality")
    assert is_valid is False
    assert "station_id" in missing
    assert "timestamp" in missing


def test_schema_validator_empty_dataframe(spark_session: SparkSession):
    df_empty = spark_session.createDataFrame([], AIR_QUALITY_SCHEMA)
    assert SchemaValidator.check_empty(df_empty) is True

    df_non_empty = spark_session.createDataFrame([("STN_001", "2026-01-01 00:00:00", 25.0, 50.0, 15.0, 5.0, 0.5, 20.0)], AIR_QUALITY_SCHEMA)
    assert SchemaValidator.check_empty(df_non_empty) is False

"""
Unit tests for ParquetStorageManager.
Validates Windows-safe partitioned overwrites, append mode, and dataset integrity.
"""
import tempfile
from pathlib import Path
from pyspark.sql import SparkSession
from app.processing.partitioning import ParquetStorageManager


def test_parquet_overwrite_prevents_duplicate_accumulation(spark_session: SparkSession):
    with tempfile.TemporaryDirectory() as tmp_dir:
        target_dir = Path(tmp_dir) / "partitioned_aqi.parquet"
        
        # Initial dataset: 4 records across 2 cities
        data1 = [
            ("STN_001", "Delhi", "2026-01-01", 50.0),
            ("STN_001", "Delhi", "2026-01-02", 55.0),
            ("STN_002", "Mumbai", "2026-01-01", 30.0),
            ("STN_002", "Mumbai", "2026-01-02", 35.0),
        ]
        df1 = spark_session.createDataFrame(data1, ["station_id", "city", "date", "aqi"])
        
        # Write 1 (overwrite)
        ParquetStorageManager.write_parquet(df1, str(target_dir), partition_by=["city"], mode="overwrite")
        read1 = ParquetStorageManager.read_parquet(spark_session, str(target_dir))
        assert read1.count() == 4
        
        # Write 2 with same 4 records (overwrite again)
        ParquetStorageManager.write_parquet(df1, str(target_dir), partition_by=["city"], mode="overwrite")
        read2 = ParquetStorageManager.read_parquet(spark_session, str(target_dir))
        # CRITICAL ASSERTION: Record count must remain exactly 4, NOT accumulate to 8!
        assert read2.count() == 4
        
        # Write 3 with different 2 records (overwrite)
        data3 = [
            ("STN_003", "Kolkata", "2026-01-01", 80.0),
            ("STN_003", "Kolkata", "2026-01-02", 85.0),
        ]
        df3 = spark_session.createDataFrame(data3, ["station_id", "city", "date", "aqi"])
        ParquetStorageManager.write_parquet(df3, str(target_dir), partition_by=["city"], mode="overwrite")
        read3 = ParquetStorageManager.read_parquet(spark_session, str(target_dir))
        assert read3.count() == 2


def test_parquet_append_mode(spark_session: SparkSession):
    with tempfile.TemporaryDirectory() as tmp_dir:
        target_dir = Path(tmp_dir) / "appended_aqi.parquet"
        
        data1 = [("STN_001", "Delhi", 50.0)]
        df1 = spark_session.createDataFrame(data1, ["station_id", "city", "aqi"])
        ParquetStorageManager.write_parquet(df1, str(target_dir), mode="overwrite")
        
        data2 = [("STN_002", "Mumbai", 30.0)]
        df2 = spark_session.createDataFrame(data2, ["station_id", "city", "aqi"])
        ParquetStorageManager.write_parquet(df2, str(target_dir), mode="append")
        
        read_df = ParquetStorageManager.read_parquet(spark_session, str(target_dir))
        assert read_df.count() == 2

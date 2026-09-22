"""
Source adapters for importing real or synthetic datasets across various file formats.
"""
from pathlib import Path
from typing import Optional, Union
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

from app.utils.logging import get_logger
from app.utils.paths import resolve_path

logger = get_logger("AirSpark.SourceAdapters")


class FileSourceAdapter:
    """Reads data from CSV, Parquet, or JSON with optional explicit schemas."""

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def read_csv(
        self,
        file_path: Union[str, Path],
        schema: Optional[StructType] = None,
        header: bool = True,
        infer_schema: bool = False,
    ) -> DataFrame:
        """Read CSV into Spark DataFrame with schema enforcement."""
        p = resolve_path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found at: {p}")

        reader = self.spark.read.option("header", str(header).lower())
        if schema:
            reader = reader.schema(schema)
        elif infer_schema:
            reader = reader.option("inferSchema", "true")

        target_path_str = str(p).replace("\\", "/")
        logger.info(f"Ingesting CSV file from: {target_path_str}")
        return reader.csv(target_path_str)

    def read_parquet(self, file_path: Union[str, Path]) -> DataFrame:
        """Read Parquet directory/file into Spark DataFrame."""
        p = resolve_path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Parquet path not found at: {p}")
        target_path_str = str(p).replace("\\", "/")
        logger.info(f"Ingesting Parquet from: {target_path_str}")
        return self.spark.read.parquet(target_path_str)

"""
Batch ingestion engine for AirSpark.
Loads Air Quality, Weather, and Station Metadata into Spark DataFrames.
"""
from typing import Dict, Optional, Union
from pathlib import Path
from pyspark.sql import DataFrame, SparkSession

from app.config.settings import get_settings
from app.ingestion.source_adapters import FileSourceAdapter
from app.processing.schemas import (
    AIR_QUALITY_SCHEMA,
    WEATHER_SCHEMA,
    STATION_METADATA_SCHEMA,
)
from app.utils.logging import get_logger

logger = get_logger("AirSpark.BatchIngestion")


class BatchIngestionEngine:
    """Orchestrates ingestion of heterogeneous data sources."""

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.adapter = FileSourceAdapter(spark)
        self.settings = get_settings()

    def ingest_all(
        self,
        aq_path: Optional[Union[str, Path]] = None,
        weather_path: Optional[Union[str, Path]] = None,
        stations_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, DataFrame]:
        """
        Ingest all required data sources.
        """
        aq_p = aq_path or self.settings.paths.get("raw_air_quality")
        weather_p = weather_path or self.settings.paths.get("raw_weather")
        stations_p = stations_path or self.settings.paths.get("raw_stations")

        logger.info("Starting batch data ingestion for all sources")

        aq_df = self.adapter.read_csv(aq_p, schema=AIR_QUALITY_SCHEMA)
        weather_df = self.adapter.read_csv(weather_p, schema=WEATHER_SCHEMA)
        stations_df = self.adapter.read_csv(stations_p, schema=STATION_METADATA_SCHEMA)

        logger.info("Batch ingestion completed successfully")
        return {
            "air_quality": aq_df,
            "weather": weather_df,
            "stations": stations_df,
        }

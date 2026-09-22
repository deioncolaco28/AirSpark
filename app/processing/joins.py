"""
Data Integration and Join Engine for AirSpark.
Safely integrates Air Quality, Weather, and Station Metadata into a unified Big Data model.
"""
from typing import Dict, Any, Tuple
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from app.utils.logging import get_logger

logger = get_logger("AirSpark.Processing.Joins")


class DataIntegrator:
    """Integrates heterogeneous sources with schema alignment and join diagnostics."""

    @staticmethod
    def integrate_sources(
        aq_df: DataFrame,
        weather_df: DataFrame,
        stations_df: DataFrame,
        temporal_alignment: str = "exact",  # "exact" or "hourly"
    ) -> Tuple[DataFrame, Dict[str, Any]]:
        """
        Join Air Quality, Weather, and Station Metadata into an integrated DataFrame.
        """
        logger.info(f"Integrating multi-source datasets with temporal alignment mode '{temporal_alignment}'")
        aq_count = aq_df.count()
        weather_count = weather_df.count()
        stations_count = stations_df.count()

        # Check for empty datasets
        if aq_count == 0 or stations_count == 0:
            logger.warning("One or more key input datasets are empty during integration")

        # Prepare temporal join keys
        # Clean weather columns to prevent duplicate column collisions with aq_df
        aq_cols_set = set(aq_df.columns)
        weather_cols = []
        for c in weather_df.columns:
            if c in ["station_id", "timestamp"]:
                continue
            if c in aq_cols_set:
                # Rename colliding column from weather (e.g., is_any_domain_outlier)
                weather_df = weather_df.withColumnRenamed(c, f"weather_{c}")
                weather_cols.append(f"weather_{c}")
            else:
                weather_cols.append(c)

        if temporal_alignment == "hourly":
            aq_aligned = aq_df.withColumn("join_time", F.date_trunc("hour", F.col("timestamp")))
            weather_aligned = weather_df.withColumn("join_time", F.date_trunc("hour", F.col("timestamp")))
            
            weather_select = weather_aligned.select("station_id", "join_time", *weather_cols).dropDuplicates(["station_id", "join_time"])

            aq_weather = aq_aligned.join(
                weather_select,
                on=["station_id", "join_time"],
                how="left"
            ).drop("join_time")
        else:
            # Exact timestamp match
            weather_select = weather_df.select("station_id", "timestamp", *weather_cols)

            aq_weather = aq_df.join(
                weather_select,
                on=["station_id", "timestamp"],
                how="left"
            )

        # Join Station Metadata on station_id
        aq_weather_cols_set = set(aq_weather.columns)
        station_cols = []
        for c in stations_df.columns:
            if c == "station_id":
                continue
            if c in aq_weather_cols_set:
                stations_df = stations_df.withColumnRenamed(c, f"station_meta_{c}")
                station_cols.append(f"station_meta_{c}")
            else:
                station_cols.append(c)

        integrated_df = aq_weather.join(
            stations_df.select("station_id", *station_cols),
            on="station_id",
            how="left"
        )


        integrated_count = integrated_df.count()
        matched_stations_count = integrated_df.select("station_id").distinct().count()

        join_stats = {
            "air_quality_records": aq_count,
            "weather_records": weather_count,
            "station_count": stations_count,
            "integrated_records": integrated_count,
            "matched_stations": matched_stations_count,
            "temporal_alignment": temporal_alignment,
        }

        logger.info(f"Data integration completed: {integrated_count} integrated records from {matched_stations_count} stations.")
        return integrated_df, join_stats

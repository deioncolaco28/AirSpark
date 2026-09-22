"""
Pollutant contribution and breakdown analytics for AirSpark.
"""
from typing import Dict, Any
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from app.processing.schemas import POLLUTANT_COLS
from app.utils.logging import get_logger

logger = get_logger("AirSpark.AQI.PollutantAnalysis")


class PollutantAnalyzer:
    """Analyzes pollutant dominance and distribution."""

    @staticmethod
    def compute_dominance_distribution(df: DataFrame) -> DataFrame:
        """
        Compute percentage breakdown of dominant pollutants across all observations.
        """
        if "dominant_pollutant" not in df.columns:
            return df

        total_count = df.count()
        if total_count == 0:
            return df.groupBy("dominant_pollutant").count()

        dominance_df = (
            df.groupBy("dominant_pollutant")
            .agg(
                F.count("*").alias("count"),
                F.round(F.count("*") / total_count * 100.0, 2).alias("percentage")
            )
            .orderBy(F.col("count").desc())
        )
        return dominance_df

    @staticmethod
    def compute_pollutant_summary(df: DataFrame) -> DataFrame:
        """
        Compute mean, min, max, stddev for all pollutant columns and overall AQI.
        """
        agg_cols = []
        for pol in POLLUTANT_COLS:
            if pol in df.columns:
                agg_cols.extend([
                    F.round(F.avg(pol), 2).alias(f"{pol}_avg"),
                    F.round(F.min(pol), 2).alias(f"{pol}_min"),
                    F.round(F.max(pol), 2).alias(f"{pol}_max"),
                ])

        if "aqi" in df.columns:
            agg_cols.extend([
                F.round(F.avg("aqi"), 2).alias("aqi_avg"),
                F.round(F.min("aqi"), 2).alias("aqi_min"),
                F.round(F.max("aqi"), 2).alias("aqi_max"),
            ])

        return df.select(agg_cols)

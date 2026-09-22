"""
PySpark Feature Engineering and Transformations for AirSpark.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from app.utils.logging import get_logger

logger = get_logger("AirSpark.Processing.Transformations")


class FeatureTransformer:
    """Generates temporal, meteorological, and interaction features."""

    @staticmethod
    def add_temporal_features(df: DataFrame) -> DataFrame:
        """Add calendar features (year, month, day, day_of_week, hour, is_weekend)."""
        return (
            df.withColumn("year", F.year("timestamp"))
            .withColumn("month", F.month("timestamp"))
            .withColumn("day", F.dayofmonth("timestamp"))
            .withColumn("hour", F.hour("timestamp"))
            .withColumn("day_of_week", F.dayofweek("timestamp"))
            .withColumn("is_weekend", F.when(F.dayofweek("timestamp").isin([1, 7]), True).otherwise(False))
        )

    @staticmethod
    def add_ratio_features(df: DataFrame) -> DataFrame:
        """Add domain interaction ratios (e.g. PM2.5/PM10 ratio)."""
        out_df = df
        if "pm2_5" in df.columns and "pm10" in df.columns:
            out_df = out_df.withColumn(
                "pm_ratio",
                F.when(
                    (F.col("pm10").isNotNull()) & (F.col("pm10") > 0),
                    F.round(F.col("pm2_5") / F.col("pm10"), 3)
                ).otherwise(F.lit(None))
            )
        return out_df

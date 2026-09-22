"""
Timestamp and feature normalization for AirSpark.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from app.utils.logging import get_logger

logger = get_logger("AirSpark.Quality.Normalization")


class Normalizer:
    """Standardizes timestamps, strings, and measurement units."""

    @staticmethod
    def normalize_timestamps(df: DataFrame, timestamp_col: str = "timestamp") -> DataFrame:
        """
        Safely parse diverse string timestamp formats into standard Spark TimestampType using try_to_timestamp.
        Tolerates invalid timestamp strings by returning NULL.
        """
        logger.info(f"Normalizing timestamp column '{timestamp_col}'")
        col = F.col(timestamp_col)
        
        # Use try_to_timestamp with fallback formats
        parsed_ts = F.coalesce(
            F.try_to_timestamp(col, F.lit("yyyy-MM-dd HH:mm:ss")),
            F.try_to_timestamp(col, F.lit("yyyy-MM-dd'T'HH:mm:ss")),
            F.try_to_timestamp(col, F.lit("yyyy-MM-dd'T'HH:mm:ss.SSS")),
            F.try_to_timestamp(col, F.lit("yyyy-MM-dd HH:mm:ss.SSSSSS")),
            F.try_to_timestamp(col, F.lit("yyyy-MM-dd")),
            F.try_to_timestamp(col)
        )

        return df.withColumn(timestamp_col, parsed_ts)

    @staticmethod
    def standardize_strings(df: DataFrame, string_cols: list) -> DataFrame:
        """Trim whitespace and standardize casing for string columns."""
        out_df = df
        for col in string_cols:
            if col in df.columns:
                out_df = out_df.withColumn(col, F.trim(F.col(col)))
        return out_df

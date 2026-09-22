"""
Windowed Streaming Aggregations for AirSpark.
Implements event-time watermarking and tumbling / sliding window aggregations.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from app.config.settings import get_settings
from app.utils.logging import get_logger

logger = get_logger("AirSpark.Streaming.Aggregations")


class StreamingAggregator:
    """Configures watermarking and sliding window aggregations on streaming DataFrames."""

    @staticmethod
    def apply_windowed_aggregation(
        streaming_df: DataFrame,
        watermark_delay: str = "10 minutes",
        window_duration: str = "10 minutes",
        slide_duration: str = "5 minutes",
    ) -> DataFrame:
        """
        Apply watermark and windowed aggregations over streaming sensor events.
        """
        logger.info(f"Applying streaming window aggregation [Watermark: {watermark_delay}, Window: {window_duration}, Slide: {slide_duration}]")

        watermarked_df = streaming_df.withWatermark("timestamp", watermark_delay)

        windowed_df = (
            watermarked_df.groupBy(
                F.window("timestamp", window_duration, slide_duration),
                "station_id"
            )
            .agg(
                F.round(F.avg("pm2_5"), 2).alias("avg_pm2_5"),
                F.round(F.max("pm2_5"), 2).alias("max_pm2_5"),
                F.round(F.avg("pm10"), 2).alias("avg_pm10"),
                F.round(F.avg("no2"), 2).alias("avg_no2"),
                F.round(F.avg("so2"), 2).alias("avg_so2"),
                F.round(F.avg("co"), 2).alias("avg_co"),
                F.round(F.avg("o3"), 2).alias("avg_o3"),
                F.count("*").alias("window_readings_count")
            )
        )
        return windowed_df

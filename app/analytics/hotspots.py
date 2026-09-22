"""
Hotspot Identification and Spatial Pollution Clustering for AirSpark.
Detects persistent high-pollution zones and computes hotspot severity scores.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from app.utils.logging import get_logger

logger = get_logger("AirSpark.Analytics.Hotspots")


class HotspotAnalyzer:
    """Identifies and ranks geographic pollution hotspots."""

    @staticmethod
    def identify_hotspots(df: DataFrame, unhealthy_threshold: float = 150.0) -> DataFrame:
        """
        Identify stations with persistent high AQI levels.
        Computes the percentage of hours exceeding the threshold and a Hotspot Severity Index.
        """
        logger.info(f"Computing pollution hotspots with threshold AQI >= {unhealthy_threshold}")

        hotspot_df = (
            df.groupBy("station_id")
            .agg(
                F.first("station_name").alias("station_name") if "station_name" in df.columns else F.first("station_id").alias("station_name"),
                F.first("city").alias("city") if "city" in df.columns else F.lit("Unknown").alias("city"),
                F.first("latitude").alias("latitude") if "latitude" in df.columns else F.lit(0.0).alias("latitude"),
                F.first("longitude").alias("longitude") if "longitude" in df.columns else F.lit(0.0).alias("longitude"),
                F.count("*").alias("total_readings"),
                F.count(F.when(F.col("aqi") >= unhealthy_threshold, True)).alias("unhealthy_count"),
                F.round(F.avg("aqi"), 2).alias("mean_aqi"),
                F.round(F.max("aqi"), 2).alias("max_aqi"),
            )
            .withColumn(
                "unhealthy_pct",
                F.round(F.col("unhealthy_count") / F.col("total_readings") * 100.0, 2)
            )
            .withColumn(
                "hotspot_score",
                F.round((F.col("unhealthy_pct") * 0.6) + ((F.col("mean_aqi") / 500.0) * 100.0 * 0.4), 2)
            )
            .withColumn(
                "hotspot_level",
                F.when(F.col("hotspot_score") >= 60.0, F.lit("Critical Hotspot"))
                .when(F.col("hotspot_score") >= 35.0, F.lit("Moderate Hotspot"))
                .when(F.col("hotspot_score") >= 15.0, F.lit("Low Hotspot"))
                .otherwise(F.lit("Clean / Compliant"))
            )
            .orderBy(F.col("hotspot_score").desc())
        )
        return hotspot_df

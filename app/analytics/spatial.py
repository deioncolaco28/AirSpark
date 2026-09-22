"""
Spatial Analytics Engine for AirSpark.
Computes station-level performance, city/regional summaries, and spatial rankings using PySpark.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from app.processing.schemas import POLLUTANT_COLS
from app.utils.logging import get_logger

logger = get_logger("AirSpark.Analytics.Spatial")


class SpatialAnalytics:
    """Computes distributed spatial aggregations and rankings."""

    @staticmethod
    def compute_station_summary(df: DataFrame) -> DataFrame:
        """
        Aggregate air quality metrics per monitoring station.
        """
        logger.info("Computing station-level spatial summary")

        agg_exprs = [
            F.first("station_name").alias("station_name") if "station_name" in df.columns else F.first("station_id").alias("station_name"),
            F.first("city").alias("city") if "city" in df.columns else F.lit("Unknown").alias("city"),
            F.first("latitude").alias("latitude") if "latitude" in df.columns else F.lit(0.0).alias("latitude"),
            F.first("longitude").alias("longitude") if "longitude" in df.columns else F.lit(0.0).alias("longitude"),
            F.first("station_type").alias("station_type") if "station_type" in df.columns else F.lit("General").alias("station_type"),
            F.round(F.avg("aqi"), 2).alias("mean_aqi"),
            F.round(F.min("aqi"), 2).alias("min_aqi"),
            F.round(F.max("aqi"), 2).alias("max_aqi"),
            F.round(F.stddev("aqi"), 2).alias("std_aqi"),
            F.count("*").alias("total_readings"),
        ]

        for pol in POLLUTANT_COLS:
            if pol in df.columns:
                agg_exprs.append(F.round(F.avg(pol), 2).alias(f"mean_{pol}"))

        station_df = df.groupBy("station_id").agg(*agg_exprs).orderBy(F.col("mean_aqi").desc())
        return station_df

    @staticmethod
    def compute_city_summary(df: DataFrame) -> DataFrame:
        """
        Aggregate air quality metrics per city.
        """
        logger.info("Computing city-level spatial summary")
        if "city" not in df.columns:
            return df

        city_df = (
            df.groupBy("city")
            .agg(
                F.countDistinct("station_id").alias("station_count"),
                F.round(F.avg("aqi"), 2).alias("mean_aqi"),
                F.round(F.min("aqi"), 2).alias("min_aqi"),
                F.round(F.max("aqi"), 2).alias("max_aqi"),
                F.round(F.avg("pm2_5"), 2).alias("mean_pm2_5") if "pm2_5" in df.columns else F.lit(None),
                F.round(F.avg("pm10"), 2).alias("mean_pm10") if "pm10" in df.columns else F.lit(None),
                F.count("*").alias("total_readings"),
            )
            .orderBy(F.col("mean_aqi").desc())
        )
        return city_df

    @staticmethod
    def rank_stations_by_aqi(df: DataFrame, ascending: bool = False) -> DataFrame:
        """
        Rank stations by average AQI.
        """
        station_summary = SpatialAnalytics.compute_station_summary(df)
        order_col = F.col("mean_aqi").asc() if ascending else F.col("mean_aqi").desc()
        return station_summary.orderBy(order_col)

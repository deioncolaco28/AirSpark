"""
Temporal Analytics Engine for AirSpark.
Provides hourly, daily, monthly, seasonal, and rolling-window aggregations using PySpark.
"""
from typing import Dict, Any
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from app.processing.schemas import POLLUTANT_COLS
from app.utils.logging import get_logger

logger = get_logger("AirSpark.Analytics.Temporal")


class TemporalAnalytics:
    """Computes distributed time-series aggregations."""

    @staticmethod
    def compute_hourly_profile(df: DataFrame) -> DataFrame:
        """
        Compute diurnal (hour of day 0-23) mean AQI and pollutant averages across all stations.
        """
        logger.info("Computing diurnal hourly profile")
        df_with_hour = df.withColumn("hour", F.hour("timestamp"))

        agg_exprs = [
            F.round(F.avg("aqi"), 2).alias("avg_aqi"),
            F.round(F.min("aqi"), 2).alias("min_aqi"),
            F.round(F.max("aqi"), 2).alias("max_aqi"),
            F.count("*").alias("sample_count"),
        ]

        for pol in POLLUTANT_COLS:
            if pol in df.columns:
                agg_exprs.append(F.round(F.avg(pol), 2).alias(f"avg_{pol}"))

        hourly_df = df_with_hour.groupBy("hour").agg(*agg_exprs).orderBy("hour")
        return hourly_df

    @staticmethod
    def compute_daily_analytics(df: DataFrame) -> DataFrame:
        """
        Compute daily aggregations (mean AQI, min AQI, max AQI, dominant pollutant, pollutant means).
        """
        logger.info("Computing daily temporal analytics")
        df_with_date = df.withColumn("date", F.to_date("timestamp"))

        agg_exprs = [
            F.round(F.avg("aqi"), 2).alias("mean_aqi"),
            F.round(F.min("aqi"), 2).alias("min_aqi"),
            F.round(F.max("aqi"), 2).alias("max_aqi"),
            F.count("*").alias("total_readings"),
        ]

        for pol in POLLUTANT_COLS:
            if pol in df.columns:
                agg_exprs.append(F.round(F.avg(pol), 2).alias(f"mean_{pol}"))

        daily_df = df_with_date.groupBy("date").agg(*agg_exprs).orderBy("date")
        return daily_df

    @staticmethod
    def compute_monthly_analytics(df: DataFrame) -> DataFrame:
        """
        Compute monthly trends and seasonal patterns.
        """
        logger.info("Computing monthly temporal analytics")
        df_with_month = (
            df.withColumn("year", F.year("timestamp"))
            .withColumn("month", F.month("timestamp"))
            .withColumn("year_month", F.date_format("timestamp", "yyyy-MM"))
        )

        monthly_df = (
            df_with_month.groupBy("year_month", "year", "month")
            .agg(
                F.round(F.avg("aqi"), 2).alias("mean_aqi"),
                F.round(F.max("aqi"), 2).alias("peak_aqi"),
                F.round(F.avg("pm2_5"), 2).alias("mean_pm2_5") if "pm2_5" in df.columns else F.lit(None),
                F.count("*").alias("record_count"),
            )
            .orderBy("year_month")
        )
        return monthly_df

    @staticmethod
    def compute_seasonal_analytics(df: DataFrame) -> DataFrame:
        """
        Compute seasonal statistics (Winter, Summer, Monsoon, Post-Monsoon).
        """
        logger.info("Computing seasonal analytics")
        # Define season based on month
        season_expr = (
            F.when(F.month("timestamp").isin([12, 1, 2]), F.lit("Winter"))
            .when(F.month("timestamp").isin([3, 4, 5]), F.lit("Summer / Pre-Monsoon"))
            .when(F.month("timestamp").isin([6, 7, 8, 9]), F.lit("Monsoon"))
            .otherwise(F.lit("Post-Monsoon / Autumn"))
        )

        df_seasonal = df.withColumn("season", season_expr)
        seasonal_summary = (
            df_seasonal.groupBy("season")
            .agg(
                F.round(F.avg("aqi"), 2).alias("mean_aqi"),
                F.round(F.min("aqi"), 2).alias("min_aqi"),
                F.round(F.max("aqi"), 2).alias("max_aqi"),
                F.round(F.avg("pm2_5"), 2).alias("avg_pm2_5") if "pm2_5" in df.columns else F.lit(None),
                F.count("*").alias("total_records"),
            )
            .orderBy(F.col("mean_aqi").desc())
        )
        return seasonal_summary

    @staticmethod
    def compute_rolling_averages(df: DataFrame, window_hours: int = 24) -> DataFrame:
        """
        Compute rolling average AQI per station using Spark Window functions.
        """
        # Convert timestamp to epoch seconds for rangeBetween
        df_epoch = df.withColumn("epoch_sec", F.col("timestamp").cast("long"))
        seconds_window = window_hours * 3600

        window_spec = (
            Window.partitionBy("station_id")
            .orderBy("epoch_sec")
            .rangeBetween(-seconds_window, 0)
        )

        return df_epoch.withColumn(
            f"rolling_{window_hours}h_avg_aqi",
            F.round(F.avg("aqi").over(window_spec), 2)
        ).drop("epoch_sec")

"""
Explainable Anomaly Detection Engine for AirSpark.
Distinguishes between sensor data-quality glitches and genuine acute pollution events.
"""
from typing import Dict, Any
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from app.utils.logging import get_logger

logger = get_logger("AirSpark.Analytics.Anomalies")


class AnomalyDetector:
    """Classifies anomalous readings with explainable diagnostic reasons."""

    @staticmethod
    def detect_and_classify_anomalies(
        df: DataFrame,
        pm25_spike_threshold: float = 150.0,
        jump_ratio_threshold: float = 3.0,
    ) -> DataFrame:
        """
        Evaluate temporal jump ratios and multi-pollutant consistency.
        Classifies records into:
        - 'NORMAL'
        - 'ACUTE_POLLUTION_EVENT' (Multi-pollutant correlated elevation)
        - 'SENSOR_MALFUNCTION_SUSPECT' (Single isolated extreme surge without corresponding pollutants)
        """
        logger.info("Running explainable anomaly detection and classification")

        # Window partitioned by station and ordered by timestamp
        window_spec = (
            Window.partitionBy("station_id")
            .orderBy("timestamp")
            .rowsBetween(-1, -1)
        )

        # Lag previous PM2.5 value
        df_lag = df.withColumn("prev_pm2_5", F.lag("pm2_5", 1).over(Window.partitionBy("station_id").orderBy("timestamp")))
        
        # Relative jump ratio
        df_jump = df_lag.withColumn(
            "pm25_jump_ratio",
            F.when(
                (F.col("prev_pm2_5").isNotNull()) & (F.col("prev_pm2_5") > 5.0),
                F.round(F.col("pm2_5") / F.col("prev_pm2_5"), 2)
            ).otherwise(F.lit(1.0))
        )

        # Multi-pollutant co-elevation condition (PM2.5 and PM10 or NO2 are elevated together)
        co_elevation_cond = (
            (F.col("pm2_5") > pm25_spike_threshold) &
            (
                (F.col("pm10") > F.col("pm2_5") * 1.1) | 
                (F.col("no2") > 50.0) |
                (F.col("co") > 2.0)
            )
        )

        isolated_spike_cond = (
            (F.col("pm2_5") > pm25_spike_threshold) &
            (F.col("pm25_jump_ratio") > jump_ratio_threshold) &
            (~co_elevation_cond)
        )

        classified_df = (
            df_jump.withColumn(
                "anomaly_classification",
                F.when(isolated_spike_cond, F.lit("SENSOR_MALFUNCTION_SUSPECT"))
                .when(co_elevation_cond, F.lit("ACUTE_POLLUTION_EVENT"))
                .otherwise(F.lit("NORMAL"))
            )
            .withColumn(
                "is_anomaly",
                F.col("anomaly_classification") != "NORMAL"
            )
        )

        return classified_df

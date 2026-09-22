

"""
Missing value detection and imputation strategies for AirSpark.
"""
from typing import Dict, List, Any
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from app.utils.logging import get_logger

logger = get_logger("AirSpark.Quality.MissingValues")


class MissingValueHandler:
    """Detects and imputes missing numeric and categorical values using PySpark."""

    @staticmethod
    def audit_missing_values(df: DataFrame, cols: List[str]) -> Dict[str, int]:
        """
        Count null / NaN values for each specified column.
        """
        exprs = [F.count(F.when(F.col(c).isNull() | F.isnan(c) if c in df.columns else F.lit(False), c)).alias(c) for c in cols if c in df.columns]
        if not exprs:
            return {}

        result = df.select(exprs).collect()[0].asDict()
        return result

    @staticmethod
    def impute_missing(
        df: DataFrame,
        numeric_cols: List[str],
        partition_col: str = "station_id",
        order_col: str = "timestamp",
    ) -> DataFrame:
        """
        Impute missing numeric values using a 2-stage strategy:
        1. Temporal forward fill within station partition (Window ordered by timestamp).
        2. Station-level median fallback.
        3. Global column mean fallback for any remaining nulls.
        """
        logger.info(f"Imputing missing values for numeric columns: {numeric_cols}")
        cleaned_df = df

        # Step 1: Forward Fill within station
        window_spec = (
            Window.partitionBy(partition_col)
            .orderBy(order_col)
            .rowsBetween(Window.unboundedPreceding, Window.currentRow)
        )

        for col in numeric_cols:
            if col in cleaned_df.columns:
                # Forward fill last non-null value
                cleaned_df = cleaned_df.withColumn(
                    f"{col}_ffill",
                    F.last(F.col(col), ignorenulls=True).over(window_spec)
                )
                cleaned_df = cleaned_df.withColumn(col, F.coalesce(F.col(col), F.col(f"{col}_ffill"))).drop(f"{col}_ffill")

        # Step 2: Station average fallback for leading nulls
        station_window = Window.partitionBy(partition_col)
        for col in numeric_cols:
            if col in cleaned_df.columns:
                station_avg = F.avg(F.col(col)).over(station_window)
                cleaned_df = cleaned_df.withColumn(col, F.coalesce(F.col(col), station_avg))

        # Step 3: Global average fallback
        # Calculate global averages in one pass
        global_avg_exprs = [F.avg(F.col(c)).alias(f"global_avg_{c}") for c in numeric_cols if c in cleaned_df.columns]
        if global_avg_exprs:
            global_avgs = cleaned_df.select(global_avg_exprs).first()
            if global_avgs:
                for col in numeric_cols:
                    if col in cleaned_df.columns:
                        avg_val = global_avgs[f"global_avg_{col}"]
                        if avg_val is not None:
                            cleaned_df = cleaned_df.withColumn(col, F.coalesce(F.col(col), F.lit(float(avg_val))))

        return cleaned_df

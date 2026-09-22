"""
Statistical outlier detection and handling for AirSpark.
Provides IQR, Z-Score, and physical domain-bound evaluation without blindly deleting pollution spikes.
"""
from typing import Dict, List, Tuple
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from app.utils.logging import get_logger

logger = get_logger("AirSpark.Quality.Outliers")


class OutlierDetector:
    """Detects and flags/caps statistical and domain outliers."""

    @staticmethod
    def audit_and_handle_domain_limits(
        df: DataFrame,
        domain_limits: Dict[str, List[float]],
        action: str = "cap",
    ) -> Tuple[DataFrame, Dict[str, int]]:
        """
        Enforce physical domain limits and attach explicit domain outlier flags.
        Returns (transformed_df, domain_outlier_counts_dict).
        """
        active_limits = {c: lim for c, lim in domain_limits.items() if c in df.columns}
        if not active_limits:
            return df, {}

        # 1. Attach explicit domain outlier flags per pollutant and overall
        processed_df = df
        flag_conditions = []
        for col_name, lim in active_limits.items():
            low_limit, high_limit = float(lim[0]), float(lim[1])
            domain_cond = (F.col(col_name) < low_limit) | (F.col(col_name) > high_limit)
            processed_df = processed_df.withColumn(f"is_{col_name}_domain_outlier", domain_cond)
            flag_conditions.append(domain_cond)

        if flag_conditions:
            combined_domain_cond = flag_conditions[0]
            for cond in flag_conditions[1:]:
                combined_domain_cond = combined_domain_cond | cond
            processed_df = processed_df.withColumn("is_any_domain_outlier", combined_domain_cond)
        else:
            processed_df = processed_df.withColumn("is_any_domain_outlier", F.lit(False))

        # 2. Batch compute domain violation counts in ONE Spark aggregation pass
        exprs = [
            F.sum(F.when(F.col(f"is_{col_name}_domain_outlier"), 1).otherwise(0)).alias(col_name)
            for col_name in active_limits
        ]
        exprs.append(F.sum(F.when(F.col("is_any_domain_outlier"), 1).otherwise(0)).alias("total_domain_outliers"))

        count_row = processed_df.select(exprs).first()
        outlier_counts = count_row.asDict() if count_row else {c: 0 for c in active_limits}
        outlier_counts = {k: (int(v) if v is not None else 0) for k, v in outlier_counts.items()}

        # 3. If action == 'cap', apply physical boundary capping while preserving flags
        if action == "cap":
            for col_name, lim in active_limits.items():
                low_limit, high_limit = float(lim[0]), float(lim[1])
                processed_df = processed_df.withColumn(
                    col_name,
                    F.when(F.col(col_name) < low_limit, F.lit(low_limit))
                    .when(F.col(col_name) > high_limit, F.lit(high_limit))
                    .otherwise(F.col(col_name))
                )

        return processed_df, outlier_counts

    @staticmethod
    def calculate_iqr_bounds(
        df: DataFrame,
        numeric_cols: List[str],
        multiplier: float = 3.0,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate approximate IQR bounds using single-pass Spark approxQuantile across all columns.
        """
        bounds = {}
        active_cols = [c for c in numeric_cols if c in df.columns]
        if not active_cols:
            return bounds

        for col in active_cols:
            try:
                quantiles = df.approxQuantile(col, [0.25, 0.75], 0.05)
                if quantiles and len(quantiles) == 2:
                    q1, q3 = float(quantiles[0]), float(quantiles[1])
                    iqr = q3 - q1
                    lower = q1 - multiplier * iqr
                    upper = q3 + multiplier * iqr
                    bounds[col] = (lower, upper)
            except Exception as e:
                logger.debug(f"approxQuantile on {col} note: {e}")
        
        return bounds

    @staticmethod
    def flag_statistical_outliers(
        df: DataFrame,
        numeric_cols: List[str],
        multiplier: float = 3.0,
    ) -> Tuple[DataFrame, Dict[str, int]]:
        """
        Flag rows exceeding IQR statistical thresholds and compute real outlier row counts.
        """
        bounds = OutlierDetector.calculate_iqr_bounds(df, numeric_cols, multiplier)
        flagged_df = df

        condition_list = []
        for col, (lower, upper) in bounds.items():
            outlier_cond = (F.col(col) < lower) | (F.col(col) > upper)
            flagged_df = flagged_df.withColumn(f"is_{col}_outlier", outlier_cond)
            condition_list.append((col, outlier_cond))

        if condition_list:
            combined_cond = condition_list[0][1]
            for _, cond in condition_list[1:]:
                combined_cond = combined_cond | cond
            flagged_df = flagged_df.withColumn("is_any_outlier", combined_cond)

            # Compute real statistical outlier counts in ONE Spark aggregation pass
            agg_exprs = [
                F.sum(F.when(F.col(f"is_{c}_outlier"), 1).otherwise(0)).alias(c)
                for c, _ in condition_list
            ]
            agg_exprs.append(F.sum(F.when(F.col("is_any_outlier"), 1).otherwise(0)).alias("total_outlier_rows"))

            count_row = flagged_df.select(agg_exprs).first()
            outlier_counts = count_row.asDict() if count_row else {}
            outlier_counts = {k: (int(v) if v is not None else 0) for k, v in outlier_counts.items()}
        else:
            flagged_df = flagged_df.withColumn("is_any_outlier", F.lit(False))
            outlier_counts = {"total_outlier_rows": 0}

        return flagged_df, outlier_counts

    @staticmethod
    def detect_iqr_outliers(
        df: DataFrame,
        numeric_cols: List[str],
        factor: float = 3.0,
    ) -> Tuple[DataFrame, Dict[str, int]]:
        """Alias for flag_statistical_outliers."""
        return OutlierDetector.flag_statistical_outliers(df, numeric_cols, multiplier=factor)


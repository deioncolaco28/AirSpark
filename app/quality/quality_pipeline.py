"""
Comprehensive Data Quality Pipeline for AirSpark.
Executes schema validation, deduplication, timestamp normalization,
domain checking, missing value imputation, and statistical outlier flagging.
"""
from typing import Dict, Any, Tuple
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from app.config.settings import get_settings
from app.quality.validation import SchemaValidator
from app.quality.normalization import Normalizer
from app.quality.missing_values import MissingValueHandler
from app.quality.outliers import OutlierDetector
from app.processing.schemas import (
    REQUIRED_AIR_QUALITY_COLS,
    POLLUTANT_COLS,
    WEATHER_COLS,
    REQUIRED_STATION_COLS,
)
from app.utils.logging import get_logger

logger = get_logger("AirSpark.Quality.Pipeline")


class DataQualityPipeline:
    """End-to-end Data Quality Pipeline."""

    def __init__(self):
        self.settings = get_settings()
        self.dq_cfg = self.settings.data_quality_config

    def process_air_quality(self, raw_df: DataFrame) -> Tuple[DataFrame, Dict[str, Any]]:
        """
        Process Air Quality DataFrame through complete DQ pipeline.
        Returns (cleaned_df, quality_report).
        """
        logger.info("Executing Data Quality Pipeline on Air Quality dataset")
        initial_count = raw_df.count()

        # 1. Schema Validation
        is_valid, missing_cols = SchemaValidator.validate_columns(
            raw_df, REQUIRED_AIR_QUALITY_COLS, dataset_name="Air Quality"
        )
        if not is_valid:
            raise ValueError(f"Air Quality dataset missing columns: {missing_cols}")

        # 2. Timestamp & String Normalization
        df = Normalizer.normalize_timestamps(raw_df, timestamp_col="timestamp")
        df = Normalizer.standardize_strings(df, ["station_id"])
        
        # Filter out records where timestamp could not be parsed or station_id is null
        df = df.filter(F.col("timestamp").isNotNull() & F.col("station_id").isNotNull())
        valid_ts_count = df.count()
        invalid_ts_count = initial_count - valid_ts_count

        # 3. Deduplication on (station_id, timestamp)
        df_dedup = df.dropDuplicates(subset=["station_id", "timestamp"])
        dedup_count = df_dedup.count()
        duplicate_count = valid_ts_count - dedup_count

        # 4. Initial Missing Value Audit
        initial_missing = MissingValueHandler.audit_missing_values(df_dedup, POLLUTANT_COLS)

        # 5. Domain Limits & Negative Value Correction
        domain_limits = self.dq_cfg.get("domain_limits", {})
        df_domain, domain_outliers = OutlierDetector.audit_and_handle_domain_limits(
            df_dedup, domain_limits, action="cap"
        )

        # 6. Missing Value Imputation
        df_imputed = MissingValueHandler.impute_missing(
            df_domain, numeric_cols=POLLUTANT_COLS, partition_col="station_id", order_col="timestamp"
        )

        # 7. Post-imputation Missing Value Audit
        post_missing = MissingValueHandler.audit_missing_values(df_imputed, POLLUTANT_COLS)

        # 8. Outlier Detection (Statistical Flagging)
        multiplier = float(self.dq_cfg.get("iqr_multiplier", 3.0))
        cleaned_df, outlier_report = OutlierDetector.flag_statistical_outliers(
            df_imputed, numeric_cols=POLLUTANT_COLS, multiplier=multiplier
        )

        final_count = cleaned_df.count()
        retention_pct = round((final_count / initial_count * 100.0), 2) if initial_count > 0 else 0.0

        quality_report = {
            "dataset": "Air Quality",
            "initial_records": initial_count,
            "final_records": final_count,
            "invalid_records": invalid_ts_count,
            "duplicate_records": duplicate_count,
            "domain_outliers_corrected": domain_outliers,
            "statistical_outliers_flagged": outlier_report.get("total_outlier_rows", 0),
            "statistical_outliers_breakdown": outlier_report,
            "missing_before_cleaning": initial_missing,
            "missing_after_cleaning": post_missing,
            "retention_percentage": retention_pct,
            "quality_score": round(max(0.0, 100.0 - (duplicate_count + invalid_ts_count + sum(initial_missing.values())) / (initial_count * len(POLLUTANT_COLS) + 1e-5) * 100), 2),
        }

        logger.info(f"Air Quality DQ Completed. Quality Score: {quality_report['quality_score']}%, Retained: {retention_pct}%, Outliers: {quality_report['statistical_outliers_flagged']}")
        return cleaned_df, quality_report

    def process_weather(self, raw_df: DataFrame) -> Tuple[DataFrame, Dict[str, Any]]:
        """
        Process Weather DataFrame through DQ pipeline.
        """
        logger.info("Executing Data Quality Pipeline on Weather dataset")
        initial_count = raw_df.count()

        # 1. Validation & Normalization
        df = Normalizer.normalize_timestamps(raw_df, timestamp_col="timestamp")
        df = Normalizer.standardize_strings(df, ["station_id"])
        df = df.filter(F.col("timestamp").isNotNull() & F.col("station_id").isNotNull())
        valid_ts_count = df.count()
        invalid_ts_count = initial_count - valid_ts_count

        # 2. Deduplication
        df_dedup = df.dropDuplicates(subset=["station_id", "timestamp"])
        duplicate_count = valid_ts_count - df_dedup.count()

        # 3. Missing values audit & imputation
        initial_missing = MissingValueHandler.audit_missing_values(df_dedup, WEATHER_COLS)
        domain_limits = self.dq_cfg.get("domain_limits", {})
        df_domain, domain_outliers = OutlierDetector.audit_and_handle_domain_limits(
            df_dedup, domain_limits, action="cap"
        )
        cleaned_df = MissingValueHandler.impute_missing(
            df_domain, numeric_cols=WEATHER_COLS, partition_col="station_id", order_col="timestamp"
        )
        post_missing = MissingValueHandler.audit_missing_values(cleaned_df, WEATHER_COLS)

        final_count = cleaned_df.count()
        quality_report = {
            "dataset": "Weather",
            "initial_records": initial_count,
            "final_records": final_count,
            "invalid_records": invalid_ts_count,
            "duplicate_records": duplicate_count,
            "domain_outliers_corrected": domain_outliers,
            "missing_before_cleaning": initial_missing,
            "missing_after_cleaning": post_missing,
            "retention_percentage": round((final_count / initial_count * 100.0), 2) if initial_count > 0 else 0.0,
        }
        return cleaned_df, quality_report

    def process_stations(self, raw_df: DataFrame) -> Tuple[DataFrame, Dict[str, Any]]:
        """
        Process Station Metadata.
        """
        logger.info("Executing Data Quality Pipeline on Station Metadata")
        initial_count = raw_df.count()
        
        # Validate columns
        is_valid, missing_cols = SchemaValidator.validate_columns(
            raw_df, REQUIRED_STATION_COLS, dataset_name="Station Metadata"
        )
        if not is_valid:
            raise ValueError(f"Station Metadata missing columns: {missing_cols}")

        df = Normalizer.standardize_strings(raw_df, ["station_id", "city", "state", "station_type"])
        df = df.filter(F.col("station_id").isNotNull() & F.col("latitude").isNotNull() & F.col("longitude").isNotNull())
        df_dedup = df.dropDuplicates(subset=["station_id"])
        
        final_count = df_dedup.count()
        quality_report = {
            "dataset": "Stations",
            "initial_records": initial_count,
            "final_records": final_count,
            "duplicate_records": initial_count - final_count,
            "retention_percentage": round((final_count / initial_count * 100.0), 2) if initial_count > 0 else 0.0,
        }
        return df_dedup, quality_report

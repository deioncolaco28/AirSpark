"""
Schema and data integrity validation for AirSpark datasets.
"""
from typing import List, Tuple
from pyspark.sql import DataFrame
from app.utils.logging import get_logger

logger = get_logger("AirSpark.Quality.Validation")


class SchemaValidator:
    """Validates presence and basic types of required DataFrame columns."""

    @staticmethod
    def validate_columns(df: DataFrame, required_cols: List[str], dataset_name: str = "Dataset") -> Tuple[bool, List[str]]:
        """
        Validate that all required columns are present in DataFrame.
        Returns (is_valid, missing_columns).
        """
        existing_cols = set(df.columns)
        missing_cols = [col for col in required_cols if col not in existing_cols]

        if missing_cols:
            error_msg = f"Validation failed for {dataset_name}: missing required columns: {missing_cols}"
            logger.error(error_msg)
            return False, missing_cols

        logger.info(f"Schema validation passed for {dataset_name} ({len(existing_cols)} columns verified)")
        return True, []

    @staticmethod
    def check_empty(df: DataFrame, dataset_name: str = "Dataset") -> bool:
        """Check if DataFrame contains any records."""
        is_empty = df.rdd.isEmpty() if hasattr(df, "rdd") else (df.limit(1).count() == 0)
        if is_empty:
            logger.warning(f"{dataset_name} is completely empty!")
        return is_empty

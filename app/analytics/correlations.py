"""
Weather-Pollutant Correlation Engine for AirSpark.
Computes Pearson correlation matrix between pollutants and weather parameters.
"""
from typing import Dict, List, Any
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from app.processing.schemas import POLLUTANT_COLS, WEATHER_COLS
from app.utils.logging import get_logger

logger = get_logger("AirSpark.Analytics.Correlations")


class CorrelationAnalyzer:
    """Computes correlation matrices for pollutants and meteorological factors."""

    @staticmethod
    def compute_correlation_matrix(df: DataFrame) -> pd.DataFrame:
        """
        Compute Pearson correlation matrix between all numeric pollutants and weather parameters in a single pass.
        Returns a Pandas DataFrame representing the correlation matrix.
        """
        logger.info("Computing Pearson correlation matrix")
        numeric_candidates = POLLUTANT_COLS + WEATHER_COLS + ["aqi"]
        active_cols = [c for c in numeric_candidates if c in df.columns]

        clean_df = df.select(active_cols).dropna()
        if clean_df.limit(5).count() < 5:
            logger.warning("Insufficient non-null records to compute correlation matrix")
            return pd.DataFrame(index=active_cols, columns=active_cols).fillna(0.0)

        # Single-pass conversion to pandas for vector correlation
        pandas_numeric = clean_df.toPandas()
        corr_matrix = pandas_numeric.corr(method="pearson").round(3)
        return corr_matrix.fillna(0.0)

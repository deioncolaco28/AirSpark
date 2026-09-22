"""
Distributed AQI Calculation Engine for Apache Spark.
Calculates individual pollutant sub-indices, overall AQI, dominant pollutant,
and AQI categories natively using Spark SQL Catalyst expressions for maximum distributed performance.
"""
from typing import List, Optional, Dict, Any, Tuple
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType

from app.config.settings import get_settings
from app.processing.schemas import POLLUTANT_COLS
from app.utils.logging import get_logger

logger = get_logger("AirSpark.AQI.Calculator")


class AQICalculator:
    """Computes distributed AQI metrics using native Spark SQL expressions."""

    def __init__(self, standard: Optional[str] = None):
        self.settings = get_settings()
        self.standard = standard or self.settings.aqi_standard
        std_config = self.settings.aqi_config.get(self.standard, {})
        self.breakpoints_map: Dict[str, Any] = std_config.get("breakpoints", {})
        self.categories: List[Dict[str, Any]] = std_config.get("categories", [])

    def _build_sub_index_expr(self, col_name: str, pollutant: str) -> Tuple[F.Column, F.Column, F.Column]:
        """
        Construct native Spark SQL Column expressions for:
        (standard_sub_index [0-500], is_out_of_range_bool, extrapolated_sub_index)
        """
        pollutant_key = pollutant.lower().replace(".", "_")
        breakpoints = self.breakpoints_map.get(pollutant_key, [])

        if not breakpoints:
            return F.lit(None).cast(DoubleType()), F.lit(False), F.lit(None).cast(DoubleType())

        expr = None
        for bp in breakpoints:
            c_low, c_high, i_low, i_high = float(bp[0]), float(bp[1]), float(bp[2]), float(bp[3])
            # Piecewise linear formula: ((i_high - i_low) / (c_high - c_low)) * (conc - c_low) + i_low
            slope = (i_high - i_low) / (c_high - c_low) if (c_high - c_low) != 0 else 0.0
            bracket_val = F.round(slope * (F.col(col_name) - c_low) + i_low, 1)

            cond = (F.col(col_name) >= c_low) & (F.col(col_name) <= c_high)
            if expr is None:
                expr = F.when(cond, bracket_val)
            else:
                expr = expr.when(cond, bracket_val)

        # Handle top bracket exceedance
        last_bp = breakpoints[-1]
        c_low, c_high, i_low, i_high = float(last_bp[0]), float(last_bp[1]), float(last_bp[2]), float(last_bp[3])
        slope = (i_high - i_low) / (c_high - c_low) if (c_high - c_low) != 0 else 0.0

        # Standard AQI is strictly capped at top index bracket (500.0)
        standard_expr = expr.when(F.col(col_name) > c_high, F.lit(i_high)) if expr is not None else F.lit(None).cast(DoubleType())
        
        # Out of range condition
        out_of_range_cond = (F.col(col_name) > c_high)

        # Extrapolated research sub-index
        extrapolated_val = F.round(slope * (F.col(col_name) - c_low) + i_low, 1)
        extrapolated_expr = expr.when(F.col(col_name) > c_high, extrapolated_val) if expr is not None else F.lit(None).cast(DoubleType())

        return standard_expr, out_of_range_cond, extrapolated_expr

    def _build_category_expr(self, aqi_col: str = "aqi", out_of_range_col: str = "is_aqi_out_of_range") -> F.Column:
        """Construct native Spark SQL Column expression for standard AQI categories."""
        expr = None
        for cat in self.categories:
            c_min, c_max, name = float(cat["aqi_min"]), float(cat["aqi_max"]), str(cat["name"])
            cond = (F.col(aqi_col) >= c_min) & (F.col(aqi_col) <= c_max)
            if expr is None:
                expr = F.when(cond, F.lit(name))
            else:
                expr = expr.when(cond, F.lit(name))

        top_category_name = self.categories[-1]["name"] if self.categories else "Hazardous"
        if expr is not None:
            expr = (
                F.when(F.col(out_of_range_col) == True, F.lit(top_category_name))
                .otherwise(F.coalesce(expr, F.lit("Unknown")))
            )
        else:
            expr = F.lit("Unknown")

        return expr

    def calculate_aqi_df(self, df: DataFrame) -> DataFrame:
        """
        Add standard pollutant sub-indices, overall AQI (0-500), dominant pollutant,
        out-of-range validity flags, and AQI category to DataFrame.
        """
        logger.info(f"Computing AQI across DataFrame using '{self.standard}' standard (Native Catalyst Expressions)")
        result_df = df
        sub_index_cols = []
        out_of_range_conds = []
        extrapolated_sub_cols = []

        # 1. Compute sub-index and exceedance flags for each available pollutant natively
        for pol in POLLUTANT_COLS:
            if pol in df.columns:
                sub_col_name = f"sub_index_{pol}"
                extrap_col_name = f"extrapolated_sub_{pol}"
                
                std_expr, oor_cond, extrap_expr = self._build_sub_index_expr(pol, pol)
                result_df = result_df.withColumn(sub_col_name, std_expr)
                result_df = result_df.withColumn(extrap_col_name, extrap_expr)
                
                sub_index_cols.append(sub_col_name)
                extrapolated_sub_cols.append(extrap_col_name)
                out_of_range_conds.append(oor_cond)

        # 2. Overall standard AQI is the maximum of all computed sub-indices bounded strictly to [0.0, 500.0]
        if sub_index_cols:
            greatest_sub = F.greatest(*[F.coalesce(F.col(c), F.lit(0.0)) for c in sub_index_cols])
            result_df = result_df.withColumn("aqi", F.least(F.lit(500.0), F.greatest(F.lit(0.0), greatest_sub)))
            
            # Extrapolated research AQI
            greatest_extrap = F.greatest(*[F.coalesce(F.col(c), F.lit(0.0)) for c in extrapolated_sub_cols])
            result_df = result_df.withColumn("extrapolated_aqi", greatest_extrap)
        else:
            result_df = result_df.withColumn("aqi", F.lit(None).cast(DoubleType()))
            result_df = result_df.withColumn("extrapolated_aqi", F.lit(None).cast(DoubleType()))

        # 3. Overall out-of-range exceedance flag
        if out_of_range_conds:
            combined_oor = out_of_range_conds[0]
            for cond in out_of_range_conds[1:]:
                combined_oor = combined_oor | cond
            result_df = result_df.withColumn("is_aqi_out_of_range", combined_oor)
            result_df = result_df.withColumn("aqi_exceedance_flag", combined_oor)
        else:
            result_df = result_df.withColumn("is_aqi_out_of_range", F.lit(False))
            result_df = result_df.withColumn("aqi_exceedance_flag", F.lit(False))


        # 4. Determine dominant pollutant
        dominant_expr = F.lit("None")
        for pol in reversed(POLLUTANT_COLS):
            sub_col = f"sub_index_{pol}"
            if sub_col in sub_index_cols:
                dominant_expr = F.when(
                    (F.col(sub_col).isNotNull()) & (F.col(sub_col) == F.col("aqi")) & (F.col("aqi") > 0),
                    F.lit(pol.upper().replace("_", "."))
                ).otherwise(dominant_expr)

        result_df = result_df.withColumn("dominant_pollutant", dominant_expr)

        # 5. Determine AQI Category
        result_df = result_df.withColumn("aqi_category", self._build_category_expr("aqi", "is_aqi_out_of_range"))

        # Clean up temporary extrapolated sub-columns from main schema
        for c in extrapolated_sub_cols:
            result_df = result_df.drop(c)

        logger.info("AQI calculation successfully applied to DataFrame with strict standard range enforcement")
        return result_df

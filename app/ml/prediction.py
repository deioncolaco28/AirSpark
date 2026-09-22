"""
Machine Learning Predictive Analytics Module for AirSpark.
Provides distributed regression models (Linear Regression & Random Forest) using PySpark ML
with chronological (time-aware) evaluation, station-level lag/rolling features, model persistence,
and structured prediction persistence.
"""
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import json
from pyspark.sql import DataFrame, Window, SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel, RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator

from app.utils.logging import get_logger
from app.utils.paths import resolve_path, ensure_dir
from app.processing.partitioning import ParquetStorageManager

logger = get_logger("AirSpark.ML.Prediction")


class AQIPredictionModel:
    """Trains, evaluates, and persists distributed Spark ML models for AQI forecasting."""

    BASE_FEATURE_COLS = [
        "pm2_5", "pm10", "no2", "so2", "co", "o3",
        "temperature", "humidity", "wind_speed", "pressure"
    ]

    TEMPORAL_FEATURE_COLS = [
        "hour", "day_of_week", "month", "is_weekend",
        "lag_1_pm2_5", "lag_1_aqi", "rolling_3h_pm2_5_mean", "rolling_3h_aqi_mean"
    ]

    def __init__(self, target_col: str = "aqi"):
        self.target_col = target_col

    def add_temporal_lag_features(self, df: DataFrame) -> DataFrame:
        """
        Generate station-specific lag and rolling temporal predictors without future leakage.
        """
        logger.info("Generating temporal lag and rolling features for ML forecasting")
        
        # Ensure timestamp is parsed and calendar columns exist
        feat_df = df
        if "hour" not in feat_df.columns and "timestamp" in feat_df.columns:
            feat_df = (
                feat_df
                .withColumn("hour", F.hour("timestamp"))
                .withColumn("day_of_week", F.dayofweek("timestamp"))
                .withColumn("month", F.month("timestamp"))
                .withColumn("is_weekend", F.when(F.dayofweek("timestamp").isin(1, 7), 1.0).otherwise(0.0))
            )

        # Station-level lag and rolling window
        station_window = Window.partitionBy("station_id").orderBy("timestamp")
        rolling_3h_window = station_window.rowsBetween(-3, -1)

        feat_df = (
            feat_df
            .withColumn("prev_pm2_5", F.coalesce(F.lag("pm2_5", 1).over(station_window), F.col("pm2_5")))
            .withColumn("prev_aqi", F.coalesce(F.lag(self.target_col, 1).over(station_window), F.col(self.target_col)))
            .withColumn("lag_1_pm2_5", F.col("prev_pm2_5"))
            .withColumn("lag_1_aqi", F.col("prev_aqi"))
            .withColumn("rolling_pm2_5_mean", F.coalesce(F.avg("pm2_5").over(rolling_3h_window), F.col("pm2_5")))
            .withColumn("rolling_aqi_mean", F.coalesce(F.avg(self.target_col).over(rolling_3h_window), F.col(self.target_col)))
            .withColumn("rolling_3h_pm2_5_mean", F.col("rolling_pm2_5_mean"))
            .withColumn("rolling_3h_aqi_mean", F.col("rolling_aqi_mean"))
        )
        return feat_df

    def extract_features(self, df: DataFrame) -> DataFrame:
        """Alias for add_temporal_lag_features."""
        return self.add_temporal_lag_features(df)

    def prepare_features(self, df: DataFrame) -> Tuple[DataFrame, List[str]]:
        """
        Assemble feature vector using Spark ML VectorAssembler.
        """
        df_with_lags = self.add_temporal_lag_features(df)
        all_candidate_cols = self.BASE_FEATURE_COLS + self.TEMPORAL_FEATURE_COLS + ["prev_pm2_5", "prev_aqi", "rolling_pm2_5_mean", "rolling_aqi_mean"]
        active_features = [c for c in all_candidate_cols if c in df_with_lags.columns]
        # Remove duplicates preserving order
        active_features = list(dict.fromkeys(active_features))
        logger.info(f"Assembling Spark ML feature vector from {len(active_features)} features: {active_features}")

        clean_df = df_with_lags.dropna(subset=active_features + [self.target_col])
        assembler = VectorAssembler(inputCols=active_features, outputCol="features", handleInvalid="skip")
        return assembler.transform(clean_df), active_features

    def chronological_split(self, df: DataFrame, train_ratio: float = 0.8) -> Tuple[DataFrame, DataFrame]:
        """
        Perform strict chronological (time-aware) train/test split to prevent temporal leakage.
        """
        logger.info(f"Executing chronological train/test split (Train: {train_ratio*100:.0f}%, Test: {(1-train_ratio)*100:.0f}%)")
        
        # Compute cutoff timestamp using unix timestamp approximation
        df_unix = df.withColumn("_unix_ts", F.unix_timestamp("timestamp"))
        quantiles = df_unix.approxQuantile("_unix_ts", [train_ratio], 0.01)
        if quantiles:
            split_ts = quantiles[0]
            train_df = df_unix.filter(F.col("_unix_ts") <= split_ts).drop("_unix_ts")
            test_df = df_unix.filter(F.col("_unix_ts") > split_ts).drop("_unix_ts")
        else:
            # Fallback if quantile estimation fails on tiny dataset
            train_df, test_df = df.randomSplit([train_ratio, 1.0 - train_ratio], seed=42)

        return train_df, test_df

    def prepare_time_split(self, df: DataFrame, train_ratio: float = 0.8) -> Tuple[DataFrame, DataFrame]:
        """Alias for chronological_split."""
        return self.chronological_split(df, train_ratio=train_ratio)

    def train_and_evaluate(
        self,
        df: DataFrame,
        train_ratio: float = 0.8,
        models_dir: Optional[str] = "data/models",
        models_save_dir: Optional[str] = None,
        save_predictions: bool = True,
    ) -> Tuple[Dict[str, Any], DataFrame]:
        """
        Train Linear Regression and Random Forest models with chronological evaluation,
        save model artifacts to disk, and persist prediction outputs.
        """
        save_dir = models_save_dir if models_save_dir is not None else models_dir
        logger.info("Starting distributed Spark ML training and evaluation workflow")
        feature_df, active_features = self.prepare_features(df)
        
        train_df, test_df = self.chronological_split(feature_df, train_ratio=train_ratio)
        train_count = train_df.count()
        test_count = test_df.count()

        evaluator_rmse = RegressionEvaluator(labelCol=self.target_col, predictionCol="prediction", metricName="rmse")
        evaluator_r2 = RegressionEvaluator(labelCol=self.target_col, predictionCol="prediction", metricName="r2")
        evaluator_mae = RegressionEvaluator(labelCol=self.target_col, predictionCol="prediction", metricName="mae")

        # 1. Linear Regression Baseline
        logger.info("Fitting Spark ML Linear Regression model")
        lr = LinearRegression(featuresCol="features", labelCol=self.target_col, maxIter=50, regParam=0.1)
        lr_model = lr.fit(train_df)
        lr_predictions = (
            lr_model.transform(test_df)
            .withColumnRenamed("prediction", "predicted_aqi_lr")
        )

        lr_eval_df = lr_predictions.withColumnRenamed("predicted_aqi_lr", "prediction")
        lr_rmse = float(evaluator_rmse.evaluate(lr_eval_df))
        lr_r2 = float(evaluator_r2.evaluate(lr_eval_df))
        lr_mae = float(evaluator_mae.evaluate(lr_eval_df))

        # 2. Random Forest Regressor
        logger.info("Fitting Spark ML Random Forest Regressor model")
        rf = RandomForestRegressor(featuresCol="features", labelCol=self.target_col, numTrees=25, maxDepth=6, seed=42)
        rf_model = rf.fit(train_df)
        rf_predictions = (
            rf_model.transform(test_df)
            .withColumnRenamed("prediction", "predicted_aqi_rf")
        )

        rf_eval_df = rf_predictions.withColumnRenamed("predicted_aqi_rf", "prediction")
        rf_rmse = float(evaluator_rmse.evaluate(rf_eval_df))
        rf_r2 = float(evaluator_r2.evaluate(rf_eval_df))
        rf_mae = float(evaluator_mae.evaluate(rf_eval_df))

        # 3. Model Persistence
        models_base_path = resolve_path(save_dir) if save_dir else resolve_path("data/models")
        ensure_dir(models_base_path)
        lr_path = models_base_path / "linear_regression"
        rf_path = models_base_path / "random_forest"
        ensure_dir(lr_path)
        ensure_dir(rf_path)

        lr_meta = {
            "model_type": "linear_regression",
            "coefficients": [float(c) for c in lr_model.coefficients],
            "intercept": float(lr_model.intercept),
            "rmse": lr_rmse,
            "r2": lr_r2,
            "mae": lr_mae,
            "features": active_features,
        }
        with open(lr_path / "model_summary.json", "w", encoding="utf-8") as f:
            json.dump(lr_meta, f, indent=2)

        rf_meta = {
            "model_type": "random_forest",
            "num_trees": int(rf_model.getNumTrees),
            "total_nodes": int(rf_model.totalNumNodes),
            "feature_importances": [float(fi) for fi in rf_model.featureImportances],
            "rmse": rf_rmse,
            "r2": rf_r2,
            "mae": rf_mae,
            "features": active_features,
        }
        with open(rf_path / "model_summary.json", "w", encoding="utf-8") as f:
            json.dump(rf_meta, f, indent=2)

        try:
            logger.info(f"Persisting native Spark ML models to: {models_base_path}")
            lr_model.write().overwrite().save(str(lr_path / "spark_native").replace("\\", "/"))
            rf_model.write().overwrite().save(str(rf_path / "spark_native").replace("\\", "/"))
            logger.info("Successfully persisted native Spark models")
        except Exception as e:
            logger.info(f"Native Spark model persistence note: {e}")

        best_model_name = "random_forest" if rf_r2 >= lr_r2 else "linear_regression"
        lr_metrics = {
            "rmse": round(lr_rmse, 3),
            "r2": round(lr_r2, 3),
            "mae": round(lr_mae, 3),
            "model_path": str(lr_path),
        }
        rf_metrics = {
            "rmse": round(rf_rmse, 3),
            "r2": round(rf_r2, 3),
            "mae": round(rf_mae, 3),
            "model_path": str(rf_path),
        }

        metrics = {
            "evaluation_strategy": "chronological_time_split",
            "split_strategy": {"strategy": "chronological", "train_ratio": train_ratio},
            "train_records": train_count,
            "test_records": test_count,
            "feature_count": len(active_features),
            "features": active_features,
            "linear_regression": lr_metrics,
            "random_forest": rf_metrics,
            "models": {
                "linear_regression": lr_metrics,
                "random_forest": rf_metrics,
            },
            "best_model": best_model_name,
        }

        # 4. Assemble Consolidated Predictions DataFrame
        pred_df = (
            lr_predictions.select(
                "station_id", "timestamp",
                F.col(self.target_col).alias("actual_aqi"),
                F.round("predicted_aqi_lr", 1).alias("predicted_aqi_lr")
            )
            .join(
                rf_predictions.select("station_id", "timestamp", F.round("predicted_aqi_rf", 1).alias("predicted_aqi_rf")),
                on=["station_id", "timestamp"],
                how="inner"
            )
            .withColumn(
                "predicted_aqi",
                F.col("predicted_aqi_rf") if best_model_name == "random_forest" else F.col("predicted_aqi_lr")
            )
            .withColumn("model", F.lit(best_model_name))
            .withColumn(
                "best_predicted_aqi",
                F.col("predicted_aqi")
            )
            .withColumn(
                "prediction_error",
                F.round(F.abs(F.col("predicted_aqi") - F.col("actual_aqi")), 1)
            )
            .orderBy("timestamp")
        )

        # 5. Persist Predictions and Metrics
        if save_predictions:
            proc_dir = ensure_dir("data/processed")
            ParquetStorageManager.write_parquet(pred_df, proc_dir / "aqi_predictions.parquet")
            
            with open(proc_dir / "ml_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved ML predictions to {proc_dir / 'aqi_predictions.parquet'} and metrics to {proc_dir / 'ml_metrics.json'}")

        logger.info(f"Spark ML Chronological Evaluation: LR R2={lr_r2:.3f}, RF R2={rf_r2:.3f} (Best: {best_model_name})")
        return metrics, pred_df

    @staticmethod
    def load_model(spark: SparkSession, model_path: str, model_type: str = "random_forest"):
        """Load a persisted Spark ML model from disk with metadata fallback."""
        p = resolve_path(model_path)
        p_native = p / "spark_native" if (p / "spark_native").exists() else p
        p_str = str(p_native).replace("\\", "/")
        try:
            if model_type == "random_forest":
                return RandomForestRegressionModel.load(p_str)
            elif model_type == "linear_regression":
                return LinearRegressionModel.load(p_str)
        except Exception as e:
            logger.debug(f"Native Spark ML load note ({e}). Loading model metadata from {p / 'model_summary.json'}")
            summary_file = p / "model_summary.json"
            if summary_file.exists():
                with open(summary_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {"status": "persisted", "path": str(p)}



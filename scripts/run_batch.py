"""
Master Batch Processing Pipeline Script for AirSpark.
Executes end-to-end Big Data ETL, distributed AQI calculation, spatio-temporal analytics,
hotspot clustering, anomaly classification, Spark ML predictive analytics, and partitioned Parquet persistence.
Supports both Controlled Development and India-Scale Nationwide datasets.
"""
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyspark.sql import functions as F
from app.processing.spark_session import SparkSessionManager
from app.ingestion.batch_ingestion import BatchIngestionEngine
from app.quality.quality_pipeline import DataQualityPipeline
from app.processing.joins import DataIntegrator
from app.processing.transformations import FeatureTransformer
from app.aqi.calculator import AQICalculator
from app.aqi.pollutant_analysis import PollutantAnalyzer
from app.analytics.temporal import TemporalAnalytics
from app.analytics.spatial import SpatialAnalytics
from app.analytics.hotspots import HotspotAnalyzer
from app.analytics.correlations import CorrelationAnalyzer
from app.analytics.anomalies import AnomalyDetector
from app.ml.prediction import AQIPredictionModel
from app.processing.partitioning import ParquetStorageManager
from app.utils.logging import get_logger
from app.utils.paths import resolve_path, ensure_dir
from app.utils.helpers import format_duration

logger = get_logger("AirSpark.BatchPipeline")


def run_batch_pipeline(
    aqi_standard: str = "US_EPA",
    skip_ml: bool = False,
    dataset: str = "dev",
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
):
    start_time = time.perf_counter()

    # Determine input and output paths based on dataset tier
    if dataset == "india":
        in_path = Path(input_dir or "data/raw_india")
        out_path = Path(output_dir or "data/processed_india")
        mod_path = Path(models_dir or "data/models_india")
        pipeline_title = "AirSpark India-Scale CAAQMS Big Data Batch Pipeline"
    else:
        in_path = Path(input_dir or "data/raw")
        out_path = Path(output_dir or "data/processed")
        mod_path = Path(models_dir or "data/models")
        pipeline_title = "AirSpark Controlled Development Batch Processing Pipeline"

    print("\n=======================================================")
    print(f"      {pipeline_title}")
    print("=======================================================\n")
    print(f" [*] Input Dataset Source:    {in_path}")
    print(f" [*] Output Parquet Lake:     {out_path}")
    print(f" [*] Standard:                {aqi_standard}")
    print(f" [*] ML Predictive Analytics: {'Skipped' if skip_ml else 'Enabled'}\n")

    # 1. Spark Session Initialization
    print("[1/10] Initializing PySpark Distributed Engine ...")
    spark = SparkSessionManager.get_session(app_name="AirSpark-Batch-Pipeline")

    # 2. Ingestion
    print("[2/10] Ingesting Multi-Source Heterogeneous Datasets ...")
    ingestion_engine = BatchIngestionEngine(spark)
    try:
        raw_sources = ingestion_engine.ingest_all(
            aq_path=in_path / "air_quality.csv",
            weather_path=in_path / "weather.csv",
            stations_path=in_path / "stations.csv"
        )
    except Exception as e:
        print(f"[-] Ingestion failed: {e}")
        if dataset == "india":
            print("    -> Tip: Run 'python scripts/generate_india_data.py' first to create India-scale data.")
        else:
            print("    -> Tip: Run 'python run.py generate' first to create synthetic sample data.")
        return 1

    aq_raw = raw_sources["air_quality"]
    weather_raw = raw_sources["weather"]
    stations_raw = raw_sources["stations"]
    print(f"       -> Ingested: {aq_raw.count():,} AQ records, {weather_raw.count():,} Weather records, {stations_raw.count():,} Stations")

    # 3. Data Quality Pipeline
    print("[3/10] Executing Data Quality Pipeline (Validation, Deduplication, Imputation, Outlier Flagging) ...")
    dq_pipeline = DataQualityPipeline()
    cleaned_aq, aq_report = dq_pipeline.process_air_quality(aq_raw)
    cleaned_weather, weather_report = dq_pipeline.process_weather(weather_raw)
    cleaned_stations, stations_report = dq_pipeline.process_stations(stations_raw)

    print(f"       -> AQ Quality Score: {aq_report['quality_score']}% (Retention: {aq_report['retention_percentage']}%)")
    print(f"       -> Duplicates Removed: {aq_report['duplicate_records']:,} AQ, {weather_report['duplicate_records']:,} Weather")
    print(f"       -> Statistical Outliers Flagged: {aq_report['statistical_outliers_flagged']:,} records")
    print(f"       -> Domain Limits Corrected: {aq_report['domain_outliers_corrected'].get('total_domain_outliers', 0):,} values")

    # 4. Data Integration
    print("[4/10] Integrating Air Quality, Weather, and Station Metadata ...")
    integrated_df, join_stats = DataIntegrator.integrate_sources(
        cleaned_aq, cleaned_weather, cleaned_stations, temporal_alignment="exact"
    )
    integrated_count = integrated_df.count()
    print(f"       -> Integrated Dataset: {integrated_count:,} unified records across {join_stats['matched_stations']} stations")

    # 5. Feature Engineering
    print("[5/10] Applying Temporal & Meteorological Feature Engineering ...")
    transformed_df = FeatureTransformer.add_temporal_features(integrated_df)
    transformed_df = FeatureTransformer.add_ratio_features(transformed_df)

    # 6. AQI Calculation Engine
    print(f"[6/10] Computing Standard-Compliant AQI ({aqi_standard}) & Dominant Pollutants ...")
    aqi_calculator = AQICalculator(standard=aqi_standard)
    aqi_df = aqi_calculator.calculate_aqi_df(transformed_df)

    # Compute AQI distribution summary
    aqi_stats_row = aqi_df.select(
        F.round(F.min("aqi"), 1).alias("min_aqi"),
        F.round(F.max("aqi"), 1).alias("max_aqi"),
        F.round(F.avg("aqi"), 1).alias("mean_aqi"),
        F.sum(F.when(F.col("is_aqi_out_of_range"), 1).otherwise(0)).alias("out_of_range_count")
    ).first()
    aqi_stats = aqi_stats_row.asDict() if aqi_stats_row else {}
    print(f"       -> Standard AQI: Mean={aqi_stats.get('mean_aqi')}, Min={aqi_stats.get('min_aqi')}, Max={aqi_stats.get('max_aqi')} (Out-of-range: {aqi_stats.get('out_of_range_count', 0)})")

    # 7. Spatial & Temporal Analytics & Hotspots
    print("[7/10] Running Distributed Temporal & Spatial Analytics ...")
    hourly_profile_df = TemporalAnalytics.compute_hourly_profile(aqi_df)
    daily_analytics_df = TemporalAnalytics.compute_daily_analytics(aqi_df)
    monthly_analytics_df = TemporalAnalytics.compute_monthly_analytics(aqi_df)
    seasonal_analytics_df = TemporalAnalytics.compute_seasonal_analytics(aqi_df)
    station_summary_df = SpatialAnalytics.compute_station_summary(aqi_df)
    city_summary_df = SpatialAnalytics.compute_city_summary(aqi_df)
    state_summary_df = SpatialAnalytics.compute_state_summary(aqi_df)
    hotspots_df = HotspotAnalyzer.identify_hotspots(aqi_df)

    # 8. Correlations & Explainable Anomalies
    print("[8/10] Computing Weather-Pollutant Correlation Matrix & Classifying Anomalies ...")
    corr_pandas_df = CorrelationAnalyzer.compute_correlation_matrix(aqi_df)
    classified_df = AnomalyDetector.detect_and_classify_anomalies(aqi_df)
    dominance_df = PollutantAnalyzer.compute_dominance_distribution(aqi_df)

    # 9. Optional Spark ML Predictive Analytics
    ml_metrics = {}
    if not skip_ml:
        print("[9/10] Training Distributed Spark ML Forecasting Models (Time-Aware Split) ...")
        ml_model = AQIPredictionModel(target_col="aqi")
        ensure_dir(str(mod_path))
        ensure_dir(str(out_path))
        ml_metrics, _ = ml_model.train_and_evaluate(
            aqi_df,
            train_ratio=0.8,
            models_dir=str(mod_path),
            predictions_path=str(out_path / "aqi_predictions.parquet"),
            save_predictions=True
        )
        print(f"       -> Linear Regression: RMSE={ml_metrics['linear_regression']['rmse']}, R2={ml_metrics['linear_regression']['r2']}")
        print(f"       -> Random Forest:     RMSE={ml_metrics['random_forest']['rmse']}, R2={ml_metrics['random_forest']['r2']}")
    else:
        print("[9/10] Spark ML Training Skipped by CLI Flag")

    # 10. Storage & Partitioned Parquet Persistence
    print("[10/10] Persisting Processed Tables to Parquet & Summary Scorecards ...")
    proc_dir = ensure_dir(str(out_path))

    # Save partitioned main integrated dataset
    ParquetStorageManager.write_parquet(
        classified_df,
        output_path=str(proc_dir / "integrated_aqi.parquet"),
        partition_by=["year", "month"],
        mode="overwrite"
    )
    ParquetStorageManager.write_parquet(daily_analytics_df, str(proc_dir / "daily_analytics.parquet"), mode="overwrite")
    ParquetStorageManager.write_parquet(hourly_profile_df, str(proc_dir / "hourly_profile.parquet"), mode="overwrite")
    ParquetStorageManager.write_parquet(station_summary_df, str(proc_dir / "station_summary.parquet"), mode="overwrite")
    ParquetStorageManager.write_parquet(hotspots_df, str(proc_dir / "hotspots.parquet"), mode="overwrite")

    # Compute category counts
    cat_counts_df = aqi_df.groupBy("aqi_category").count().collect()
    category_distribution = {r["aqi_category"]: r["count"] for r in cat_counts_df}

    # Save summary metadata JSON for fast dashboard loading
    metadata = {
        "dataset_tier": dataset,
        "pipeline_execution_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "aqi_standard": aqi_standard,
        "total_records": integrated_count,
        "aqi_statistics": aqi_stats,
        "category_distribution": category_distribution,
        "quality_report_aq": aq_report,
        "quality_report_weather": weather_report,
        "join_stats": join_stats,
        "ml_metrics": ml_metrics,
        "correlations": corr_pandas_df.to_dict(),
        "dominance_distribution": [r.asDict() for r in dominance_df.collect()],
        "seasonal_summary": [r.asDict() for r in seasonal_analytics_df.collect()],
        "city_summary": [r.asDict() for r in city_summary_df.collect()],
        "state_summary": [r.asDict() for r in state_summary_df.collect()] if state_summary_df is not None else [],
    }

    with open(proc_dir / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    total_duration = time.perf_counter() - start_time
    print("\n-------------------------------------------------------")
    print(f" [SUCCESS] Batch Pipeline ({dataset.upper()}) completed in {format_duration(total_duration)}!")
    print(f"           Integrated Dataset Records: {integrated_count:,}")
    print(f"           Parquet Datasets saved to: {proc_dir}")
    print("-------------------------------------------------------\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run AirSpark Master Big Data Batch Processing Pipeline")
    parser.add_argument("--dataset", type=str, default="dev", choices=["dev", "india"], help="Dataset tier: 'dev' (10-station) or 'india' (nationwide CAAQMS)")
    parser.add_argument("--aqi-standard", type=str, default="US_EPA", choices=["US_EPA", "INDIA_CPCB"], help="AQI Standard")
    parser.add_argument("--skip-ml", action="store_true", help="Skip Spark ML model training")
    parser.add_argument("--input-dir", type=str, default=None, help="Custom raw input directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom processed output directory")
    parser.add_argument("--models-dir", type=str, default=None, help="Custom models directory")
    args = parser.parse_args()

    sys.exit(run_batch_pipeline(
        aqi_standard=args.aqi_standard,
        skip_ml=args.skip_ml,
        dataset=args.dataset,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        models_dir=args.models_dir
    ))


if __name__ == "__main__":
    main()

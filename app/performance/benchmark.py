"""
Performance Benchmarking Engine for AirSpark.
Executes systematic scalability, throughput, and partitioning experiments on real generated datasets.
"""
import time
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from pyspark.sql import SparkSession

from app.config.settings import get_settings
from app.performance.dataset_generator import SyntheticDataGenerator
from app.quality.quality_pipeline import DataQualityPipeline
from app.processing.joins import DataIntegrator
from app.aqi.calculator import AQICalculator
from app.analytics.temporal import TemporalAnalytics
from app.analytics.spatial import SpatialAnalytics
from app.performance.metrics import BenchmarkResult, save_benchmark_report
from app.utils.logging import get_logger
from app.utils.paths import resolve_path, ensure_dir

logger = get_logger("AirSpark.Performance.Benchmark")


class BenchmarkRunner:
    """Executes distributed scalability benchmarks."""

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.settings = get_settings()
        self.generator = SyntheticDataGenerator(seed=42)
        self.dq_pipeline = DataQualityPipeline()
        self.aqi_calculator = AQICalculator()

    def run_spark_pipeline(self, aq_df_pandas: pd.DataFrame, weather_df_pandas: pd.DataFrame, stations_df_pandas: pd.DataFrame, num_partitions: int = 4) -> Dict[str, float]:
        """
        Run Spark pipeline on in-memory Pandas inputs converted to Spark DataFrames with timed stages.
        """
        # Ingestion & Partitioning
        t0 = time.perf_counter()
        aq_spark = self.spark.createDataFrame(aq_df_pandas).repartition(num_partitions)
        weather_spark = self.spark.createDataFrame(weather_df_pandas).repartition(num_partitions)
        stations_spark = self.spark.createDataFrame(stations_df_pandas)

        # 1. Data Quality & Integration
        cleaned_aq, _ = self.dq_pipeline.process_air_quality(aq_spark)
        cleaned_weather, _ = self.dq_pipeline.process_weather(weather_spark)
        cleaned_stations, _ = self.dq_pipeline.process_stations(stations_spark)
        integrated_df, _ = DataIntegrator.integrate_sources(cleaned_aq, cleaned_weather, cleaned_stations)
        # Force Spark action to measure ETL time accurately
        _ = integrated_df.count()
        t1 = time.perf_counter()
        etl_time = t1 - t0

        # 2. AQI Calculation
        t_aqi_start = time.perf_counter()
        aqi_df = self.aqi_calculator.calculate_aqi_df(integrated_df)
        _ = aqi_df.count()
        t_aqi_end = time.perf_counter()
        aqi_time = t_aqi_end - t_aqi_start

        # 3. Analytics Aggregations (Temporal + Spatial)
        t_ana_start = time.perf_counter()
        daily_df = TemporalAnalytics.compute_daily_analytics(aqi_df)
        station_df = SpatialAnalytics.compute_station_summary(aqi_df)
        _ = daily_df.count()
        _ = station_df.count()
        t_ana_end = time.perf_counter()
        analytics_time = t_ana_end - t_ana_start

        total_time = time.perf_counter() - t0

        return {
            "etl_time": round(etl_time, 3),
            "aqi_time": round(aqi_time, 3),
            "analytics_time": round(analytics_time, 3),
            "total_time": round(total_time, 3),
        }

    def run_pandas_baseline(self, aq_df: pd.DataFrame, weather_df: pd.DataFrame, stations_df: pd.DataFrame) -> float:
        """
        Measure baseline execution time using single-threaded Pandas implementation.
        """
        t0 = time.perf_counter()
        
        # Deduplication
        aq_clean = aq_df.drop_duplicates(subset=["station_id", "timestamp"]).copy()
        weather_clean = weather_df.drop_duplicates(subset=["station_id", "timestamp"]).copy()
        
        # Fill missing
        aq_clean = aq_clean.ffill().bfill()
        weather_clean = weather_clean.ffill().bfill()

        # Join
        merged = pd.merge(aq_clean, weather_clean, on=["station_id", "timestamp"], how="left")
        merged = pd.merge(merged, stations_df, on="station_id", how="left")

        # Basic AQI proxy calculation
        merged["pm2_5_clean"] = merged["pm2_5"].clip(lower=0, upper=500)
        merged["aqi_approx"] = merged["pm2_5_clean"] * 1.5

        # Group by station & date
        merged["date"] = pd.to_datetime(merged["timestamp"]).dt.date
        _ = merged.groupby(["station_id", "date"])["aqi_approx"].mean()

        total_time = time.perf_counter() - t0
        return round(total_time, 3)

    def run_scalability_benchmark(self) -> List[BenchmarkResult]:
        """
        Run scalability benchmarks across Small, Medium, and Large datasets.
        """
        logger.info("Starting AirSpark Scalability & Performance Benchmark Suite")
        results: List[BenchmarkResult] = []

        # Configurations for benchmark sizes
        configs = [
            {"name": "Small", "num_stations": 5, "days": 7, "freq": 60, "partitions": 4},
            {"name": "Medium", "num_stations": 15, "days": 21, "freq": 30, "partitions": 4},
            {"name": "Large", "num_stations": 30, "days": 45, "freq": 15, "partitions": 8},
        ]

        for cfg in configs:
            name = cfg["name"]
            logger.info(f"--- Running Benchmark for: {name} Dataset ---")
            
            # Generate dataset
            aq_df, weather_df, stations_df = self.generator.generate_datasets(
                num_stations=cfg["num_stations"],
                days=cfg["days"],
                frequency_minutes=cfg["freq"],
                inject_quality_issues=True
            )

            record_count = len(aq_df)
            size_mb = round((aq_df.memory_usage(deep=True).sum() + weather_df.memory_usage(deep=True).sum()) / (1024 * 1024), 2)

            # Spark execution
            spark_metrics = self.run_spark_pipeline(aq_df, weather_df, stations_df, num_partitions=cfg["partitions"])
            throughput = round(record_count / spark_metrics["total_time"], 1) if spark_metrics["total_time"] > 0 else 0.0

            # Pandas reference baseline
            pandas_time = self.run_pandas_baseline(aq_df, weather_df, stations_df)
            runtime_ratio = round(pandas_time / spark_metrics["total_time"], 2) if spark_metrics["total_time"] > 0 else None

            from datetime import timezone
            res = BenchmarkResult(
                dataset_name=name,
                record_count=record_count,
                data_size_mb=size_mb,
                spark_etl_time_sec=spark_metrics["etl_time"],
                spark_aqi_time_sec=spark_metrics["aqi_time"],
                spark_analytics_time_sec=spark_metrics["analytics_time"],
                spark_total_time_sec=spark_metrics["total_time"],
                spark_throughput_records_per_sec=throughput,
                pandas_baseline_total_time_sec=pandas_time,
                pandas_to_spark_runtime_ratio=runtime_ratio,
                partition_count=cfg["partitions"],
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            results.append(res)
            logger.info(f"Finished {name}: {record_count} records | Spark Total: {res.spark_total_time_sec}s | Throughput: {throughput} rec/s")

        # Save to JSON and CSV
        out_json = save_benchmark_report(results)
        
        # Save CSV format
        csv_path = resolve_path("data/benchmarks/benchmark_results.csv")
        ensure_dir(csv_path.parent)
        pd.DataFrame([r.to_dict() for r in results]).to_csv(csv_path, index=False)

        logger.info(f"Benchmark results successfully saved to {out_json} and {csv_path}")
        return results

    def run_partitioning_benchmark(self, partition_counts: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Evaluate processing time across different Spark partition counts.
        """
        logger.info("Starting Partitioning Scalability Experiment")
        parts = partition_counts or [1, 2, 4, 8]
        
        # Generate medium dataset for partitioning tests
        aq_df, weather_df, stations_df = self.generator.generate_datasets(
            num_stations=10, days=14, frequency_minutes=30, inject_quality_issues=False
        )

        partition_results = []
        for p in parts:
            t0 = time.perf_counter()
            spark_metrics = self.run_spark_pipeline(aq_df, weather_df, stations_df, num_partitions=p)
            duration = spark_metrics["total_time"]
            throughput = round(len(aq_df) / duration, 1) if duration > 0 else 0.0

            partition_results.append({
                "partition_count": p,
                "record_count": len(aq_df),
                "duration_sec": duration,
                "throughput_records_per_sec": throughput,
            })
            logger.info(f"Partition Count: {p:02d} | Duration: {duration:.3f}s | Throughput: {throughput} rec/s")

        # Save to JSON
        part_json_path = resolve_path("data/benchmarks/partitioning_results.json")
        ensure_dir(part_json_path.parent)
        pd.DataFrame(partition_results).to_json(part_json_path, orient="records", indent=2)

        return partition_results

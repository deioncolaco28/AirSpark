"""
CLI Script to execute AirSpark Performance & Scalability Benchmarking Suite.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.processing.spark_session import SparkSessionManager
from app.performance.benchmark import BenchmarkRunner
from app.utils.logging import get_logger

logger = get_logger("AirSpark.BenchmarkCLI")


def main():
    print("\n=======================================================")
    print("      AirSpark Scalability & Performance Benchmarking   ")
    print("=======================================================\n")

    spark = SparkSessionManager.get_session(app_name="AirSpark-Benchmark-Suite")
    runner = BenchmarkRunner(spark)

    print("[*] Running Dataset Scalability Benchmarks (Small, Medium, Large) ...\n")
    results = runner.run_scalability_benchmark()

    print("\n----------------------------------------------------------------------------------------------------------------")
    print(f"{'Dataset':<10} | {'Records':<10} | {'Size (MB)':<10} | {'Spark Time':<12} | {'Throughput':<15} | {'Pandas Baseline':<16} | {'Pandas/Spark Ratio'}")
    print("----------------------------------------------------------------------------------------------------------------")
    for r in results:
        pandas_str = f"{r.pandas_baseline_total_time_sec:.3f} s" if r.pandas_baseline_total_time_sec else "N/A"
        ratio_str = f"{r.pandas_to_spark_runtime_ratio:.2f}x" if r.pandas_to_spark_runtime_ratio else "N/A"
        print(f"{r.dataset_name:<10} | {r.record_count:<10} | {r.data_size_mb:<10.2f} | {r.spark_total_time_sec:<10.3f} s | {r.spark_throughput_records_per_sec:<12.1f} r/s | {pandas_str:<16} | {ratio_str}")
    print("----------------------------------------------------------------------------------------------------------------\n")

    print("[*] Running Partitioning Efficiency Experiment ...\n")
    part_results = runner.run_partitioning_benchmark([1, 2, 4, 8])

    print("---------------------------------------------------------------------")
    print(f"{'Partitions':<12} | {'Records':<10} | {'Execution Time':<18} | {'Throughput'}")
    print("---------------------------------------------------------------------")
    for p in part_results:
        print(f"{p['partition_count']:<12} | {p['record_count']:<10} | {p['duration_sec']:<16.3f} s | {p['throughput_records_per_sec']:.1f} r/s")
    print("---------------------------------------------------------------------\n")

    print("[SUCCESS] Benchmark report saved to data/benchmarks/benchmark_results.json")


if __name__ == "__main__":
    main()

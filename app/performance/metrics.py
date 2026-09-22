"""
Performance metrics data structures and summary helpers for AirSpark.
"""
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
from app.utils.paths import resolve_path, ensure_dir


@dataclass
class BenchmarkResult:
    dataset_name: str
    record_count: int
    data_size_mb: float
    spark_etl_time_sec: float
    spark_aqi_time_sec: float
    spark_analytics_time_sec: float
    spark_total_time_sec: float
    spark_throughput_records_per_sec: float
    pandas_baseline_total_time_sec: Optional[float]
    pandas_to_spark_runtime_ratio: Optional[float]
    partition_count: int
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def save_benchmark_report(results: List[BenchmarkResult], output_path: str = "data/benchmarks/benchmark_results.json") -> str:
    """Save benchmark results to JSON file."""
    p = resolve_path(output_path)
    ensure_dir(p.parent)
    
    data = [r.to_dict() for r in results]
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return str(p)

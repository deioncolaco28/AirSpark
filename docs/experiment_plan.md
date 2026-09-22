# AirSpark Experimental Benchmark Plan

This document describes the experimental protocol and research benchmark plan for evaluating AirSpark's distributed performance, throughput, scalability, and stream processing latency.

---

## 1. Experimental Setup & Hypotheses

### Hypothesis 1 (Throughput & Scalability)
As the volume of input records increases from Small ($10^3$) to Medium ($10^4$) and Large ($10^5$), Apache Spark distributed query engine will maintain near-linear throughput (records processed per second), outpacing single-threaded in-memory Pandas baselines as dataset sizes exceed cache capacity.

### Hypothesis 2 (Partitioning Efficiency)
Increasing Spark partition count from 1 to 2, 4, 8, and 16 will reduce wall-clock execution time up to the hardware core saturation point, demonstrating distributed data parallel speedup.

### Hypothesis 3 (Stream Processing Bounded Latency)
Spark Structured Streaming micro-batch processing will sustain sub-second windowed aggregations and real-time alert dispatch under continuous simulated sensor ingestion.

---

## 2. Experimental Configurations

| Experiment ID | Parameter Under Test | Tested Values | Monitored Metrics |
| :--- | :--- | :--- | :--- |
| **EXP-01** | Dataset Scale | Small, Medium, Large | ETL Time, AQI Time, Analytics Time, Total Wall-Clock Time, Throughput (records/s) |
| **EXP-02** | Partitioning Count | 1, 2, 4, 8, 16 partitions | Execution Duration (s), Task Execution Time, Throughput |
| **EXP-03** | Speedup vs Baseline | Spark vs Single-threaded Pandas | Speedup Multiplier ($T_{\text{Pandas}} / T_{\text{Spark}}$) |
| **EXP-04** | Streaming Micro-batch | 2s Interval, 10-min window | Micro-batch processing duration, Alert detection recall |

---

## 3. Reproducibility Instructions

To execute the entire experimental benchmark suite and generate empirical results:

```bash
# 1. Run the benchmark runner
python scripts/run_benchmark.py

# 2. View generated empirical report
cat data/benchmarks/benchmark_results.json
cat data/benchmarks/benchmark_results.csv
```

All benchmark measurements are computed in real-time during execution on the local host and saved to structured JSON/CSV for academic review.

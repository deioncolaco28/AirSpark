# AirSpark Experimental Benchmark Plan

This document describes the experimental protocol and research benchmark plan for evaluating AirSpark's distributed performance, throughput, scalability, and stream processing latency across dataset tiers.

---

## 1. Experimental Setup & Hypotheses

### Hypothesis 1 (Throughput & Scalability Across Dataset Tiers)
As the volume of input records increases from Small Dev ($10^3$) to Medium ($10^4$) and the India-Scale Experimental Dataset tier (44 stations, $3.2 \times 10^4$ records), Apache Spark's distributed query engine will maintain scalable throughput (records processed per second), demonstrating distributed execution speedup in local Spark mode.

### Hypothesis 2 (Partitioning Efficiency)
Increasing Spark partition count from 1 to 2, 4, 8, and 16 will reduce wall-clock execution time up to the hardware core saturation point, demonstrating distributed data parallel speedup in local mode.

### Hypothesis 3 (Stream Processing Bounded Latency)
Spark Structured Streaming micro-batch processing will sustain sub-second windowed aggregations and real-time alert dispatch under continuous simulated sensor ingestion.

---

## 2. Experimental Configurations

| Experiment ID | Parameter Under Test | Tested Values | Monitored Metrics | Scope Note |
| :--- | :--- | :--- | :--- | :--- |
| **EXP-01** | Dataset Scale | Small Dev (10 stn), Medium (20 stn), India-Scale Experimental (44 stn) | ETL Time, AQI Time, Analytics Time, Total Wall-Clock Time, Throughput (records/s) | India-scale refers to the 44-station experimental tier (~32k rows), not the complete national database. |
| **EXP-02** | Partitioning Count | 1, 2, 4, 8, 16 partitions | Execution Duration (s), Task Execution Time, Throughput | Evaluates local JVM multi-threading parallelization. |
| **EXP-03** | Speedup vs Baseline | Spark Local Mode vs Single-threaded Pandas | Execution Time Ratio ($T_{\text{Pandas}} / T_{\text{Spark}}$) | Quantifies overhead vs distributed compute advantages. |
| **EXP-04** | Streaming Micro-batch | 2s Interval, 10-min window | Micro-batch processing duration, Alert detection recall | Evaluates sliding-window stream replay latency. |

---

## 3. Reproducibility Instructions

To execute the entire experimental benchmark suite and generate empirical results:

```powershell
# 1. Run the benchmark runner
python run.py benchmark

# 2. View generated empirical report
cat data/benchmarks/benchmark_results.json
cat data/benchmarks/benchmark_results.csv
```

All benchmark measurements are computed in real-time during execution on the local host and saved to structured JSON/CSV for academic review.

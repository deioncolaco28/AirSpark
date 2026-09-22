"""
Real-time Streaming Simulator and Spark Structured Streaming Runner for AirSpark.
Simulates high-velocity continuous sensor streams with Spark Structured Streaming,
computing real-time AQI, evaluating threshold alerts, and exporting live feeds.
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import json

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.processing.spark_session import SparkSessionManager
from app.streaming.pipeline import StreamingPipeline
from app.streaming.alerts import AlertEvaluator
from app.utils.logging import get_logger
from app.utils.paths import ensure_dir, resolve_path

logger = get_logger("AirSpark.StreamingRunner")


def run_streaming_demo(duration: int = 10, rows_per_sec: int = 5, num_stations: int = 5):
    print("\n=======================================================")
    print("      AirSpark Real-Time Spark Structured Streaming     ")
    print("=======================================================\n")

    print(f"[*] Initializing Spark Structured Streaming Engine ...")
    print(f"    - Duration: {duration} seconds")
    print(f"    - Event Rate: {rows_per_sec} sensor readings/second")
    print(f"    - Simulated Stations: {num_stations}\n")

    spark = SparkSessionManager.get_session(app_name="AirSpark-Streaming-Engine")
    streaming_pipeline = StreamingPipeline(spark)
    alert_evaluator = AlertEvaluator()

    print("[*] Launching Spark Structured Streaming query into real-time memory sink...")
    processed_df = streaming_pipeline.run_bounded_stream(
        duration_seconds=duration,
        rows_per_second=rows_per_sec,
        num_stations=num_stations,
        query_name="airspark_live_feed"
    )

    records = [r.asDict() for r in processed_df.collect()]
    print(f"\n[+] Real-time stream processed {len(records)} events successfully.")

    # Evaluate alerts and window statistics on the streaming records
    alerts = []
    for rec in records:
        alert = alert_evaluator.evaluate_record(rec)
        if alert:
            alerts.append(alert)

    # Compute station-level window aggregations
    stream_pdf = pd.DataFrame(records) if records else pd.DataFrame()
    window_stats = []
    if not stream_pdf.empty and "station_id" in stream_pdf.columns:
        for stn, group in stream_pdf.groupby("station_id"):
            window_stats.append({
                "station_id": stn,
                "readings_count": int(len(group)),
                "avg_aqi": round(float(group["aqi"].mean()), 1) if "aqi" in group else 0.0,
                "max_aqi": round(float(group["aqi"].max()), 1) if "aqi" in group else 0.0,
                "avg_pm2_5": round(float(group["pm2_5"].mean()), 1) if "pm2_5" in group else 0.0,
                "max_pm2_5": round(float(group["pm2_5"].max()), 1) if "pm2_5" in group else 0.0,
            })

    # Save streaming summary for dashboard
    stream_out_dir = ensure_dir("data/streaming/output")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_streamed_records": len(records),
        "total_alerts": len(alerts),
        "alert_type_counts": pd.Series([a["alert_type"] for a in alerts]).value_counts().to_dict() if alerts else {},
        "window_statistics": window_stats,
        "recent_alerts": alerts[:15],
        "latest_records": [{k: (str(v) if isinstance(v, datetime) else v) for k, v in r.items()} for r in records[-10:]] if records else [],
    }

    if not stream_pdf.empty:
        stream_pdf.to_csv(stream_out_dir / "latest_stream.csv", index=False)
        if window_stats:
            pd.DataFrame(window_stats).to_csv(stream_out_dir / "windowed_stats.csv", index=False)

    with open(stream_out_dir / "streaming_alerts.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[+] Real-time alerts detected: {len(alerts)}")
    for a in alerts[:5]:
        print(f"    [ALERT] [{a['severity']}] {a['message']}")

    print("\n-------------------------------------------------------")
    print(" [SUCCESS] Spark Structured Streaming demo finished successfully!")
    print(f"           Live telemetry saved to: {stream_out_dir}")
    print("-------------------------------------------------------\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run AirSpark Real-Time Streaming Demonstration")
    parser.add_argument("--duration", type=int, default=10, help="Streaming execution duration in seconds")
    parser.add_argument("--rate", type=int, default=5, help="Events per second")
    parser.add_argument("--stations", type=int, default=5, help="Number of simulated stations")
    args = parser.parse_args()

    sys.exit(run_streaming_demo(duration=args.duration, rows_per_sec=args.rate, num_stations=args.stations))


if __name__ == "__main__":
    main()

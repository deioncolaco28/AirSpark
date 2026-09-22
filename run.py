"""
AirSpark Master Command Line Orchestrator.
Unified entry point to run all pipeline stages, tests, benchmarks, and dashboard.
Supports both Controlled Development and India-Scale Nationwide CAAQMS datasets.
"""
import sys
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_command(cmd_args):
    """Execute a subprocess command in the current Python environment."""
    return subprocess.run([sys.executable] + cmd_args, cwd=str(PROJECT_ROOT)).returncode


def main():
    parser = argparse.ArgumentParser(
        description="AirSpark: A Scalable Big Data Analytics Framework for Multi-Source Air Quality Monitoring and Spatio-Temporal AQI Analysis Using Apache Spark"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommands
    subparsers.add_parser("validate", help="Validate system dependencies, Python, Java, PySpark")
    
    gen_p = subparsers.add_parser("generate", help="Generate multi-source synthetic datasets (Controlled or India-scale)")
    gen_p.add_argument("--dataset", type=str, default="dev", choices=["dev", "india"], help="Dataset tier: 'dev' (10 stations) or 'india' (44 stations nationwide)")
    gen_p.add_argument("--stations", type=int, default=10, help="Number of monitoring stations (for dev tier)")
    gen_p.add_argument("--days", type=int, default=30, help="Number of days of data")
    gen_p.add_argument("--frequency", type=int, default=60, help="Frequency in minutes")
    gen_p.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation")

    gen_ind_p = subparsers.add_parser("generate-india", help="Generate authentic India-scale CAAQMS multi-station dataset")
    gen_ind_p.add_argument("--days", type=int, default=30, help="Number of days of data")
    gen_ind_p.add_argument("--frequency", type=int, default=60, help="Frequency in minutes")
    gen_ind_p.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation")

    batch_p = subparsers.add_parser("batch", help="Run master distributed batch processing pipeline")
    batch_p.add_argument("--dataset", type=str, default="dev", choices=["dev", "india"], help="Target dataset: 'dev' or 'india'")
    batch_p.add_argument("--aqi-standard", type=str, default="US_EPA", choices=["US_EPA", "INDIA_CPCB"], help="AQI Standard (US_EPA or INDIA_CPCB)")
    batch_p.add_argument("--skip-ml", action="store_true", help="Skip Spark ML model training")
    
    stream_p = subparsers.add_parser("streaming", help="Run Spark Structured Streaming live sensor demo")
    stream_p.add_argument("--duration", type=int, default=10, help="Streaming duration in seconds")
    stream_p.add_argument("--rate", type=int, default=5, help="Events per second")
    stream_p.add_argument("--stations", type=int, default=5, help="Number of simulated stations")

    subparsers.add_parser("benchmark", help="Run scalability and performance benchmark suite")
    
    test_p = subparsers.add_parser("test", help="Run comprehensive unit, integration, and end-to-end tests")
    test_p.add_argument("-v", "--verbose", action="store_true", default=True, help="Verbose pytest output")
    
    dash_p = subparsers.add_parser("dashboard", help="Start interactive Streamlit web dashboard")
    dash_p.add_argument("--port", type=int, default=8501, help="Port for Streamlit dashboard")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "validate":
        return run_command(["scripts/validate_environment.py"])
    elif args.command == "generate":
        if args.dataset == "india":
            return run_command([
                "scripts/generate_india_data.py",
                "--days", str(args.days),
                "--frequency", str(args.frequency),
                "--seed", str(args.seed),
            ])
        else:
            return run_command([
                "scripts/generate_sample_data.py",
                "--stations", str(args.stations),
                "--days", str(args.days),
                "--frequency", str(args.frequency),
                "--seed", str(args.seed),
            ])
    elif args.command == "generate-india":
        return run_command([
            "scripts/generate_india_data.py",
            "--days", str(args.days),
            "--frequency", str(args.frequency),
            "--seed", str(args.seed),
        ])
    elif args.command == "batch":
        cmd = ["scripts/run_batch.py", "--dataset", args.dataset, "--aqi-standard", args.aqi_standard]
        if args.skip_ml:
            cmd.append("--skip-ml")
        return run_command(cmd)
    elif args.command == "streaming":
        return run_command([
            "scripts/run_streaming.py",
            "--duration", str(args.duration),
            "--rate", str(args.rate),
            "--stations", str(args.stations),
        ])
    elif args.command == "benchmark":
        return run_command(["scripts/run_benchmark.py"])
    elif args.command == "test":
        flags = ["-v"] if args.verbose else []
        return subprocess.run([sys.executable, "-m", "pytest", "tests"] + flags, cwd=str(PROJECT_ROOT)).returncode
    elif args.command == "dashboard":
        return subprocess.run(["streamlit", "run", "dashboard/app.py", "--server.port", str(args.port)], cwd=str(PROJECT_ROOT)).returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())

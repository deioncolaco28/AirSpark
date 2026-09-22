"""
CLI Script to generate realistic synthetic air-quality and weather datasets for AirSpark.
"""
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.performance.dataset_generator import SyntheticDataGenerator
from app.utils.logging import get_logger
from app.utils.paths import ensure_dir, resolve_path

logger = get_logger("AirSpark.GenerateDataCLI")


def main():
    parser = argparse.ArgumentParser(description="Generate realistic multi-source synthetic datasets for AirSpark")
    parser.add_argument("--stations", type=int, default=10, help="Number of monitoring stations to generate")
    parser.add_argument("--days", type=int, default=14, help="Number of days of continuous time-series data")
    parser.add_argument("--frequency", type=int, default=30, help="Reading frequency in minutes (e.g., 15, 30, 60)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Target output directory")
    parser.add_argument("--clean", action="store_true", help="Generate perfectly clean data without synthetic anomalies")
    
    args = parser.parse_args()

    print(f"\n[*] Generating Synthetic Air Quality Dataset:")
    print(f"    - Stations: {args.stations}")
    print(f"    - Time Horizon: {args.days} days")
    print(f"    - Interval: Every {args.frequency} minutes")
    print(f"    - Random Seed: {args.seed}")
    print(f"    - Target Dir: {args.output_dir}\n")

    generator = SyntheticDataGenerator(seed=args.seed)
    aq_df, weather_df, stations_df = generator.generate_datasets(
        num_stations=args.stations,
        days=args.days,
        frequency_minutes=args.frequency,
        inject_quality_issues=not args.clean
    )

    saved_paths = generator.save_datasets(
        aq_df=aq_df,
        weather_df=weather_df,
        stations_df=stations_df,
        output_dir=args.output_dir
    )

    # Also save a copy in sample dir for immediate fallback
    sample_dir = resolve_path("data/sample")
    ensure_dir(sample_dir)
    generator.save_datasets(aq_df.head(100), weather_df.head(100), stations_df, output_dir="data/sample")

    print("\n[SUCCESS] Datasets generated successfully:")
    for name, path in saved_paths.items():
        print(f"    - {name:<15}: {path}")
    print("\n")


if __name__ == "__main__":
    main()

"""
CLI Script to generate realistic India-Scale multi-source air quality, weather, and station datasets for AirSpark.
"""
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.india_dataset import IndiaScaleDataIngestor
from app.utils.logging import get_logger

logger = get_logger("AirSpark.GenerateIndiaDataCLI")


def main():
    parser = argparse.ArgumentParser(
        description="Generate realistic India-scale CAAQMS multi-source synthetic datasets for AirSpark"
    )
    parser.add_argument("--days", type=int, default=30, help="Number of historical days to generate (default: 30)")
    parser.add_argument("--frequency", type=int, default=60, help="Measurement interval in minutes (default: 60)")
    parser.add_argument("--output-dir", type=str, default="data/raw_india", help="Output directory for CSV files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation")
    parser.add_argument("--no-quality-issues", action="store_true", help="Disable synthetic quality artifacts")

    args = parser.parse_args()

    print("\n=======================================================")
    print("   AirSpark India-Scale CAAQMS Dataset Generator")
    print("=======================================================\n")
    print(f" [*] Target Output Directory: {args.output_dir}")
    print(f" [*] Temporal Range:          {args.days} days ({args.frequency} min intervals)")
    print(f" [*] Random Seed:              {args.seed}")
    print(" [*] Generating multi-source data across Indian states & cities ...\n")

    ingestor = IndiaScaleDataIngestor(seed=args.seed)
    aq_df, w_df, st_df = ingestor.generate_datasets(
        days=args.days,
        frequency_minutes=args.frequency,
        inject_quality_issues=not args.no_quality_issues
    )

    paths = ingestor.save_datasets(aq_df, w_df, st_df, output_dir=args.output_dir)

    print("-------------------------------------------------------")
    print(" [SUCCESS] India-Scale Datasets Generated Successfully!")
    print(f"           Stations Metadata:    {len(st_df)} stations across {st_df['state'].nunique()} states / {st_df['city'].nunique()} cities")
    print(f"           Air Quality Records:  {len(aq_df):,} rows -> {paths['air_quality']}")
    print(f"           Weather Records:      {len(w_df):,} rows -> {paths['weather']}")
    print(f"           Station Metadata:     {len(st_df):,} rows -> {paths['stations']}")
    print(f"           Dataset Provenance:   {paths['provenance']}")
    print("-------------------------------------------------------\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

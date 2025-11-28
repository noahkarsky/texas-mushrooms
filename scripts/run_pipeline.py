#!/usr/bin/env python
"""
Run the full Texas Mushrooms data processing pipeline.

This script orchestrates all preprocessing steps:
1. Clean raw scraped data (photos, species parsing)
2. Build modeling dataset with weather features

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --no-filter  # Use all years, not just 2018-2024
"""

from __future__ import annotations

import logging
from pathlib import Path

# Set up logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    import argparse

    from texas_mushrooms.pipeline.processing import run_full_pipeline

    # Resolve paths relative to repo root
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Run the full Texas Mushrooms data processing pipeline."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=repo_root / "data" / "raw",
        help="Directory containing raw days.csv and photos.csv",
    )
    parser.add_argument(
        "--weather-csv",
        type=Path,
        default=repo_root / "data" / "external" / "daily_weather.csv",
        help="Path to daily_weather.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "processed",
        help="Output directory for processed datasets",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Don't filter to 2018-2024 (use all years)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TEXAS MUSHROOMS DATA PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Raw data: {args.raw_dir}")
    logger.info(f"Weather:  {args.weather_csv}")
    logger.info(f"Output:   {args.output_dir}")
    logger.info(f"Filter years: {not args.no_filter}")
    logger.info("=" * 60)

    # Check inputs exist
    if not args.raw_dir.exists():
        logger.error(f"Raw data directory not found: {args.raw_dir}")
        logger.error("Run: python -m texas_mushrooms.cli crawl")
        return

    if not args.weather_csv.exists():
        logger.error(f"Weather data not found: {args.weather_csv}")
        logger.error("Run: python -m texas_mushrooms.pipeline.weather")
        return

    # Run pipeline
    results = run_full_pipeline(
        raw_dir=args.raw_dir,
        weather_csv=args.weather_csv,
        output_dir=args.output_dir,
        filter_years=not args.no_filter,
    )

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE - Output files:")
    logger.info("=" * 60)
    for name, df in results.items():
        logger.info(f"  {name}: {len(df)} rows")

    logger.info("")
    logger.info("Next steps:")
    logger.info("  - Run EDA notebook: notebooks/EDA.ipynb")
    logger.info("  - Run spatial analysis: python scripts/run_spatial_analysis.py")


if __name__ == "__main__":
    main()

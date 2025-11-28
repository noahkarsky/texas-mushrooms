#!/usr/bin/env python
"""
Prepare datasets for Texas Mushrooms.

This script orchestrates all preprocessing steps:
1. Clean raw scraped data (photos, species parsing)
2. Apply mushroom taxonomy filter (exclude crusts, slime molds, shelf fungi, lichens)
3. Build modeling dataset with weather features

Usage:
    python scripts/prepare_datasets.py
    python scripts/prepare_datasets.py --no-filter-years   # Use all years, not just 2018-2024
    python scripts/prepare_datasets.py --no-filter-species # Include all species (no taxonomy filter)
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
        description="Prepare datasets for Texas Mushrooms."
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
        help="Deprecated: use --no-filter-years instead",
    )
    parser.add_argument(
        "--no-filter-years",
        action="store_true",
        help="Don't filter to 2018-2024 (use all years)",
    )
    parser.add_argument(
        "--no-filter-species",
        action="store_true",
        help="Don't filter by mushroom taxonomy (include crusts, slime molds, etc.)",
    )

    args = parser.parse_args()
    
    # Handle deprecated --no-filter flag
    filter_years = not (args.no_filter or args.no_filter_years)
    filter_species = not args.no_filter_species

    logger.info("=" * 60)
    logger.info("TEXAS MUSHROOMS DATA PREPARATION")
    logger.info("=" * 60)
    logger.info(f"Raw data: {args.raw_dir}")
    logger.info(f"Weather:  {args.weather_csv}")
    logger.info(f"Output:   {args.output_dir}")
    logger.info(f"Filter years (2018-2024): {filter_years}")
    logger.info(f"Filter species (taxonomy): {filter_species}")
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
        filter_years=filter_years,
        filter_species=filter_species,
    )

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("PREPARATION COMPLETE - Output files:")
    logger.info("=" * 60)
    for name, df in results.items():
        logger.info(f"  {name}: {len(df)} rows")

    logger.info("")
    logger.info("Next steps:")
    logger.info("  - Run EDA notebook: notebooks/EDA.ipynb")
    logger.info("  - Run spatial analysis: python scripts/run_spatial_analysis.py")


if __name__ == "__main__":
    main()

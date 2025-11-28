"""
Data processing pipeline for Texas Mushrooms.

This module provides functions to:
1. Load and clean raw scraped data (days.csv, photos.csv)
2. Parse species lists and extract metadata
3. Filter observations by mushroom taxonomy (exclude crusts, slime molds, etc.)
4. Build modeling datasets with weather features
5. Export processed artifacts

Usage:
    python -m texas_mushrooms.pipeline.processing
"""

from __future__ import annotations

import ast
import logging
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from texas_mushrooms.config.filter_config import MushroomFilter, SpatialFilter

logger = logging.getLogger(__name__)

# Data coverage filter - only use years with good coverage
START_YEAR = 2018
END_YEAR = 2024


# =============================================================================
# Parsing Helpers
# =============================================================================


def parse_list(value: Any) -> list[str]:
    """Parse a string representation of a list into an actual list."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or value.strip() == "":
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


def parse_photo_species(val: Any) -> list[str]:
    """Parse species from photo records, handling dict format from scraper."""
    lst = parse_list(val)
    # Handle list of dicts from new scraper format
    if lst and isinstance(lst[0], dict):
        return [
            str(d.get("latin_name"))
            for d in lst
            if isinstance(d, dict) and d.get("latin_name")
        ]
    return lst


def normalize_species_name(name: str) -> str:
    """Normalize species names by removing qualifiers like cf., aff., var."""
    if not isinstance(name, str):
        return ""
    s = name.strip()
    s = re.sub(r"\b(cf\.|aff\.|var\.|ssp\.|spp?\.)\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[()\[\]{}]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def list_to_str(xs: Iterable[str]) -> str:
    """Convert a list of strings to semicolon-separated string."""
    return ";".join([x for x in xs if isinstance(x, str)])


# =============================================================================
# Data Loading
# =============================================================================


def load_days(days_csv: Path, filter_years: bool = True) -> pd.DataFrame:
    """
    Load days.csv and return a DataFrame with parsed date column.

    Args:
        days_csv: Path to days.csv
        filter_years: If True, filter to START_YEAR-END_YEAR range
    """
    df = pd.read_csv(days_csv)
    df["date"] = pd.to_datetime(df["date"])

    if filter_years:
        df = df[
            (df["date"].dt.year >= START_YEAR) & (df["date"].dt.year <= END_YEAR)
        ].copy()
        logger.info(f"Filtered days to {START_YEAR}-{END_YEAR}: {len(df)} rows")

    df = df.sort_values("date")
    return df


def load_photos(photos_csv: Path, filter_years: bool = True) -> pd.DataFrame:
    """
    Load photos.csv and return a DataFrame with parsed date column.

    Args:
        photos_csv: Path to photos.csv
        filter_years: If True, filter to START_YEAR-END_YEAR range
    """
    df = pd.read_csv(photos_csv)
    df["date"] = pd.to_datetime(df["date"])

    if filter_years:
        df = df[
            (df["date"].dt.year >= START_YEAR) & (df["date"].dt.year <= END_YEAR)
        ].copy()
        logger.info(f"Filtered photos to {START_YEAR}-{END_YEAR}: {len(df)} rows")

    df = df.sort_values("date")
    return df


def load_daily_weather(weather_csv: Path) -> pd.DataFrame:
    """Load daily_weather.csv and parse the date column."""
    df = pd.read_csv(weather_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


# =============================================================================
# Species Processing
# =============================================================================


def process_species_columns(
    days_df: pd.DataFrame, photos_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse species list columns and derive convenience columns.

    Returns updated copies of days_df and photos_df.
    """
    days_df = days_df.copy()
    photos_df = photos_df.copy()

    # Parse identified_species in days
    if "identified_species" in days_df.columns:
        days_df["identified_species_list"] = days_df["identified_species"].apply(
            parse_list
        )
        days_df["identified_species_count"] = days_df["identified_species_list"].apply(
            lambda xs: len(xs) if isinstance(xs, list) else 0
        )

    # Parse species in photos
    if "species" in photos_df.columns:
        photos_df["species_list"] = photos_df["species"].apply(parse_photo_species)
        photos_df["first_species"] = photos_df["species_list"].apply(
            lambda xs: xs[0] if isinstance(xs, list) and xs else "Unidentified"
        )

    return days_df, photos_df


def build_species_frequency(
    days_df: pd.DataFrame, photos_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build species frequency table from photo-level or day-level data.

    Returns DataFrame with columns: species, occurrences, source
    """
    counter: Counter[str] = Counter()
    source = "photo-level"

    if "species_list" in photos_df.columns:
        for xs in photos_df["species_list"]:
            counter.update(xs or [])
    elif "identified_species_list" in days_df.columns:
        for xs in days_df["identified_species_list"]:
            counter.update(xs or [])
        source = "day-level"

    species_freq_df = pd.DataFrame(
        [(sp, n) for sp, n in counter.items()], columns=["species", "occurrences"]
    ).sort_values("occurrences", ascending=False)
    species_freq_df["source"] = source

    return species_freq_df


# =============================================================================
# Taxonomy Filtering
# =============================================================================


def filter_photos_by_taxonomy(
    photos_df: pd.DataFrame,
    mushroom_filter: MushroomFilter,
    species_col: str = "first_species",
    caption_col: str | None = "full_caption",
) -> pd.DataFrame:
    """
    Filter photos DataFrame based on mushroom taxonomy configuration.
    
    Excludes observations of crusts, slime molds, shelf fungi (like Trametes),
    and lichens while keeping interesting stalked mushrooms like boletes.
    
    Args:
        photos_df: DataFrame with photo records.
        mushroom_filter: MushroomFilter configuration loaded from YAML.
        species_col: Column containing the primary species name.
        caption_col: Column containing caption text (optional, set to None to skip).
        
    Returns:
        Filtered DataFrame with only "cool" mushrooms.
    """
    if species_col not in photos_df.columns:
        logger.warning(f"Species column '{species_col}' not found, returning all rows")
        return photos_df.copy()
    
    def check_row(row: pd.Series) -> bool:
        species = row.get(species_col, "")
        caption = row.get(caption_col, "") if caption_col else ""
        return mushroom_filter.should_include(
            species_name=species if isinstance(species, str) else "",
            caption=caption if isinstance(caption, str) else "",
        )
    
    mask = photos_df.apply(check_row, axis=1)
    filtered_df = photos_df[mask].copy()
    
    n_removed = len(photos_df) - len(filtered_df)
    logger.info(
        f"Taxonomy filter: {len(filtered_df)} photos retained, "
        f"{n_removed} excluded ({n_removed / len(photos_df):.1%})"
    )
    
    return filtered_df


def get_exclusion_stats(
    photos_df: pd.DataFrame,
    mushroom_filter: MushroomFilter,
    species_col: str = "first_species",
    caption_col: str | None = "full_caption",
) -> pd.DataFrame:
    """
    Get statistics on which species/genera are being excluded and why.
    
    Args:
        photos_df: DataFrame with photo records.
        mushroom_filter: MushroomFilter configuration.
        species_col: Column containing the primary species name.
        caption_col: Column containing caption text (optional).
        
    Returns:
        DataFrame with columns: species, count, exclusion_reason
    """
    if species_col not in photos_df.columns:
        return pd.DataFrame(columns=["species", "count", "exclusion_reason"])
    
    exclusion_data: list[dict[str, Any]] = []
    
    for species, group in photos_df.groupby(species_col):
        if not isinstance(species, str):
            continue
        caption = ""
        if caption_col and caption_col in group.columns:
            # Use first non-empty caption as sample
            captions = group[caption_col].dropna()
            caption = captions.iloc[0] if len(captions) > 0 else ""
        
        reason = mushroom_filter.get_exclusion_reason(species, caption)
        if reason:
            exclusion_data.append({
                "species": species,
                "count": len(group),
                "exclusion_reason": reason,
            })
    
    return pd.DataFrame(exclusion_data).sort_values("count", ascending=False)


# =============================================================================
# Photo Exports
# =============================================================================


def build_photos_cleaned(photos_df: pd.DataFrame) -> pd.DataFrame:
    """Build cleaned photos DataFrame with parsed species."""
    df = photos_df.copy()

    if "species_list" in df.columns:
        df["species_list_str"] = df["species_list"].apply(list_to_str)
        df["first_species"] = df["species_list"].apply(
            lambda xs: xs[0] if isinstance(xs, list) and xs else "Unidentified"
        )

    return df


def build_photo_geospatial(photos_df: pd.DataFrame) -> pd.DataFrame:
    """Build geospatial export with lat/lon and species labels."""
    df = photos_df.dropna(subset=["latitude", "longitude"]).copy()

    if "first_species" not in df.columns:
        df["first_species"] = "Unidentified"

    df["label_species"] = df["first_species"]

    cols = ["date", "page_url", "photo_url", "latitude", "longitude", "label_species"]
    available_cols = [c for c in cols if c in df.columns]

    return df[available_cols]


# =============================================================================
# Weather Features (for modeling dataset)
# =============================================================================


def build_calendar_from_weather(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Use the weather date range as the global calendar for modeling."""
    dates = weather_df["date"].sort_values().unique()
    return pd.DataFrame({"date": dates})


def attach_mushroom_presence(
    calendar: pd.DataFrame,
    days_df: pd.DataFrame,
    photos_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Add has_mushroom, photo_count, and species_count columns to the calendar.

    - has_mushroom: 1 if date appears in days_df, else 0
    - photo_count: number of photos that day, or 0 if none
    - species_count: number of unique species identified that day
    """
    mushroom_dates = days_df[["date"]].drop_duplicates()
    mushroom_dates["has_mushroom"] = 1

    out = calendar.merge(mushroom_dates, on="date", how="left")
    out["has_mushroom"] = out["has_mushroom"].fillna(0).astype(int)

    if "photo_count" in days_df.columns:
        counts = days_df[["date", "photo_count"]]
        out = out.merge(counts, on="date", how="left")
        out["photo_count"] = out["photo_count"].fillna(0).astype(int)
    else:
        out["photo_count"] = 0

    # Calculate unique species count per day from photos
    if photos_df is not None and "species_list" in photos_df.columns:
        species_by_day = (
            photos_df.explode("species_list")
            .dropna(subset=["species_list"])
            .groupby("date")["species_list"]
            .nunique()
            .reset_index(name="species_count")
        )
        out = out.merge(species_by_day, on="date", how="left")
        out["species_count"] = out["species_count"].fillna(0).astype(int)
    else:
        out["species_count"] = 0

    return out


def add_weather_features(
    daily_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join weather fields and engineer lagged rain and seasonal features."""
    df = daily_df.merge(weather_df, on="date", how="left")
    df = df.sort_values("date")

    # Rain features (rolling windows)
    if "rain_sum" in df.columns:
        df["rain_1"] = df["rain_sum"]
        df["rain_3"] = df["rain_sum"].rolling(window=3, min_periods=1).sum()
        df["rain_7"] = df["rain_sum"].rolling(window=7, min_periods=1).sum()

    # Temperature features
    if "temperature_max" in df.columns and "temperature_min" in df.columns:
        if "temperature_mean" not in df.columns:
            df["temperature_mean"] = (
                df["temperature_max"] + df["temperature_min"]
            ) / 2.0
        df["temp_range"] = df["temperature_max"] - df["temperature_min"]

    # Seasonality features
    day_of_year = df["date"].dt.dayofyear
    df["seasonal_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    df["seasonal_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)

    return df


# =============================================================================
# Main Pipeline Functions
# =============================================================================


def run_preprocessing(
    raw_dir: Path,
    output_dir: Path,
    filter_years: bool = True,
    filter_species: bool = True,
    spatial_filter: SpatialFilter | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run the full preprocessing pipeline on raw scraped data.

    Args:
        raw_dir: Directory containing raw days.csv and photos.csv.
        output_dir: Output directory for processed datasets.
        filter_years: If True, filter to START_YEAR-END_YEAR range.
        filter_species: If True, apply mushroom taxonomy filter to exclude
            crusts, slime molds, shelf fungi, and lichens.
        spatial_filter: Optional spatial bounding box filter.

    Outputs:
        - photos_cleaned.csv: All photos with parsed species
        - photo_geospatial.csv: Photos with lat/lon for mapping
        - species_frequency.csv: Species occurrence counts

    Returns dict of DataFrames for further use.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    # Load raw data
    logger.info("Loading raw data...")
    days_df = load_days(raw_dir / "days.csv", filter_years=filter_years)
    photos_df = load_photos(raw_dir / "photos.csv", filter_years=filter_years)
    logger.info(f"Loaded {len(days_df)} days, {len(photos_df)} photos")

    # Process species columns
    logger.info("Processing species columns...")
    days_df, photos_df = process_species_columns(days_df, photos_df)

    # Apply spatial filter if requested
    if spatial_filter:
        from texas_mushrooms.pipeline.filters import filter_by_bbox
        
        logger.info(f"Applying spatial filter: {spatial_filter}")
        photos_df = filter_by_bbox(photos_df, spatial_filter)

    # Apply mushroom taxonomy filter if requested
    if filter_species:
        try:
            from texas_mushrooms.config.filter_config import MushroomFilter
            mushroom_filter = MushroomFilter.from_yaml()
            logger.info(f"Applying mushroom taxonomy filter: {mushroom_filter.summary()}")
            
            # Log what will be excluded before filtering
            exclusion_stats = get_exclusion_stats(photos_df, mushroom_filter)
            if not exclusion_stats.empty:
                logger.info(f"Top excluded genera/species:")
                for _, row in exclusion_stats.head(10).iterrows():
                    logger.info(f"  {row['species']}: {row['count']} photos ({row['exclusion_reason']})")
            
            photos_df = filter_photos_by_taxonomy(photos_df, mushroom_filter)
        except FileNotFoundError as e:
            logger.warning(f"Could not load mushroom filter config: {e}")
            logger.warning("Proceeding without species filter")
    else:
        logger.info("Skipping mushroom taxonomy filter (--no-filter-species)")

    # Build and export photos_cleaned
    photos_cleaned = build_photos_cleaned(photos_df)
    out_path = output_dir / "photos_cleaned.csv"
    photos_cleaned.to_csv(out_path, index=False)
    logger.info(f"✓ Exported photos_cleaned.csv ({len(photos_cleaned)} rows)")

    # Build and export photo_geospatial
    photo_geo = build_photo_geospatial(photos_cleaned)
    out_path = output_dir / "photo_geospatial.csv"
    photo_geo.to_csv(out_path, index=False)
    logger.info(f"✓ Exported photo_geospatial.csv ({len(photo_geo)} rows)")

    # Build and export species_frequency
    species_freq = build_species_frequency(days_df, photos_df)
    out_path = output_dir / "species_frequency.csv"
    species_freq.to_csv(out_path, index=False)
    logger.info(f"✓ Exported species_frequency.csv ({len(species_freq)} rows)")

    logger.info("Preprocessing complete!")

    return {
        "days": days_df,
        "photos": photos_df,
        "photos_cleaned": photos_cleaned,
        "photo_geospatial": photo_geo,
        "species_frequency": species_freq,
    }


def build_modeling_dataset(
    days_csv: Path,
    weather_csv: Path,
    output_dir: Path,
    filter_years: bool = True,
    photos_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the daily modeling dataset with weather features.

    Outputs:
        - mushroom_daily.csv: Daily dataset for modeling

    Returns the DataFrame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BUILDING MODELING DATASET")
    logger.info("=" * 60)

    days_df = load_days(days_csv, filter_years=filter_years)
    weather_df = load_daily_weather(weather_csv)

    # Load photos if not provided (needed for species_count)
    if photos_df is None:
        photos_csv = days_csv.parent / "photos.csv"
        if photos_csv.exists():
            photos_df = load_photos(photos_csv, filter_years=filter_years)
            _, photos_df = process_species_columns(days_df, photos_df)

    # Filter weather to same date range
    if filter_years:
        weather_df = weather_df[
            (weather_df["date"].dt.year >= START_YEAR)
            & (weather_df["date"].dt.year <= END_YEAR)
        ].copy()

    calendar = build_calendar_from_weather(weather_df)
    base = attach_mushroom_presence(calendar, days_df, photos_df)
    dataset = add_weather_features(base, weather_df)
    dataset = dataset.sort_values("date")

    # Save outputs
    dataset.to_csv(output_dir / "mushroom_daily.csv", index=False)

    logger.info(f"✓ Saved mushroom_daily.csv ({len(dataset)} rows)")
    logger.info(
        f"  Date range: {dataset['date'].min().date()} to {dataset['date'].max().date()}"
    )
    logger.info(
        f"  Mushroom days: {dataset['has_mushroom'].sum()} "
        f"({dataset['has_mushroom'].mean():.1%})"
    )
    logger.info(
        f"  Total species observations: {dataset['species_count'].sum()}"
    )

    return dataset


def run_full_pipeline(
    raw_dir: Path,
    weather_csv: Path,
    output_dir: Path,
    filter_years: bool = True,
    filter_species: bool = True,
    spatial_filter: SpatialFilter | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run the complete processing pipeline:
    1. Preprocess raw data (photos, species)
    2. Build modeling dataset (weather features)

    Args:
        raw_dir: Directory containing raw days.csv and photos.csv.
        weather_csv: Path to daily_weather.csv.
        output_dir: Output directory for processed datasets.
        filter_years: If True, filter to START_YEAR-END_YEAR range.
        filter_species: If True, apply mushroom taxonomy filter.
        spatial_filter: Optional spatial bounding box filter.

    Returns dict of all output DataFrames.
    """
    results = run_preprocessing(
        raw_dir, 
        output_dir, 
        filter_years=filter_years, 
        filter_species=filter_species,
        spatial_filter=spatial_filter,
    )

    modeling_df = build_modeling_dataset(
        raw_dir / "days.csv",
        weather_csv,
        output_dir,
        filter_years=filter_years,
        photos_df=results.get("photos"),
    )
    results["mushroom_daily"] = modeling_df

    logger.info("=" * 60)
    logger.info("FULL PIPELINE COMPLETE")
    logger.info("=" * 60)

    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run the Texas Mushrooms data processing pipeline."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw days.csv and photos.csv",
    )
    parser.add_argument(
        "--weather-csv",
        type=Path,
        default=Path("data/external/daily_weather.csv"),
        help="Path to daily_weather.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed datasets",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help=f"Don't filter to {START_YEAR}-{END_YEAR} (use all years)",
    )
    parser.add_argument(
        "--no-filter-species",
        action="store_true",
        help="Don't filter by mushroom taxonomy (include all species)",
    )

    args = parser.parse_args()

    run_full_pipeline(
        args.raw_dir,
        args.weather_csv,
        args.output_dir,
        filter_years=not args.no_filter,
        filter_species=not args.no_filter_species,
        spatial_filter=None,
    )

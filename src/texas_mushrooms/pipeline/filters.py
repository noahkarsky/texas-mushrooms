"""Data filtering utilities for mushroom observations.

Functions for filtering by year and applying taxonomy filters.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import pandas as pd

from texas_mushrooms.config.filter_config import MushroomFilter

if TYPE_CHECKING:
    from texas_mushrooms.config.filter_config import SpatialFilter

logger = logging.getLogger(__name__)


def filter_by_year(
    df: pd.DataFrame,
    date_col: str = "date",
    start_year: int = 2018,
    end_year: int = 2024,
) -> pd.DataFrame:
    """Filter DataFrame to specified year range.
    
    Args:
        df: Input DataFrame with date column.
        date_col: Name of the date column.
        start_year: Starting year (inclusive).
        end_year: Ending year (inclusive).
        
    Returns:
        Filtered DataFrame containing only rows within year range.
    """
    df = df.copy()
    df["_date_dt"] = pd.to_datetime(df[date_col])
    initial_count = len(df)
    df = df[
        (df["_date_dt"].dt.year >= start_year) & (df["_date_dt"].dt.year <= end_year)
    ]
    df = df.drop(columns=["_date_dt"])
    excluded = initial_count - len(df)
    logger.info(f"Year filter ({start_year}-{end_year}): {initial_count} -> {len(df)} rows ({excluded} excluded)")
    return df


def apply_mushroom_filter(
    df: pd.DataFrame,
    mushroom_filter: Optional[MushroomFilter] = None,
) -> pd.DataFrame:
    """Apply taxonomy filter to keep only 'cool' stalked mushrooms.
    
    Excludes: crusts, slime molds, shelf fungi, and lichens.
    Uses genus-level blacklist with explicit species overrides.
    
    Args:
        df: DataFrame with 'species' and optional 'caption' columns.
        mushroom_filter: MushroomFilter instance. If None, loads from YAML config.
        
    Returns:
        Filtered DataFrame containing only included species.
    """
    if "species" not in df.columns:
        logger.warning("'species' column not found, skipping mushroom filter")
        return df
    
    if mushroom_filter is None:
        try:
            mushroom_filter = MushroomFilter.from_yaml()
            logger.info(f"Loaded filter config: {mushroom_filter.summary()}")
        except FileNotFoundError as e:
            logger.error(f"Failed to load filter config: {e}")
            return df
    
    # Determine caption column name (could be 'caption' or 'description')
    caption_col = None
    if "caption" in df.columns:
        caption_col = "caption"
    elif "description" in df.columns:
        caption_col = "description"
    
    # Apply filter
    initial_count = len(df)
    df_filtered = df.copy()
    
    # Create filter mask by checking each row
    def should_include_row(row: pd.Series) -> bool:
        caption = row.get(caption_col, "") if caption_col else ""
        caption = str(caption) if caption else ""
        return mushroom_filter.should_include(row["species"], caption)
    
    mask = df_filtered.apply(should_include_row, axis=1)
    df_filtered = df_filtered[mask]
    excluded_count = initial_count - len(df_filtered)
    
    logger.info(f"Mushroom filter: {initial_count} -> {len(df_filtered)} rows ({excluded_count} excluded)")
    return df_filtered


def filter_by_bbox(
    df: pd.DataFrame,
    bbox: "SpatialFilter",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    """Filter DataFrame to points within a spatial bounding box.
    
    Args:
        df: Input DataFrame with latitude and longitude columns.
        bbox: SpatialFilter instance defining the bounding box.
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.
        
    Returns:
        Filtered DataFrame containing only rows within the bounding box.
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        logger.warning(f"Missing coordinates columns ({lat_col}, {lon_col}), skipping spatial filter")
        return df
        
    initial_count = len(df)
    
    # Filter mask
    mask = (
        (df[lat_col] >= bbox.min_lat) & 
        (df[lat_col] <= bbox.max_lat) & 
        (df[lon_col] >= bbox.min_lon) & 
        (df[lon_col] <= bbox.max_lon)
    )
    
    df_filtered = df[mask].copy()
    excluded = initial_count - len(df_filtered)
    
    logger.info(f"Spatial filter ({bbox}): {initial_count} -> {len(df_filtered)} rows ({excluded} excluded)")
    return df_filtered

"""
Spatial processing module for Texas Mushrooms project.
Handles H3 indexing, grid generation, and spatial data enrichment.
"""
from __future__ import annotations

import h3
import pandas as pd
from typing import List, Tuple


def add_h3_indices(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    resolution: int = 7,
) -> pd.DataFrame:
    """
    Add H3 hexagonal indices to a DataFrame based on latitude and longitude.

    Args:
        df: DataFrame containing latitude and longitude columns.
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.
        resolution: H3 resolution (0-15). 7 is ~1.2km edge, 8 is ~0.46km.

    Returns:
        DataFrame with a new 'h3_index' column.
    """
    # Ensure we don't modify the original
    df = df.copy()

    # Filter out invalid coordinates
    valid_coords = df[df[lat_col].notna() & df[lon_col].notna()]

    if valid_coords.empty:
        df["h3_index"] = None
        return df

    # Apply H3 indexing
    # Note: h3-py v4 changed geo_to_h3 to latlng_to_cell
    lat_vals = valid_coords[lat_col].to_numpy()
    lon_vals = valid_coords[lon_col].to_numpy()

    h3_list = []
    # Prefer v4 API if available; fall back to older API
    if hasattr(h3, "latlng_to_cell"):
        latlng_to_cell = h3.latlng_to_cell
        for lat, lon in zip(lat_vals, lon_vals):
            if pd.isna(lat) or pd.isna(lon):
                h3_list.append(None)
            else:
                h3_list.append(latlng_to_cell(float(lat), float(lon), resolution))
    else:
        for lat, lon in zip(lat_vals, lon_vals):
            if pd.isna(lat) or pd.isna(lon):
                h3_list.append(None)
            else:
                h3_list.append(h3.geo_to_h3(float(lat), float(lon), resolution))

    # Build a series aligned with valid_coords and then reindex to original DF
    h3_series = pd.Series(h3_list, index=valid_coords.index)
    df["h3_index"] = h3_series.reindex(df.index)

    return df


def get_h3_centroid(h3_index: str) -> Tuple[float, float]:
    """
    Get the (lat, lon) centroid of an H3 index.
    """
    try:
        return h3.cell_to_latlng(h3_index)  # type: ignore
    except AttributeError:
        return h3.h3_to_geo(h3_index)  # type: ignore


def create_spatiotemporal_grid(
    h3_indices: List[str], start_date: str, end_date: str, freq: str = "D"
) -> pd.DataFrame:
    """
    Create a complete grid of (H3 Index x Date) to handle zero-inflation.

    Args:
        h3_indices: List of unique H3 indices to include.
        start_date: Start date string.
        end_date: End date string.
        freq: Frequency string (default 'D' for daily).

    Returns:
        DataFrame with 'h3_index' and 'date' columns, cartesian product.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    # Create MultiIndex from product
    index = pd.MultiIndex.from_product([h3_indices, dates], names=["h3_index", "date"])

    return pd.DataFrame(index=index).reset_index()

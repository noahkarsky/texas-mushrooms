"""
Weather data fetching and processing module for Texas Mushrooms project.
Retrieves historical weather data from Open-Meteo API.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests


def fetch_daily_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str = "America/Chicago",
) -> pd.DataFrame:
    """
    Call the Open-Meteo historical weather API and return a daily weather DataFrame.

    Args:
        latitude: Latitude of the location.
        longitude: Longitude of the location.
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        timezone: Timezone for daily aggregation.

    Returns:
        pd.DataFrame: Daily weather data with standardized columns.

    Raises:
        ValueError: If the API request fails or returns invalid data.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"

    daily_vars = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "wind_speed_10m_max",
        "wind_speed_10m_mean",
        "relative_humidity_2m_mean",
        "soil_temperature_0_to_7cm_mean",
        "soil_moisture_0_to_7cm_mean",
    ]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone,
        "daily": ",".join(daily_vars),
    }

    # Convert params to strings for requests
    params_str: dict[str, str] = {k: str(v) for k, v in params.items()}

    try:
        response = requests.get(url, params=params_str, timeout=30)
    except requests.RequestException as e:
        raise ValueError(f"Failed to connect to Open-Meteo API: {e}") from e

    if response.status_code != 200:
        raise ValueError(
            f"Open-Meteo API returned error {response.status_code}: {response.text}"
        )

    data = response.json()

    if "daily" not in data:
        raise ValueError("Response JSON is missing required 'daily' key")

    daily_data = data["daily"]

    # Validate all requested variables are present
    # Note: 'time' is always returned by the API for daily data
    required_keys = ["time"] + daily_vars
    missing_keys = [k for k in required_keys if k not in daily_data]
    if missing_keys:
        raise ValueError(f"Response JSON is missing daily variables: {missing_keys}")

    # Create DataFrame
    df = pd.DataFrame(daily_data)

    # Rename columns to match requirements
    column_mapping = {
        "time": "date",
        "temperature_2m_max": "temperature_max",
        "temperature_2m_min": "temperature_min",
        "temperature_2m_mean": "temperature_mean",
        "precipitation_sum": "precipitation_sum",
        "rain_sum": "rain_sum",
        "snowfall_sum": "snowfall_sum",
        "wind_speed_10m_max": "wind_speed_max",
        "wind_speed_10m_mean": "wind_speed_mean",
        "relative_humidity_2m_mean": "relative_humidity_mean",
        "soil_temperature_0_to_7cm_mean": "soil_temp_mean",
        "soil_moisture_0_to_7cm_mean": "soil_moisture_mean",
    }

    df = df.rename(columns=column_mapping)

    # Select only the columns we want (in case API returns extras)
    desired_columns = list(column_mapping.values())
    df = df[desired_columns]

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    return df


def infer_date_range_from_days_csv(days_csv: Path) -> tuple[str, str]:
    """
    Infer start and end dates from the mushroom days CSV.

    Args:
        days_csv: Path to the days CSV file.

    Returns:
        tuple[str, str]: (start_date, end_date) in YYYY-MM-DD format.
    """
    if not days_csv.exists():
        raise ValueError(f"Days CSV not found at: {days_csv}")

    try:
        df = pd.read_csv(days_csv, parse_dates=["date"])
    except Exception as e:
        raise ValueError(f"Failed to read days CSV: {e}") from e

    if "date" not in df.columns:
        raise ValueError("Days CSV is missing 'date' column")

    # Drop NA dates
    dates = df["date"].dropna()

    if dates.empty:
        raise ValueError("No valid dates found in days CSV")

    start_date = dates.min().strftime("%Y-%m-%d")
    end_date = dates.max().strftime("%Y-%m-%d")

    return start_date, end_date


def _ensure_directory(path: Path) -> None:
    """Ensure the directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def build_and_save_weather_dataset(
    days_csv: Path,
    latitude: float = 29.98444,
    longitude: float = -95.34139,
    timezone: str = "America/Chicago",
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Build the weather dataset and save it locally.

    Args:
        days_csv: Path to the mushroom days CSV.
        latitude: Latitude for weather data (default: KIAH).
        longitude: Longitude for weather data (default: KIAH).
        timezone: Timezone for weather data.
        output_dir: Directory to save output files. Defaults to data/weather.

    Returns:
        pd.DataFrame: The fetched weather data.
    """
    print(f"Inferring date range from {days_csv}...")
    start_date, end_date = infer_date_range_from_days_csv(days_csv)
    print(f"Date range: {start_date} to {end_date}")

    print(f"Fetching weather data for {latitude}, {longitude} ({timezone})...")
    weather_df = fetch_daily_weather(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        timezone=timezone,
    )

    # Optional: Join with existing enriched weather data if available
    # (Logic omitted as per "nice to have but not required" instruction,
    # but structure allows for it here)

    if output_dir is None:
        # Default to repo_root/data/external
        # Assuming this script is in src/texas_mushrooms/pipeline/weather.py
        # We can try to resolve relative to CWD or relative to file.
        # Given the prompt implies running from repo root, Path("data") / "external" is safe.
        output_dir = Path("data") / "external"

    _ensure_directory(output_dir)

    parquet_path = output_dir / "daily_weather.parquet"
    csv_path = output_dir / "daily_weather.csv"

    print(f"Saving to {output_dir}...")
    weather_df.to_parquet(parquet_path, index=False)
    weather_df.to_csv(csv_path, index=False)

    print("Summary:")
    print(f"  Rows: {len(weather_df)}")
    print(f"  Start: {weather_df['date'].min()}")
    print(f"  End: {weather_df['date'].max()}")
    print("\nHead:")
    print(weather_df.head())

    return weather_df


def fetch_elevation_batch(
    latitudes: list[float],
    longitudes: list[float],
) -> list[float | None]:
    """
    Fetch elevation for a batch of coordinates using Open-Meteo Elevation API.

    Args:
        latitudes: List of latitudes.
        longitudes: List of longitudes.

    Returns:
        List of elevations in meters.
    """
    url = "https://api.open-meteo.com/v1/elevation"

    # Protect against mismatched input lengths
    if len(latitudes) != len(longitudes):
        raise ValueError("latitudes and longitudes must have the same length")

    # Open-Meteo accepts comma-separated lists, but very large requests may fail or hit URL limits.
    # Chunk the coordinates into smaller batches for reliability.
    results: list[float | None] = []
    chunk_size = 500
    for i in range(0, len(latitudes), chunk_size):
        lats_chunk = latitudes[i : i + chunk_size]
        lons_chunk = longitudes[i : i + chunk_size]

        params = {
            "latitude": ",".join(map(str, lats_chunk)),
            "longitude": ",".join(map(str, lons_chunk)),
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            chunk_elev = data.get("elevation", [None] * len(lats_chunk))  # type: ignore
        except requests.RequestException as e:
            print(f"Warning: Failed to fetch elevation for chunk starting at {i}: {e}")
            chunk_elev = [None] * len(lats_chunk)

        results.extend(chunk_elev)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch and build daily weather dataset."
    )
    parser.add_argument(
        "--days-csv",
        type=Path,
        default=Path("data/raw/days.csv"),
        help="Path to the days CSV file (default: data/raw/days.csv)",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=29.98444,
        help="Latitude (default: 29.98444)",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=-95.34139,
        help="Longitude (default: -95.34139)",
    )
    parser.add_argument(
        "--timezone",
        type=str,
        default="America/Chicago",
        help="Timezone (default: America/Chicago)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/external"),
        help="Output directory (default: data/external)",
    )

    args = parser.parse_args()

    try:
        build_and_save_weather_dataset(
            days_csv=args.days_csv,
            latitude=args.lat,
            longitude=args.lon,
            timezone=args.timezone,
            output_dir=args.outdir,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

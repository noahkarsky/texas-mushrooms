from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_days(days_csv: Path) -> pd.DataFrame:
    """
    Load days.csv and return a DataFrame with a parsed date column.

    Expected columns include at least:
      - date (YYYY-MM-DD)
      - photo_count (int) or something similar
    """
    df = pd.read_csv(days_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


def load_daily_weather(weather_csv: Path) -> pd.DataFrame:
    """
    Load daily_weather csv and parse the date column.
    """
    df = pd.read_csv(weather_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


def build_calendar_from_weather(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Use the weather date range as the global calendar for modeling.
    """
    dates = weather_df["date"].sort_values().unique()
    calendar = pd.DataFrame({"date": dates})
    return calendar


def attach_mushroom_presence(
    calendar: pd.DataFrame,
    days_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add has_mushroom and photo_count columns to the calendar:

    - has_mushroom is 1 if the date appears in days_df, else 0
    - photo_count is the number of photos that day, or 0 if none
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

    return out


def add_weather_features(
    daily_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join weather fields and engineer lagged rain and seasonal features.
    """
    df = daily_df.merge(weather_df, on="date", how="left")
    df = df.sort_values("date")

    # Basic rain features
    df["rain_1"] = df["rain_sum"]
    df["rain_3"] = df["rain_sum"].rolling(window=3, min_periods=1).sum()
    df["rain_7"] = df["rain_sum"].rolling(window=7, min_periods=1).sum()

    # Temperature features
    if "temperature_mean" not in df.columns:
        df["temperature_mean"] = (df["temperature_max"] + df["temperature_min"]) / 2.0
    df["temp_range"] = df["temperature_max"] - df["temperature_min"]

    # Seasonality features
    day_of_year = df["date"].dt.dayofyear
    df["seasonal_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    df["seasonal_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)

    return df


def build_and_save_daily_dataset(
    days_csv: Path,
    weather_csv: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """
    End to end builder:

    - Load days and weather csvs
    - Create calendar from the weather date range
    - Mark mushroom presence and photo counts
    - Attach weather and engineered features
    - Save to parquet and csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    days_df = load_days(days_csv)
    weather_df = load_daily_weather(weather_csv)

    calendar = build_calendar_from_weather(weather_df)
    base = attach_mushroom_presence(calendar, days_df)
    dataset = add_weather_features(base, weather_df)

    dataset = dataset.sort_values("date")

    out_parquet = output_dir / "mushroom_daily.parquet"
    out_csv = output_dir / "mushroom_daily.csv"

    dataset.to_parquet(out_parquet, index=False)
    dataset.to_csv(out_csv, index=False)

    print(f"Saved daily dataset with {len(dataset)} rows")
    print(
        f"Date range: {dataset['date'].min().date()} to {dataset['date'].max().date()}"
    )
    print(
        f"Mushroom days: {dataset['has_mushroom'].sum()} "
        f"({dataset['has_mushroom'].mean():.3f} fraction)"
    )

    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build daily mushroom modeling dataset."
    )
    parser.add_argument(
        "--days-csv",
        type=Path,
        default=Path("data/raw/days.csv"),
        help="Path to days.csv",
    )
    parser.add_argument(
        "--weather-csv",
        type=Path,
        default=Path("data/external/daily_weather.csv"),
        help="Path to daily_weather.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for modeling dataset",
    )

    args = parser.parse_args()
    build_and_save_daily_dataset(args.days_csv, args.weather_csv, args.outdir)

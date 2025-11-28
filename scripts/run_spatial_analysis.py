import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import our new modules
from texas_mushrooms.pipeline import spatial, weather
from texas_mushrooms.modeling.bayesian import BayesianMushroomModel

# Data coverage filter
START_YEAR = 2018
END_YEAR = 2024


def filter_by_year(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Filter DataFrame to years with best coverage (2018-2024)."""
    df = df.copy()
    df["_date_dt"] = pd.to_datetime(df[date_col])
    df = df[(df["_date_dt"].dt.year >= START_YEAR) & (df["_date_dt"].dt.year <= END_YEAR)]
    df = df.drop(columns=["_date_dt"])
    return df


def main() -> None:
    # Set plot style
    sns.set_theme(style="whitegrid")

    # =========================================================================
    # Option A: Weather-focused model (daily counts with weather predictors)
    # =========================================================================
    print("=" * 60)
    print("## Weather-Enriched Bayesian Model")
    print("=" * 60)

    daily_path = Path(__file__).resolve().parent.parent / "data/processed/mushroom_daily.csv"
    if daily_path.exists():
        run_weather_model(daily_path)
    else:
        print(f"Weather-enriched data not found at {daily_path}")
        print("Run: python -m texas_mushrooms.pipeline.processing")

    # =========================================================================
    # Option B: Spatial model (elevation only, as before)
    # =========================================================================
    print("\n" + "=" * 60)
    print("## Spatial Model (Elevation)")
    print("=" * 60)
    run_spatial_model()


def run_weather_model(daily_path: Path) -> None:
    """
    Run a ZIP model with weather predictors on daily mushroom counts.
    """
    print("\n## 1. Load Weather-Enriched Daily Data")
    df = pd.read_csv(daily_path, parse_dates=["date"])
    
    # Filter to best coverage years
    df = filter_by_year(df, date_col="date")
    print(f"Filtered to {START_YEAR}-{END_YEAR}")
    print(f"Loaded {len(df)} rows spanning {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Columns: {list(df.columns)}")

    # Filter to rows with valid weather data
    weather_cols = [
        "rain_7",
        "soil_moisture_mean",
        "temperature_mean",
        "relative_humidity_mean",
        "seasonal_sin",
        "seasonal_cos",
    ]
    df_model = df.dropna(subset=weather_cols + ["photo_count"])
    print(f"Rows with complete weather data: {len(df_model)}")

    print("\n## 2. Standardize Predictors")
    # Standardize numeric predictors for better MCMC sampling
    predictors_to_std = [
        "rain_7",
        "soil_moisture_mean",
        "temperature_mean",
        "relative_humidity_mean",
    ]
    for col in predictors_to_std:
        std_col = f"{col}_std"
        df_model[std_col] = (df_model[col] - df_model[col].mean()) / df_model[col].std()

    # Seasonality features are already scaled [-1, 1], keep as-is
    predictor_cols = [f"{c}_std" for c in predictors_to_std] + ["seasonal_sin", "seasonal_cos"]

    print(f"Predictors: {predictor_cols}")
    print(df_model[predictor_cols].describe().round(2))

    print("\n## 3. Build and Sample ZIP Model")
    model = BayesianMushroomModel(df_model, target_col="photo_count")
    model.build_zip_model(predictors=predictor_cols)

    print("Sampling (this may take a minute)...")
    model.sample(draws=500, tune=500, chains=2, cores=1)

    print("\n## 4. Results")
    import arviz as az

    # Print numeric summary
    summary = az.summary(model.trace, var_names=["alpha", "betas", "psi"], round_to=3)
    print("\nPosterior Summary:")
    print(summary)

    # Map beta indices to predictor names
    print("\nBeta coefficient mapping:")
    for i, name in enumerate(predictor_cols):
        print(f"  betas[{i}] = {name}")

    # Plot trace
    print("\nPlotting trace...")
    model.plot_trace()
    plt.suptitle("Weather-Enriched ZIP Model", y=1.02)
    plt.tight_layout()
    plt.show()


def run_spatial_model() -> None:
    """
    Original spatial model using elevation (kept for comparison).
    """
    print("\n## 1. Load Data and Add H3 Indices")
    # Load data (assuming processed data exists, otherwise fallback to raw photos)
    repo_root = Path(__file__).resolve().parent.parent
    data_path = repo_root / "data/processed/photo_geospatial.csv"
    if not data_path.exists():
        print("Processed data not found. Loading raw photos...")
        data_path = repo_root / "data/raw/photos.csv"

    if not data_path.exists():
        print(f"Error: Could not find data at {data_path}")
        return

    df = pd.read_csv(data_path)

    # Ensure we have lat/lon
    if "latitude" not in df.columns:
        print(
            "Warning: Latitude/Longitude columns missing. Please run the processing pipeline first."
        )
        return

    # Filter to best coverage years
    if "date" in df.columns:
        df = filter_by_year(df, date_col="date")
        print(f"Filtered to {START_YEAR}-{END_YEAR}: {len(df)} rows")

    # Add H3 Indices
    df_h3 = spatial.add_h3_indices(df, resolution=7)
    print(f"Added H3 indices. Unique cells: {df_h3['h3_index'].nunique()}")
    print(df_h3.head())

    print("\n## 2. Enrich with Elevation")
    # Get unique H3 indices
    unique_h3 = df_h3["h3_index"].dropna().unique()

    # Get centroids
    centroids = [spatial.get_h3_centroid(h) for h in unique_h3]
    lats = [c[0] for c in centroids]
    lons = [c[1] for c in centroids]

    # Fetch elevation (batch)
    print(f"Fetching elevation for {len(unique_h3)} unique cells...")
    elevations = weather.fetch_elevation_batch(lats, lons)

    # Create a mapping
    h3_elevation = dict(zip(unique_h3, elevations))

    # Map back to dataframe
    df_h3["elevation"] = df_h3["h3_index"].map(h3_elevation)
    print(df_h3[["h3_index", "latitude", "longitude", "elevation"]].head())

    print("\n## 3. Prepare Data for Modeling")
    # Aggregate counts
    if "date" in df_h3.columns:
        daily_counts = (
            df_h3.groupby(["h3_index", "date"]).size().reset_index(name="count")
        )

        # Merge with elevation (static per cell)
        daily_counts["elevation"] = daily_counts["h3_index"].map(h3_elevation)

        # Fill missing elevations if any
        daily_counts = daily_counts.dropna(subset=["elevation"])

        print("Modeling Data Prepared:")
        print(daily_counts.head())
    else:
        print("Date column missing. Cannot aggregate for temporal modeling.")
        return

    print("\n## 4. Bayesian Modeling (PyMC)")
    if "date" in df_h3.columns and not daily_counts.empty:
        # Initialize model
        model = BayesianMushroomModel(daily_counts, target_col="count")

        # Build ZIP model using Elevation as a predictor
        # We standardize elevation for better sampling
        daily_counts["elevation_std"] = (
            daily_counts["elevation"] - daily_counts["elevation"].mean()
        ) / daily_counts["elevation"].std()

        print("Building model...")
        model.build_zip_model(predictors=["elevation_std"])

        # Sample
        print("Sampling...")
        # Reduced draws for demonstration speed
        # cores=1 is critical for Windows to prevent hangs
        model.sample(draws=500, tune=500, chains=2, cores=1)

        # Plot trace
        print("Plotting trace...")
        model.plot_trace()
        plt.suptitle("Spatial Model (Elevation Only)", y=1.02)
        plt.show()
    else:
        print("Skipping modeling due to missing data.")


if __name__ == "__main__":
    main()

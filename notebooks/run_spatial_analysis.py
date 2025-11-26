import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import our new modules
from texas_mushrooms.pipeline import spatial, weather
from texas_mushrooms.modeling.bayesian import BayesianMushroomModel


def main() -> None:
    # Set plot style
    sns.set_theme(style="whitegrid")

    print("## 1. Load Data and Add H3 Indices")
    # Load data (assuming processed data exists, otherwise fallback to raw photos)
    data_path = Path("data/processed/photo_geospatial.csv")
    if not data_path.exists():
        print("Processed data not found. Loading raw photos...")
        data_path = Path("data/raw/photos.csv")

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
        plt.show()
    else:
        print("Skipping modeling due to missing data.")


if __name__ == "__main__":
    main()

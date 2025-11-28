import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

# Import our new modules
from texas_mushrooms.pipeline import spatial, weather
from texas_mushrooms.pipeline.filters import apply_mushroom_filter, filter_by_year
from texas_mushrooms.modeling.bayesian import BayesianMushroomModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Data coverage filter
START_YEAR = 2018
END_YEAR = 2024

# Output directory
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data/outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Run spatial and weather analysis on mushroom observations."""
    logger.info("Starting spatial analysis pipeline")
    
    # Set plot style
    sns.set_theme(style="whitegrid")

    # =========================================================================
    # Option A: Weather-focused model (daily counts with weather predictors)
    # =========================================================================
    logger.info("="*60)
    logger.info("Running weather-enriched Bayesian model")
    logger.info("="*60)

    daily_path = Path(__file__).resolve().parent.parent / "data/processed/mushroom_daily.csv"
    if daily_path.exists():
        try:
            run_weather_model(daily_path)
        except Exception as e:
            logger.error(f"Weather model failed: {e}", exc_info=True)
    else:
        logger.error(f"Weather-enriched data not found at {daily_path}")
        logger.info("Run: python -m texas_mushrooms.pipeline.processing")

    # =========================================================================
    # Option B: Spatial model (elevation only, as before)
    # =========================================================================
    logger.info("="*60)
    logger.info("Running spatial model (elevation)")
    logger.info("="*60)
    
    try:
        run_spatial_model()
    except Exception as e:
        logger.error(f"Spatial model failed: {e}", exc_info=True)
    
    logger.info("Spatial analysis pipeline complete")


def run_weather_model(daily_path: Path) -> None:
    """Run a ZIP model with weather predictors on daily mushroom counts.
    
    Args:
        daily_path: Path to mushroom_daily.csv with weather-enriched data.
        
    Raises:
        FileNotFoundError: If data file doesn't exist.
        ValueError: If required columns are missing.
    """
    logger.info("Step 1: Load weather-enriched daily data")
    
    try:
        df = pd.read_csv(daily_path, parse_dates=["date"])
        logger.info(f"Loaded {len(df)} rows from {daily_path}")
    except FileNotFoundError:
        logger.error(f"Data file not found: {daily_path}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV: {e}")
        raise
    
    # Note: mushroom_daily.csv is aggregated by date, so no species column to filter
    # Mushroom filtering is applied only to raw photo-level data
    
    # Filter to best coverage years
    df = filter_by_year(df, date_col="date", start_year=START_YEAR, end_year=END_YEAR)
    logger.info(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    logger.debug(f"Columns: {list(df.columns)}")

    # Use species_count (unique species per day) instead of photo_count
    # This is a better measure of mushroom activity than raw photo count
    target_col = "species_count" if "species_count" in df.columns else "photo_count"
    logger.info(f"Using target variable: {target_col}")

    # Filter to rows with valid weather data
    weather_cols = [
        "rain_7",
        "soil_moisture_mean",
        "temperature_mean",
        "relative_humidity_mean",
        "seasonal_sin",
        "seasonal_cos",
    ]
    
    missing_cols = [c for c in weather_cols + [target_col] if c not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
    
    df_model = df.dropna(subset=weather_cols + [target_col])
    excluded_rows = len(df) - len(df_model)
    logger.info(f"Complete weather data: {len(df_model)} rows ({excluded_rows} excluded due to missing values)")

    logger.info("Step 2: Standardize predictors")
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
        logger.debug(f"Standardized {col} -> {std_col}")

    # Seasonality features are already scaled [-1, 1], keep as-is
    predictor_cols = [f"{c}_std" for c in predictors_to_std] + ["seasonal_sin", "seasonal_cos"]
    logger.info(f"Predictor columns: {predictor_cols}")

    logger.info("Step 3: Build and sample ZIP model")
    model = BayesianMushroomModel(df_model, target_col=target_col)
    model.build_zip_model(predictors=predictor_cols)
    logger.info("Model built")

    logger.info("Sampling from posterior (this may take a minute)...")
    model.sample(draws=500, tune=500, chains=2, cores=1)
    logger.info("Sampling complete")

    logger.info("Step 4: Extract and visualize results")

    # Print numeric summary
    summary = az.summary(model.trace, var_names=["alpha", "betas", "psi"], round_to=3)
    summary.to_csv(OUTPUT_DIR / "weather_model_summary.csv")
    logger.info(f"Model summary saved to {OUTPUT_DIR / 'weather_model_summary.csv'}")
    
    logger.info("Posterior Summary:")
    for line in str(summary).split("\n"):
        logger.info(line)

    # Map beta indices to predictor names
    logger.info("Beta coefficient mapping:")
    for i, name in enumerate(predictor_cols):
        logger.info(f"  betas[{i}] = {name}")

    # Create labeled trace plot
    logger.info("Plotting trace...")
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    var_labels = {
        "alpha": "α (Intercept)",
        "betas": "β (Weather Effects)", 
        "psi": "ψ (Zero-Inflation Prob)"
    }
    
    for i, var in enumerate(["alpha", "betas", "psi"]):
        # Posterior Plot (Left)
        ax_post = axes[i, 0]
        
        # Handle betas specially if multidimensional
        is_multidim_beta = False
        if var == "betas":
            data = model.trace.posterior[var].values
            if data.ndim == 3 and data.shape[2] > 1:
                is_multidim_beta = True
                # Flatten chains and draws
                flat_data = data.reshape(-1, data.shape[-1])
                for idx in range(flat_data.shape[1]):
                    label = predictor_cols[idx] if idx < len(predictor_cols) else f"beta_{idx}"
                    sns.kdeplot(flat_data[:, idx], ax=ax_post, label=label)
                ax_post.legend(fontsize='x-small')
        
        if not is_multidim_beta:
            az.plot_posterior(model.trace, var_names=[var], ax=ax_post, 
                             hdi_prob=0.94, point_estimate="mean")
            
        ax_post.set_title(f"{var_labels[var]} - Posterior", fontsize=11)
        ax_post.set_xlabel("Parameter Value")
        ax_post.set_ylabel("Density")
        
        # Trace Plot (Right) - Manual to avoid ArviZ shape errors
        ax_trace = axes[i, 1]
        data = model.trace.posterior[var].values
        
        if data.ndim == 2:
            for chain_idx in range(data.shape[0]):
                ax_trace.plot(data[chain_idx], alpha=0.5)
        elif data.ndim == 3:
            for dim_idx in range(data.shape[2]):
                for chain_idx in range(data.shape[0]):
                    ax_trace.plot(data[chain_idx, :, dim_idx], alpha=0.3)

        ax_trace.set_title(f"{var_labels[var]} - MCMC Trace", fontsize=11)
        ax_trace.set_xlabel("Sample")
        ax_trace.set_ylabel("Parameter Value")
    
    plt.suptitle("Weather-Enriched ZIP Model\n(Mushroom Count ~ Rain + Soil Moisture + Temp + Humidity + Seasonality)", 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    fig.savefig(OUTPUT_DIR / "weather_model_trace.png", dpi=150, bbox_inches="tight")
    logger.info(f"Trace plot saved to {OUTPUT_DIR / 'weather_model_trace.png'}")
    
    # Print interpretation
    logger.info("Beta coefficient interpretation:")
    for i, name in enumerate(predictor_cols):
        logger.info(f"  betas[{i}] = {name}")


def run_spatial_model() -> None:
    """Run spatial model using elevation as predictor.
    
    Loads photo data, adds H3 indices, enriches with elevation data,
    and builds a Bayesian ZIP model.
    
    Returns:
        None (displays plots via matplotlib).
    """
    logger.info("Step 1: Load data and add H3 indices")
    # Load data (assuming processed data exists, otherwise fallback to raw photos)
    data_path = REPO_ROOT / "data/processed/photo_geospatial.csv"
    
    if not data_path.exists():
        logger.info("Processed data not found. Trying raw photos...")
        data_path = REPO_ROOT / "data/raw/photos.csv"

    if not data_path.exists():
        logger.error(f"Could not find data at {data_path}")
        return

    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} rows from {data_path.name}")
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV: {e}")
        return

    # Ensure we have lat/lon
    if "latitude" not in df.columns or "longitude" not in df.columns:
        logger.error("Latitude/Longitude columns missing. Run the processing pipeline first.")
        return

    # Apply mushroom taxonomy filter (only for photo-level data)
    try:
        df = apply_mushroom_filter(df)
    except FileNotFoundError as e:
        logger.error(f"Failed to load mushroom filter config: {e}")
        logger.warning("Proceeding without mushroom filter")

    # Filter to best coverage years
    if "date" in df.columns:
        df = filter_by_year(df, date_col="date", start_year=START_YEAR, end_year=END_YEAR)
    else:
        logger.warning("Date column not found; skipping year filter")

    # Add H3 Indices
    try:
        df_h3 = spatial.add_h3_indices(df, resolution=7)
        logger.info(f"Added H3 indices: {df_h3['h3_index'].nunique()} unique cells")
    except Exception as e:
        logger.error(f"Failed to add H3 indices: {e}")
        return

    logger.info("Step 2: Enrich with elevation data")
    # Get unique H3 indices
    unique_h3 = df_h3["h3_index"].dropna().unique()
    logger.info(f"Processing elevation for {len(unique_h3)} unique cells")

    try:
        # Get centroids
        centroids = [spatial.get_h3_centroid(h) for h in unique_h3]
        lats = [c[0] for c in centroids]
        lons = [c[1] for c in centroids]

        # Fetch elevation (batch)
        logger.info("Fetching elevation data...")
        elevations = weather.fetch_elevation_batch(lats, lons)
        logger.info("Elevation fetch complete")

        # Create a mapping
        h3_elevation = dict(zip(unique_h3, elevations))
        
        # Map back to dataframe
        df_h3["elevation"] = df_h3["h3_index"].map(h3_elevation)
        missing_elev = df_h3["elevation"].isna().sum()
        if missing_elev > 0:
            logger.warning(f"Missing elevation for {missing_elev} rows")
    except Exception as e:
        logger.error(f"Failed to fetch elevation: {e}")
        return

    logger.info("Step 3: Prepare data for modeling")
    # Aggregate counts
    if "date" in df_h3.columns:
        daily_counts = (
            df_h3.groupby(["h3_index", "date"]).size().reset_index(name="count")
        )
        logger.info(f"Aggregated to {len(daily_counts)} h3_index-date combinations")

        # Merge with elevation (static per cell)
        daily_counts["elevation"] = daily_counts["h3_index"].map(h3_elevation)

        # Fill missing elevations if any
        initial_rows = len(daily_counts)
        daily_counts = daily_counts.dropna(subset=["elevation"])
        logger.info(f"Data ready: {len(daily_counts)} rows (dropped {initial_rows - len(daily_counts)} with missing elevation)")
    else:
        logger.error("Date column missing. Cannot aggregate for temporal modeling.")
        return

    logger.info("Step 4: Build and sample Bayesian model")
    if "date" in df_h3.columns and not daily_counts.empty:
        try:
            # Initialize model
            model = BayesianMushroomModel(daily_counts, target_col="count")

            # Build ZIP model using Elevation as a predictor
            # We standardize elevation for better sampling
            daily_counts["elevation_std"] = (
                daily_counts["elevation"] - daily_counts["elevation"].mean()
            ) / daily_counts["elevation"].std()
            logger.info("Standardized elevation")

            logger.info("Building ZIP model with elevation predictor...")
            model.build_zip_model(predictors=["elevation_std"])

            # Sample
            logger.info("Sampling from posterior (this may take a minute)...")
            # Reduced draws for demonstration speed
            # cores=1 is critical for Windows to prevent hangs
            model.sample(draws=500, tune=500, chains=2, cores=1)
            logger.info("Sampling complete")

            # Plot trace
            logger.info("Generating trace plots...")
            
            fig, axes = plt.subplots(3, 2, figsize=(12, 10))
            
            var_labels = {
                "alpha": "α (Intercept)",
                "betas": "β (Elevation Effect)", 
                "psi": "ψ (Zero-Inflation Prob)"
            }
            
            predictor_cols = ["elevation_std"]

            for i, var in enumerate(["alpha", "betas", "psi"]):
                # Posterior Plot (Left)
                ax_post = axes[i, 0]
                
                # Handle betas specially if multidimensional
                is_multidim_beta = False
                if var == "betas":
                    data = model.trace.posterior[var].values
                    if data.ndim == 3 and data.shape[2] > 1:
                        is_multidim_beta = True
                        # Flatten chains and draws
                        flat_data = data.reshape(-1, data.shape[-1])
                        for idx in range(flat_data.shape[1]):
                            label = predictor_cols[idx] if idx < len(predictor_cols) else f"beta_{idx}"
                            sns.kdeplot(flat_data[:, idx], ax=ax_post, label=label)
                        ax_post.legend(fontsize='x-small')
                
                if not is_multidim_beta:
                    az.plot_posterior(model.trace, var_names=[var], ax=ax_post, 
                                     hdi_prob=0.94, point_estimate="mean")
                    
                ax_post.set_title(f"{var_labels[var]} - Posterior", fontsize=11)
                ax_post.set_xlabel("Parameter Value")
                ax_post.set_ylabel("Density")
                
                # Manual trace plot to avoid ArviZ shape errors
                ax_trace = axes[i, 1]
                data = model.trace.posterior[var]
                values = data.values
                
                if values.ndim == 2:
                    for chain_idx in range(values.shape[0]):
                        ax_trace.plot(values[chain_idx], alpha=0.5)
                elif values.ndim == 3:
                    for dim_idx in range(values.shape[2]):
                        for chain_idx in range(values.shape[0]):
                            ax_trace.plot(values[chain_idx, :, dim_idx], alpha=0.3)

                ax_trace.set_title(f"{var_labels[var]} - MCMC Trace", fontsize=11)
                ax_trace.set_xlabel("Sample")
                ax_trace.set_ylabel("Parameter Value")
            
            plt.suptitle("Spatial Model: Elevation Effect on Mushroom Counts", 
                         fontsize=13, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            fig.savefig(OUTPUT_DIR / "spatial_model_trace.png", dpi=150, bbox_inches="tight")
            logger.info(f"Trace plot saved to {OUTPUT_DIR / 'spatial_model_trace.png'}")
            
            # Save summary
            summary = az.summary(model.trace, var_names=["alpha", "betas", "psi"], round_to=3)
            summary.to_csv(OUTPUT_DIR / "spatial_model_summary.csv")
            logger.info(f"Model summary saved to {OUTPUT_DIR / 'spatial_model_summary.csv'}")
            
            # Save daily counts data
            daily_counts.to_csv(OUTPUT_DIR / "spatial_daily_counts.csv", index=False)
            logger.info(f"Daily counts saved to {OUTPUT_DIR / 'spatial_daily_counts.csv'}")
        except Exception as e:
            logger.error(f"Modeling failed: {e}", exc_info=True)
    else:
        logger.error("Insufficient data for modeling")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        raise

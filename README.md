# Texas Mushrooms Scraper

A polite web scraper and data pipeline for [texasmushrooms.org](https://www.texasmushrooms.org/).

## Features

- **Polite Crawling**: Respects `robots.txt`, uses a custom User-Agent, and implements configurable delays.
- **Structured Data**: Extracts metadata into Pydantic models and saves as Parquet/CSV.
- **Image Downloading**: Optionally downloads all mushroom images for offline analysis, art, or modeling.
- **Clean Architecture**: Built with modern Python (3.11+), typed with `mypy`, and linted with `ruff`.
- **Geolocation Extraction**: Parses KMZ (KML) files linked from each day page to derive latitude/longitude for the day and, when possible, per-photo coordinates.
- **Historical Weather**: Fetches daily weather history (temp, rain, wind) from Open-Meteo for the observation period.

## Setup

1.  **Prerequisites**: Python 3.11+ installed.
2.  **Install Dependencies**:
    ```bash
    pip install -e .[dev]
    ```

## Usage

The scraper is run via the CLI.

### Basic Crawl (Metadata Only)

Crawl the first 5 days to test:

```bash
python -m texas_mushrooms.cli crawl --limit 5 --delay 1.0
```

This will create `data/raw/days.csv` and `data/raw/photos.csv`.

### Crawl with Image Download

To download images as well:

```bash
python -m texas_mushrooms.cli crawl --limit 5 --download-images
```

Images will be saved in `data/raw/images/YYYY-MM-DD/`.

### Full Crawl

To crawl the entire site (this will take a while due to the delay):

```bash
python -m texas_mushrooms.cli crawl --delay 1.0 --download-images
```

If you installed with Poetry, run:

```bash
poetry run python -m texas_mushrooms.cli crawl --delay 1.0 --download-images
```

Coordinates (latitude/longitude) are extracted automatically; no extra flags needed.

### Fetch Historical Weather

To build a clean daily weather dataset corresponding to the mushroom observation dates:

```bash
python -m texas_mushrooms.pipeline.weather
```

This infers the date range from `data/raw/days.csv` and saves weather data to `data/external/`.

### Process Data

Run the full processing pipeline to generate cleaned datasets and modeling features:

```bash
python scripts/prepare_datasets.py
```

Or run directly as a module:

```bash
python -m texas_mushrooms.pipeline.processing
```

This reads from `data/raw` and `data/external` and outputs to `data/processed`.

**Note:** By default, data is filtered to 2018–2024 (years with best coverage). Use `--no-filter` to include all years.

### Run Spatial Analysis

To run Bayesian modeling with weather and elevation predictors:

```bash
python scripts/run_spatial_analysis.py
```

## Data Output

-   `data/raw/days.csv`: Page-level metadata (date, weather, species list text, KMZ link, `latitude`, `longitude`, `photo_count`).
-   `data/raw/photos.csv`: Photo-level metadata (caption, species tags, image URL, per-photo `latitude` / `longitude` when resolvable from KMZ, else the day default).
-   `data/external/daily_weather.csv`: Daily weather metrics (temperature, precipitation, wind, humidity) from Open-Meteo.
-   `data/processed/`:
    -   `mushroom_daily.csv` / `.parquet`: Merged daily dataset with weather features for modeling.
    -   `photos_cleaned.csv`: All photos with parsed species lists.
    -   `photo_geospatial.csv`: Photos with lat/lon for mapping.
    -   `species_frequency.csv`: Species occurrence counts.
-   `data/raw/images/`: Directory containing downloaded images organized by date.

## Project Structure

```
texas-mushrooms/
├── data/
│   ├── raw/              # Scraped data (days.csv, photos.csv, images/)
│   ├── external/         # Weather data from Open-Meteo
│   └── processed/        # Cleaned & feature-engineered datasets
├── notebooks/
│   ├── EDA.ipynb         # Exploratory data analysis
│   └── spatial_analysis.ipynb
├── scripts/
│   ├── prepare_datasets.py   # Main data processing script (renamed)
│   └── run_spatial_analysis.py  # Bayesian modeling script
├── src/texas_mushrooms/
│   ├── cli.py            # Command-line interface
│   ├── scrape/           # Web scraping logic
│   │   ├── core.py
│   │   └── schemas.py
│   ├── pipeline/         # Data processing modules
│   │   ├── processing.py
│   │   ├── weather.py
│   │   └── spatial.py
│   └── modeling/
│       └── bayesian.py   # PyMC models
└── tests/
```

### Geolocation Details

- Each day page often links a KMZ file (`date-loc/YYYY-MM-DD.kmz`). This is a zipped KML track and point set.
- We parse all `<Placemark>` entries with `<Point>` geometry to build a mapping of name → coordinate.
- Photo filenames follow the pattern: `.../archives/YYYY/ROLL/jpeg/NNb.jpg`. We derive a key `ROLL-NN` (e.g. `3642-24`) and match it to the Placemark name.
- If a specific Placemark match is found, that photo receives exact coordinates.
- If no match exists, the day's first point (or first coordinate found anywhere in the KML) is used as a fallback for that photo.
- Long line tracks (`<LineString>`) are currently only used as a fallback source for the day-level coordinate when no points are present.

### Notes & Caveats

- Large KMZ track files can increase crawl time slightly; we fetch one KMZ per day in scope.
- If a KMZ is missing or malformed, `latitude` / `longitude` will be `null` for that day and its photos.
- Coordinate precision preserved as provided; no smoothing or map-matching applied.

## Development

Run tests:

```bash
pytest
```

Run type checking:

```bash
mypy src
```

Run linter:

```bash
ruff check src
```

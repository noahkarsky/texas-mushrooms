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

To generate modeling datasets and derived features:

```bash
python -m texas_mushrooms.pipeline.processing
```

This reads from `data/raw` and `data/external` and outputs to `data/processed`.

## Data Output

-   `data/raw/days.csv`: Page-level metadata (date, weather, species list text, KMZ link, `latitude`, `longitude`, `photo_count`).
-   `data/raw/photos.csv`: Photo-level metadata (caption, species tags, image URL, per-photo `latitude` / `longitude` when resolvable from KMZ, else the day default).
-   `data/external/daily_weather.csv`: Daily weather metrics (temperature, precipitation, wind, humidity) from Open-Meteo.
-   `data/processed/mushroom_daily.parquet` / `csv`: Merged dataset ready for modeling.
-   `data/raw/images/`: Directory containing downloaded images organized by date.

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

```
texas-mushrooms
├─ .pre-commit-config.yaml
├─ LICENSE
├─ notebooks
│  ├─ EDA.ipynb
│  └─ pre-process-data.py
├─ poetry.lock
├─ pyproject.toml
├─ README.md
├─ src
│  ├─ modeling
│  └─ texas_mushrooms
│     ├─ art
│     │  └─ __init__.py
│     ├─ cli.py
│     ├─ modeling
│     │  └─ __init__.py
│     ├─ pipeline
│     │  ├─ processing.py
│     │  ├─ weather.py
│     │  └─ __init__.py
│     ├─ scrape
│     │  ├─ core.py
│     │  ├─ schemas.py
│     │  └─ __init__.py
│     └─ __init__.py
└─ tests
   └─ test_scraper.py

```

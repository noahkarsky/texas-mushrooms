# CLAUDE.md — Texas Mushrooms

AI assistant guide for the `texas-mushrooms` codebase. Read this before making any changes.

## Project Overview

A polite web scraper and data pipeline for [texasmushrooms.org](https://www.texasmushrooms.org/). Extracts daily mushroom observations (photos, species IDs, geolocations from KMZ/KML files), enriches them with historical weather data from Open-Meteo, and prepares datasets for exploratory analysis and Bayesian spatial modeling.

## Development Setup

**Python**: 3.12–3.13 (strict; enforced in `pyproject.toml`)

```bash
# Install all dependencies (including dev)
pip install -e .[dev]

# Or with Poetry
poetry install
```

**Pre-commit hooks** (install once after cloning):
```bash
pre-commit install
```

## Essential Commands

### Quality Gates — run these before every commit

```bash
pytest              # Run tests
ruff check .        # Lint (auto-fix: ruff check --fix .)
ruff format .       # Format (line length: 88)
mypy src            # Type-check (strict mode)
```

Pre-commit runs ruff + mypy automatically on staged files.

### Scraping & Data Collection

```bash
# Test crawl — first 5 days, metadata only
python -m texas_mushrooms.cli crawl --limit 5 --delay 1.0

# Full crawl with image download (long-running, be polite)
python -m texas_mushrooms.cli crawl --delay 1.0 --download-images

# Fetch historical weather (requires data/raw/days.csv to exist)
python -m texas_mushrooms.pipeline.weather
```

### Processing Pipeline

```bash
# Full processing pipeline (taxonomy filter + spatial filter + feature engineering)
python scripts/prepare_datasets.py

# Options
python scripts/prepare_datasets.py --no-filter-years    # Include all years (default: 2018–2024)
python scripts/prepare_datasets.py --no-filter-species  # Skip taxonomy filter
python scripts/prepare_datasets.py --no-spatial-filter  # Skip bounding box filter
python scripts/prepare_datasets.py --bbox "29.9,31.2,-95.9,-94.0"

# Run as module (equivalent)
python -m texas_mushrooms.pipeline.processing
```

### Spatial Analysis / Modeling

```bash
python scripts/run_spatial_analysis.py
```

## Repository Layout

```
texas-mushrooms/
├── src/texas_mushrooms/       # Main package
│   ├── cli.py                 # CLI entry point (crawl subcommand)
│   ├── scrape/
│   │   ├── core.py            # HTML parsing, KMZ/KML geolocation extraction
│   │   └── schemas.py         # Data models: DayPage, PhotoRecord, SpeciesRef
│   ├── pipeline/
│   │   ├── processing.py      # Preprocessing, feature engineering, exports (~650 lines)
│   │   ├── weather.py         # Open-Meteo weather API integration
│   │   ├── spatial.py         # H3 hexagonal grid indexing
│   │   └── filters.py         # Year / taxonomy / bounding-box filters
│   ├── config/
│   │   └── filter_config.py   # MushroomFilter (YAML-backed) + SpatialFilter
│   └── modeling/
│       └── bayesian.py        # PyMC Poisson & Zero-Inflated Poisson models
├── scripts/
│   ├── prepare_datasets.py    # CLI wrapper for run_full_pipeline()
│   ├── run_spatial_analysis.py
│   └── run_pipeline.py
├── notebooks/
│   ├── EDA.ipynb
│   └── spatial_analysis.ipynb
├── tests/
│   ├── test_scraper.py
│   └── test_run_spatial_analysis.py
├── config/
│   └── mushroom_filter.yaml   # Taxonomy filter config (edit here, not in code)
├── data/
│   ├── raw/                   # Scraped output: days.csv, photos.csv, images/
│   ├── external/              # Weather data: daily_weather.csv
│   ├── processed/             # Pipeline output CSVs
│   └── outputs/               # Analysis artefacts
├── pyproject.toml
├── .pre-commit-config.yaml
└── README.md
```

## Data Flow

```
texasmushrooms.org
        │
        ▼  cli crawl
  scrape/core.py
  ├── parse_index()       → list of day URLs
  ├── parse_day_page()    → DayPage (weather summary, species, photos)
  └── parse_kmz()         → per-photo lat/lon via ROLL-NN placemark matching
        │
        ▼  data/raw/
  days.csv          (1 row/day: date, weather_summary, species_text, kmz_url, lat, lon)
  photos.csv        (1 row/photo: date, photo_url, caption, species_list, lat, lon)
  images/YYYY-MM-DD/
        │
        ▼  pipeline/weather.py
  data/external/daily_weather.csv  (temp, rain, wind, humidity, soil_temp, soil_moisture)
        │
        ▼  pipeline/processing.py → scripts/prepare_datasets.py
  data/processed/
  ├── photos_cleaned.csv      (photos + parsed species lists)
  ├── photo_geospatial.csv    (photos with valid lat/lon)
  ├── species_frequency.csv   (occurrence counts)
  └── mushroom_daily.csv      (daily timeseries + weather features for modeling)
        │
        ▼  modeling/bayesian.py
  Poisson / Zero-Inflated Poisson regression (PyMC)
```

## Code Conventions

### Type Hints
- Strict `mypy` compliance is required (`strict = true` in `pyproject.toml`).
- Add `from __future__ import annotations` at the top of every module.
- Never use `Any` unless unavoidable; document why if used.

### Data Structures
- **Scraping layer**: Use Python `dataclasses` (`DayPage`, `PhotoRecord`, `SpeciesRef`).
- **Processing layer**: Use `pandas.DataFrame`. Always parse dates with `pd.to_datetime`.
- **Paths**: Always use `pathlib.Path`, never raw strings for file paths.

### Web Scraping — Politeness Rules (non-negotiable)
- Never remove or reduce the `--delay` default (1.0 s between requests).
- Always pass the custom User-Agent and `Referer` header (see `core.py`).
- Respect `robots.txt` (already handled in `crawl()`).
- Handle HTTP errors gracefully — log and continue, do not raise and abort.

### Logging
- Use the standard `logging` module, not `print`.
- Log at `INFO` for progress milestones and `WARNING`/`ERROR` for failures.

### Configuration
- Taxonomy inclusions/exclusions belong in `config/mushroom_filter.yaml`, not hardcoded in Python.
- Spatial defaults (Houston bbox, KIAH weather station coords) live in `filter_config.py` and `weather.py` respectively; update there if the study area changes.

### Formatting
- Line length: **88** characters (ruff default).
- Ruff handles all formatting; do not manually reformat — just run `ruff format .`.

## Key Design Decisions

### Geolocation via KMZ
Each day page may link a KMZ (zipped KML) file. Photos follow the URL pattern `.../archives/YYYY/ROLL/jpeg/NNb.jpg`. The scraper derives a `ROLL-NN` key (e.g., `3642-24`) and matches it against KML `<Placemark>` names to assign per-photo coordinates. If no match exists, the day-level coordinate is used as fallback.

### Taxonomy Filtering
`MushroomFilter` applies a three-level priority chain:
1. **Species whitelist** — always include (e.g., medicinal Ganoderma spp.)
2. **Species/genus blacklist** — always exclude shelf fungi, crusts, slime molds, lichens
3. **Caption keywords** — fallback for unlabelled photos (e.g., "resupinate", "shelf fungus")

Edit `config/mushroom_filter.yaml` to adjust; Python code should not contain taxon names.

### Temporal Scope
Default filter: **2018–2024** — years with the most consistent observation coverage. Pass `--no-filter-years` to include all scraped years.

### Spatial Scope
Default bounding box: Houston area (29.9–31.2°N, 95.9–94.0°W). Configurable via `--bbox`.

### H3 Spatial Indexing
`pipeline/spatial.py` converts lat/lon to H3 hexagonal indices at resolutions 7 (~1.2 km) and 8 (~0.46 km), creating a spatiotemporal grid that handles zero-inflation in Bayesian models.

### Modelling
`modeling/bayesian.py` provides:
- `build_poisson_model()` — standard Poisson regression
- `build_zip_model()` — Zero-Inflated Poisson (preferred; most days have zero observations)
- Weather predictors: `rain_1d`, `rain_3d`, `rain_7d` (rolling), `temp_range`, seasonal `sin`/`cos` day-of-year features

## Data Files — Do Not Commit

`data/` is git-ignored. Never commit CSV data files or downloaded images. The `data/` tree is generated by running the pipeline.

## Adding New Features — Checklist

1. **Scraping changes**: Update `schemas.py` dataclasses first, then `core.py` parsing logic, then CSV export in `cli.py`.
2. **New pipeline step**: Add a function to the appropriate `pipeline/` module; call it from `run_full_pipeline()` in `processing.py`.
3. **New filter**: Add to `filters.py` and expose as a CLI flag in `scripts/prepare_datasets.py`.
4. **New model**: Add to `modeling/bayesian.py`; follow the existing `build_*_model()` + `sample()` pattern.
5. Run `pytest`, `ruff check .`, and `mypy src` before opening a PR.

## Visualization Guidelines (Tufte principles)

All plots in notebooks and scripts should:
- Maximize the data-ink ratio; remove non-data ink (gridlines, borders, backgrounds).
- Use small multiples for comparisons across species, years, or regions.
- Label axes and use appropriate scales; avoid chartjunk.
- Use color purposefully, not decoratively.

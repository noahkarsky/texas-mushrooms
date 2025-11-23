# Texas Mushrooms Scraper - AI Instructions

## Project Overview
Web scraper and data pipeline for `texasmushrooms.org`. Extracts daily mushroom observations, photos, and species identifications, enriches them with historical weather data, and prepares datasets for analysis.

## Architecture
- **Entry Points**:
  - `src/texas_mushrooms/cli.py`: Main CLI for scraping (`crawl`) and image downloading.
  - `src/texas_mushrooms/weather.py`: Fetches historical weather from Open-Meteo.
  - `src/texas_mushrooms/daily_dataset.py`: Processes raw data into analysis-ready datasets (feature engineering).
- **Core Logic**: `src/texas_mushrooms/scraper.py` handles HTML parsing and navigation.
- **Data Models**: `src/texas_mushrooms/models.py` uses Python `dataclasses` (`DayPage`, `PhotoRecord`) as the internal schema.
- **Data Flow**:
  1. `cli.py` -> `scraper.py`: Scrapes site -> `DayPage` objects.
  2. `cli.py`: Exports raw metadata to `data/days.csv` and `data/photos.csv`.
  3. `weather.py`: Reads date range from `days.csv` -> fetches Open-Meteo data -> `data/weather/`.
  4. `daily_dataset.py`: Merges mushroom data + weather -> `data/derived/`.

## Critical Workflows

### Scraping & Data Collection
Run as a module. Always respect `delay` to be polite.
```powershell
# Basic crawl (metadata only)
python -m texas_mushrooms.cli crawl --limit 5

# Full crawl with images (long running)
python -m texas_mushrooms.cli crawl --delay 1.0 --download-images

# Fetch weather history (requires data/days.csv)
python -m texas_mushrooms.weather
```

### Data Processing
Generate derived datasets (features, cleaned data) for analysis:
```powershell
python -m texas_mushrooms.daily_dataset
```

### Development
- **Linting**: `ruff check .`
- **Type Checking**: `mypy .`
- **Testing**: `pytest`

## Code Conventions
- **Type Hinting**: strict `mypy` compliance. Use `from __future__ import annotations`.
- **Data Structures**: Use `dataclasses` for internal models. Use `pandas` for data manipulation.
- **Path Handling**: Always use `pathlib.Path`.
- **Web Scraping**:
  - Use `scraper.get_session()` for correct User-Agent.
  - Use `BeautifulSoup` with `html.parser`.
- **Logging**: Use standard `logging`.

## Data Visualization Guidelines
Follow Edward Tufte's principles for all plots (in notebooks or scripts):
- Maximize data-ink ratio; erase non-data ink.
- Use small multiples for comparisons.
- Clear labels, appropriate scales, no chartjunk.
- Use color effectively (not decoratively).

## Key Files
- `src/texas_mushrooms/models.py`: Data schema (dataclasses).
- `src/texas_mushrooms/scraper.py`: Scraping logic.
- `src/texas_mushrooms/daily_dataset.py`: Feature engineering & data merging.
- `pyproject.toml`: Dependencies & tool config.

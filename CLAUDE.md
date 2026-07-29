# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A polite web scraper + data pipeline for [texasmushrooms.org](https://www.texasmushrooms.org/) that extracts daily mushroom observations, photos, and species IDs, enriches them with weather/elevation data, runs Bayesian models, and exposes the results through a React web UI. The Python package is `texas_mushrooms` (in `src/`), installed editable.

## Commands

```bash
# Install (pip)
pip install -e .[dev]
# or Poetry — the repo has poetry.lock; the active venv lives at
#   C:/Users/noahk/AppData/Local/pypoetry/Cache/virtualenvs/texas-mushrooms-FHq9QisE-py3.12/

# Dev checks — run all of these before every commit (mypy is strict; ruff line-length 88)
pytest
pytest tests/test_scraper.py::test_parse_index_extracts_dates   # single test
mypy src
ruff check .          # auto-fix: ruff check --fix .
ruff format .

pre-commit install           # once after cloning
pre-commit run --all-files   # runs ruff, ruff-format, mypy on commit
```

Use the project venv, not whatever `pytest`/`mypy` is first on PATH — a bare `pytest` may resolve to a system Python that lacks the deps. `pyproject.toml` requires Python `>=3.12,<3.14`, and ruff/mypy `target-version`/`python_version` are pinned to `py312` to match: leaving them at `py311` makes mypy choke on numpy's stubs, which use 3.12-only `type` statements.

The pre-commit mypy hook runs in its **own isolated env**, so any stub or library it needs must be listed under `additional_dependencies` in `.pre-commit-config.yaml` — that hook also checks `scripts/` and `tests/` (24 files), which the documented `mypy src` (17 files) does not.

## Pipeline (run in this order)

Each stage reads the previous stage's output from `data/`. Stages are separate entry points, not a single orchestrator.

1. **Scrape** → `data/raw/{days.csv,photos.csv}` (and optionally `data/raw/images/YYYY-MM-DD/`)
   ```bash
   python -m texas_mushrooms.cli crawl --limit 5              # test
   python -m texas_mushrooms.cli crawl --delay 1.0 --download-images   # full
   ```
1b. **iNaturalist (separate source)** → `data/raw/inaturalist/{observations.csv,photos.csv}` (research-grade fungi in the same bbox; kept parallel to the texasmushrooms.org data, never merged)
   ```bash
   python -m texas_mushrooms.cli inat --max-pages 1 --download-images   # test
   python -m texas_mushrooms.cli inat --delay 1.0 --download-images     # full bbox pull
   ```
2. **Weather** → `data/external/daily_weather.csv` (infers date range from `days.csv`, fetches Open-Meteo)
   ```bash
   python -m texas_mushrooms.pipeline.weather
   ```
3. **Prepare datasets** → `data/processed/{photos_cleaned,photo_geospatial,species_frequency,mushroom_daily}.csv`
   ```bash
   python scripts/prepare_datasets.py
   # options: --no-filter-years  --no-filter-species  --no-spatial-filter
   #          --bbox "29.9,31.2,-95.9,-94.0"
   ```
4. **Spatial/weather models** → `data/outputs/` (PyMC summaries, trace PNGs, `spatial_daily_counts.csv`)
   ```bash
   python scripts/run_spatial_analysis.py
   ```
5. **Export web assets** → `web/public/data/{h3_cells.geojson,photos_index.json}`
   ```bash
   python scripts/export_web_assets.py
   ```
5b. **Export season assets** (for the `/seasons` viz) → `web/public/data/{season_photos.json,season_weather.json}`
   ```bash
   python scripts/export_season_assets.py
   ```
6. **Web UI** (see `web/README.md`): `cd web && npm install && npm run dev`

`scripts/run_pipeline.py` is a backward-compat shim that delegates to `prepare_datasets.py`.

## Repository layout

```
texas-mushrooms/
├── src/texas_mushrooms/
│   ├── cli.py                 # CLI entry point (crawl, inat subcommands)
│   ├── web_proxy.py           # local image server + hotlink-bypass proxy
│   ├── scrape/
│   │   ├── core.py            # HTML parsing, KMZ/KML geolocation extraction
│   │   ├── inaturalist.py     # separate iNaturalist API source
│   │   └── schemas.py         # dataclasses: DayPage, PhotoRecord, SpeciesRef
│   ├── pipeline/
│   │   ├── processing.py      # preprocessing, feature engineering, exports
│   │   ├── weather.py         # Open-Meteo integration
│   │   ├── spatial.py         # H3 hexagonal grid indexing
│   │   └── filters.py         # year / taxonomy / bounding-box filters
│   ├── config/filter_config.py  # MushroomFilter (YAML-backed) + SpatialFilter
│   └── modeling/bayesian.py     # PyMC Poisson & Zero-Inflated Poisson models
├── scripts/                   # prepare_datasets, run_spatial_analysis,
│                              # export_web_assets, export_season_assets, run_web.ps1
├── web/                       # React + Vite UI (src/pages/{Map,Seasons}.tsx)
├── notebooks/                 # EDA.ipynb, spatial_analysis.ipynb
├── tests/                     # test_scraper.py, test_run_spatial_analysis.py
├── config/mushroom_filter.yaml  # taxonomy filter config (edit here, not in code)
└── data/                      # raw/ external/ processed/ outputs/ — git-ignored
```

## Architecture notes

- **`src/texas_mushrooms/scrape/`** — `core.py` does all HTTP/HTML work (checks `robots.txt`, custom `USER_AGENT`, `BASE_URL`, `html.parser`). `schemas.py` defines the internal models as **`dataclasses`** (`DayPage`, `PhotoRecord`, `SpeciesRef`) — the README's mention of Pydantic is stale; use `dataclasses.asdict`, not `model_dump`.
- **Geolocation** is derived from per-day KMZ (zipped KML) files. Photo filenames (`.../archives/YYYY/ROLL/jpeg/NNb.jpg`) map to a `ROLL-NN` key matched against KML Placemark names; the day's first point is the fallback. See README "Geolocation Details".
- **`pipeline/processing.py`** — the heart of stage 3. `run_full_pipeline` = `run_preprocessing` (clean photos, parse species, spatial + taxonomy filters, geospatial/frequency exports) then `build_modeling_dataset` (daily calendar from weather, mushroom presence, lagged rain + seasonality features → `mushroom_daily.csv`).
- **`modeling/bayesian.py`** — `BayesianMushroomModel` builds Poisson / zero-inflated-Poisson (ZIP) PyMC models. Weather predictors: `rain_1d`, `rain_3d`, `rain_7d` (rolling), `temp_range`, seasonal `sin`/`cos` day-of-year features. `scripts/run_spatial_analysis.py` runs two models: weather-predictor ZIP on `mushroom_daily.csv`, and elevation ZIP on H3-binned photos.
- **`scrape/inaturalist.py`** — a **separate, parallel data source** to the texasmushrooms.org scraper. Fetches research-grade fungi (`taxon_id=47170`, `quality_grade=research`) from the public iNaturalist API for a bbox (default `SpatialFilter.default()`), cursor-paginated via `id_above` to bypass the 10k-result cap. Emits observation-level + photo-level records under `data/raw/inaturalist/`; images are only downloaded when the photo license permits redistribution (`DOWNLOADABLE_LICENSES`). Wired as the `inat` CLI subcommand. `scripts/export_web_assets.py` emits **separate** web assets for it — `web/public/data/{h3_cells_inat.geojson,photos_index_inat.json}` tagged `source: "inaturalist"` — and the Map page has a source toggle (texasmushrooms / iNaturalist / both). iNat data is intentionally **not** fed into the Bayesian models or the Seasons viz.
- **`web_proxy.py`** — the upstream image host uses hotlink protection; this stdlib HTTP server serves local `data/raw/images/` and offers a `/proxy?url=&ref=` endpoint restricted to an `ALLOWED_NETLOCS` allowlist. Run with `python -m texas_mushrooms.web_proxy --port 8001`.
- **`pipeline/color.py`** — subject-weighted color extraction, the source of the Seasons dot colors. A whole-frame histogram returns the color of the *scene* (soil, litter, shadow), so this isolates the subject first: cluster the four border bands into a background color model (dropping any color present on only one edge, which is a mushroom running off-frame rather than real background), then weight every pixel by novelty vs. that model × local **smoothness** × a mild center prior, minus a sky/specular clamp. Keep the top 20% of pixels and run a numpy weighted k-means in **OKLab**. Two sign conventions matter and are covered by tests: the texture prior rewards *smoothness* (leaf litter is the most textured surface in frame; inverting it selects the ground), and the top-quantile gate is mandatory (background outnumbers subject 20–50×, so soft weighting alone just re-derives the scene). Where the subject cannot be separated confidently — grey/brown/black species, for which "distance from background" is definitionally the wrong signal — it falls back to the whole-frame color rather than inventing a vivid one. The cache stores subject color, frame color, confidence and separation, so the fallback thresholds are re-tunable at export time without re-decoding images.
- **`pipeline/caption_color.py` + `scripts/eval_photo_colors.py`** — how the above is *measured*. Captions state the color outright ("Light yellow resupinate fungus on a log…"), and splitting on the first locative preposition keeps only the subject phrase, which removes place-name false positives ("White Oak Bayou") by construction; that yields ~1,890 labeled photos. The metric is nearest-class-**prototype** accuracy under k-fold CV, never nearest hand-authored color anchor — an anchor bakes in an exposure assumption (a white cap in forest shade is OKLab L≈0.62 against a nominal white of ≈0.95) and ranks the extractors backwards. Report only the delta vs. v1 and the margin over the permutation floor. No LLM is involved anywhere in this path.
- **`scripts/export_season_assets.py` + `web/src/pages/Seasons.tsx`** — the "Seasons" viz. The script extracts a dominant color per photo (see `pipeline/color.py`; cached in `data/processed/photo_colors.csv`, keyed by `(photo id, algo version)` so a retuned extractor recomputes instead of silently reusing stale rows) and a per-year/day-of-year wetness anomaly (SPI-like z-score of trailing-30-day rain vs the 2007–2024 climatology). Local images are matched by `endswith("_" + url_basename)` (disk files are `NNN_`-prefixed). The React page renders everything on a single `<canvas>` (8,800 dots + weather stripes — too many nodes for SVG), with day-of-year x-axis small-multiples by year, a Year zoom (single tall row), a Month filter (zooms the x-axis), species filter, and hover previews via the proxy. Hover previews need `web_proxy.py` running.

- **Deployment (`.github/workflows/deploy.yml`)** — the web app publishes to GitHub Pages on push to `main`. Two constraints follow from Pages being static with no rewrite rule: routing must stay `HashRouter` (a `BrowserRouter` deep link 404s on refresh), and data must be fetched through `web/src/dataUrl.ts`, which prefixes `import.meta.env.BASE_URL` — a bare `/data/...` path breaks under the `/<repo>/` project subpath. The Python pipeline does not run in CI; `web/public/data/*.json` is committed. **No photographs are served publicly**: the upstream host uses hotlink protection and a public site must not push traffic onto it, so `buildProxySrc` returns `''` rather than the raw URL when no proxy is configured, and the Seasons tooltip falls back to swatch + metadata. The `/photos` grid was removed for this reason.

## Conventions & gotchas

- **Filtering is layered and repeated.** Three orthogonal filters recur across stages: **year** (`START_YEAR=2018`, `END_YEAR=2024`, best-coverage window — off via `--no-filter-years`), **spatial bbox** (`SpatialFilter.default()` ≈ Houston/Big Thicket area — off via `--no-spatial-filter`), and **taxonomy** (off via `--no-filter-species`). `--no-filter` is a deprecated alias for `--no-filter-years`.
- **Taxonomy filter is config-driven** by `config/mushroom_filter.yaml`, loaded via `MushroomFilter.from_yaml()` (in `config/filter_config.py`). It keeps "cool" stalked mushrooms and excludes crusts/slime molds/shelf fungi/lichens via a genus blacklist + species whitelist overrides + caption keywords. Priority chain: species whitelist → genus/species blacklist → caption keywords. Edit the YAML to change what's kept; Python code should not contain taxon names, and the `taxonomy_reference` section at the bottom is documentation only, not read by code.
- **Windows + PyMC:** always sample with `cores=1` — multiprocessing chains hang on Windows.
- **H3 resolution 7** is the standard for spatial binning (`spatial.add_h3_indices`); `spatial.py` handles both the new (`cell_to_boundary`/`latlng_to_cell`) and old h3-py APIs.
- Use `pathlib.Path` and `from __future__ import annotations` everywhere; scripts resolve paths relative to repo root via `Path(__file__).resolve().parent.parent`. Avoid `Any` unless unavoidable (mypy is strict); parse dates with `pd.to_datetime`.
- Use the `logging` module, not `print` — `INFO` for progress milestones, `WARNING`/`ERROR` for failures.
- **Scraping politeness (non-negotiable):** keep `--delay` ≥ 1.0, always send the custom User-Agent and `Referer`, respect `robots.txt`, and handle HTTP errors by logging and continuing rather than aborting.
- **Never commit `data/`** — it is git-ignored and regenerated by running the pipeline. No CSVs, no downloaded images.
- Plots follow Tufte principles (maximize data-ink, small multiples, labelled axes, purposeful color, no chartjunk) per the Copilot instructions.

## Adding features — checklist

1. **Scraping changes:** update `schemas.py` dataclasses first, then `core.py` parsing, then CSV export in `cli.py`.
2. **New pipeline step:** add a function to the relevant `pipeline/` module and call it from `run_full_pipeline()` in `processing.py`.
3. **New filter:** add to `filters.py` and expose a CLI flag in `scripts/prepare_datasets.py`.
4. **New model:** add to `modeling/bayesian.py`, following the existing build + sample pattern.
5. Run `pytest`, `ruff check .`, `ruff format .`, and `mypy src` before opening a PR.

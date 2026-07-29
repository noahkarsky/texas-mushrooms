# Texas Mushrooms

A polite web scraper, data pipeline, and visualization site built around
[texasmushrooms.org](https://www.texasmushrooms.org/) — a photo diary of mushroom walks in the
Houston / Big Thicket area.

The project scrapes the site's daily observation pages, pulls coordinates out of the KMZ tracks
attached to each day, enriches everything with historical weather and elevation, fits Bayesian
models of when and where mushrooms fruit, and serves the results through a React site with a map,
a seasonal visualization, and a photo browser.

A second, independent data source — research-grade fungi observations from the iNaturalist API for
the same bounding box — is kept alongside the scraped data for comparison. **The two sources are
never merged**; they are exported as parallel assets and toggled in the UI.

## Features

- **Polite crawling** — respects `robots.txt`, custom User-Agent, configurable delay.
- **Structured extraction** — day pages, photo captions, and species IDs parsed into typed
  `dataclasses`, saved as CSV.
- **Geolocation from KMZ** — per-day (and often per-photo) latitude/longitude derived from the
  zipped KML tracks linked on each day page.
- **Historical weather** — daily temperature, rain, wind, humidity from Open-Meteo for the full
  observation period.
- **Bayesian modeling** — Poisson / zero-inflated-Poisson PyMC models for weather-driven daily
  counts and elevation-driven spatial counts.
- **iNaturalist source** — cursor-paginated pull of research-grade fungi in the same bbox, with
  license-aware image downloading.
- **Web UI** — Leaflet H3 hex map and a canvas seasonal visualization (8,800 photos as dots colored
  by the mushroom's own dominant color). Static, and deployable to GitHub Pages.

## Setup

Python `>=3.12,<3.14`.

```bash
pip install -e .[dev]
```

The repo also has a `poetry.lock`; the existing Poetry venv on this machine is at
`C:/Users/noahk/AppData/Local/pypoetry/Cache/virtualenvs/texas-mushrooms-FHq9QisE-py3.12/`.

Node.js (LTS) is required for the web UI.

## Pipeline

Each stage reads the previous stage's output from `data/`. These are separate entry points, not a
single orchestrator — run them in order.

### 1. Scrape texasmushrooms.org

```bash
python -m texas_mushrooms.cli crawl --limit 5              # quick test
python -m texas_mushrooms.cli crawl --delay 1.0 --download-images   # full crawl
```

Writes `data/raw/days.csv` and `data/raw/photos.csv` (plus `data/raw/images/YYYY-MM-DD/` with
`--download-images`). Coordinates are extracted automatically; no extra flags needed.

### 1b. Fetch iNaturalist (optional, separate source)

```bash
python -m texas_mushrooms.cli inat --max-pages 1 --download-images   # quick test
python -m texas_mushrooms.cli inat --delay 1.0 --download-images     # full bbox pull
```

Writes `data/raw/inaturalist/{observations.csv,photos.csv}`. Only photos whose license permits
redistribution are downloaded. Use `--bbox MIN_LON MIN_LAT MAX_LON MAX_LAT` to override the default
Houston / Big Thicket box.

### 2. Weather

```bash
python -m texas_mushrooms.pipeline.weather
```

Infers the date range from `days.csv`, fetches Open-Meteo, writes `data/external/daily_weather.csv`.

### 3. Prepare datasets

```bash
python scripts/prepare_datasets.py
```

Cleans photos, parses species, applies the year / spatial / taxonomy filters, and builds the daily
modeling table. Outputs `data/processed/{photos_cleaned,photo_geospatial,species_frequency,mushroom_daily}.csv`.

Filter flags: `--no-filter-years`, `--no-spatial-filter`, `--no-filter-species`, `--bbox`.
(`--no-filter` is a deprecated alias for `--no-filter-years`.)

### 4. Spatial + weather models

```bash
python scripts/run_spatial_analysis.py
```

Fits two zero-inflated-Poisson models — weather predictors on `mushroom_daily.csv`, and elevation on
H3-binned photos — and writes summaries, trace plots, and `spatial_daily_counts.csv` to
`data/outputs/`.

### 5. Export web assets

```bash
python scripts/export_web_assets.py      # map + photo grid data
python scripts/export_season_assets.py   # seasonal viz data
```

See [Running the web app](#running-the-web-app) below.

### 6. Run the site

```bash
./scripts/run_web.ps1
```

`scripts/run_pipeline.py` is a backward-compat shim that delegates to `prepare_datasets.py`.

## Running the web app

The site is a Vite + React + TypeScript app in `web/`. It reads only the static JSON/GeoJSON files in
`web/public/data/`, so once those are exported it runs entirely offline — except for photo images,
which need the local proxy (see below).

### Quickest path

```powershell
./scripts/run_web.ps1
```

That script picks the Poetry venv (or `python` on PATH), runs `npm install` if `web/node_modules` is
missing, starts the image proxy on port 8001 in a background window, and runs the Vite dev server in
the foreground. Open **http://localhost:5173**. Ctrl+C stops both.

Useful switches:

- `-Export` — re-run `scripts/export_web_assets.py` first.
- `-SkipProxy` — don't start the image proxy (Seasons hover previews show swatches only).
- `-ProxyPort 8002` — use a different proxy port.

### Manual path

```powershell
# 1. Generate the data the site reads (only needed when the pipeline output changes)
python scripts/export_web_assets.py
python scripts/export_season_assets.py

# 2. Install deps once
cd web
npm install

# 3. Dev server -> http://localhost:5173
npm run dev
```

And, in a second terminal, the image server:

```powershell
python -m texas_mushrooms.web_proxy --port 8001
```

The Seasons page has a box labeled "proxy" — set it to `http://127.0.0.1:8001` for hover previews.
It defaults to that in development and to empty in a production build, since a published site has
no proxy. Without one the tooltip shows the color swatch, species and date instead of an image;
the upstream host uses hotlink protection, so no photo request is made at all.

For a production build: `npm run build` (output in `web/dist/`), preview with `npm run preview`.

### Pages

| Route | What it shows |
| --- | --- |
| `/` (Map) | Leaflet map of H3 res-7 cells, colored by total photos or mean elevation. Source toggle: texasmushrooms.org / iNaturalist / both. |
| `/seasons` | Canvas visualization — one dot per photo by day-of-year, painted the mushroom's dominant color (see [How the dot colors are measured](#how-the-dot-colors-are-measured)), over a wet/dry weather ribbon. Year zoom, month filter, species filter, hover previews. |

Routes are hash-based (`#/seasons`) so they survive a refresh on GitHub Pages.

### How the dot colors are measured

A whole-frame color histogram returns the color of the *scene* — soil, leaf
litter, shadow — because on a forest-floor photograph the mushroom is a small
minority of the pixels. `src/texas_mushrooms/pipeline/color.py` therefore
isolates the subject before measuring it: it builds a background color model
from the four border bands, weights every pixel by how unlike that background it
is, times local **smoothness** (leaf litter is the most textured surface in
frame; mushroom flesh is among the least) times a mild center prior, keeps the
top 20% of pixels, and runs a weighted k-means in OKLab. No model, no network,
no LLM — just Pillow and numpy, deterministic and seeded.

**How well it works.** Photographers state the color in ~1,890 captions ("Light
yellow resupinate fungus on a log…"), which is free ground truth.
`scripts/eval_photo_colors.py` scores each extractor by nearest-class-prototype
accuracy under 5-fold CV. Against the previous whole-frame octree extractor:

| | old | new |
| --- | --- | --- |
| CV accuracy (chance ≈ 0.04) | 0.152 | **0.232** |
| genus color coherence (lower is better) | 0.932 | **0.819** |
| recall: yellow / orange / black | 0.045 / 0.011 / 0.026 | **0.358 / 0.236 / 0.569** |
| recall: white / red | 0.152 / 0.287 | **0.266 / 0.452** |

**Known limits.** Brown and green recall *drop* (0.378 → 0.040, 0.244 → 0.024).
That is not a regression in disguise: the old extractor scored well on brown by
predicting brown for everything — it answered "brown" 490 times for 347 truly
brown photos, at 0.267 precision, and answered "blue" 175 times at 0.000
precision. The new extractor's precision on the colors that matter is 0.34–0.40
versus 0.01–0.27. Genuinely drab species remain hard by construction: when the
mushroom really is the color of the litter, "distance from the background" is
the wrong signal, and those photos fall back to the old whole-frame color.
The caption-labeled set is also biased — photographers name a color mainly for
distinctive specimens.

### Assets the site expects in `web/public/data/`

| File | Produced by |
| --- | --- |
| `h3_cells.geojson`, `photos_index.json` | `scripts/export_web_assets.py` |
| `h3_cells_inat.geojson`, `photos_index_inat.json` | `scripts/export_web_assets.py` (iNaturalist) |
| `season_photos.json`, `season_weather.json` | `scripts/export_season_assets.py` |

`photos_index*.json` is no longer fetched by any page — it remains an **input** to
`export_season_assets.py`, and the deploy workflow prunes it from the published site.

## Deploying

The site is fully static and lives on GitHub Pages via
[`.github/workflows/deploy.yml`](.github/workflows/deploy.yml), which builds `web/` and publishes
on every push to `main` that touches it. Enable it once under **Settings → Pages → Source: GitHub
Actions**; after that it is automatic. Published size is ~5 MB, and a visit to `/seasons` costs
about 540 KB gzipped.

The Python pipeline does **not** run in CI. The JSON and GeoJSON under `web/public/data/` are
committed, so regenerate them locally and commit when the underlying data changes.

Two constraints shape the deployment:

- **Routing uses `HashRouter`.** GitHub Pages has no rewrite rule, so `/seasons` would 404 on a
  refresh or a direct link. URLs look like `…/texas-mushrooms/#/seasons`.
- **No photographs are served publicly.** texasmushrooms.org serves its images behind hotlink
  protection, and a public site should not push its traffic onto someone else's server. With no
  proxy configured the Seasons tooltip shows the color swatch, species and date instead of an
  image, and every dot still links back to the source page. Run
  `python -m texas_mushrooms.web_proxy --port 8001` locally and enter `http://localhost:8001` in
  the page's proxy box to get hover previews while developing.

Missing files degrade gracefully — the affected page renders an empty state prompting you to run the
export.

## Data outputs

- `data/raw/days.csv` — page-level metadata (date, weather text, species list, KMZ link, lat/lon,
  photo count).
- `data/raw/photos.csv` — photo-level metadata (caption, species tags, image URL, per-photo lat/lon
  when resolvable from KMZ, else the day default).
- `data/raw/images/` — downloaded images, organized by date.
- `data/raw/inaturalist/` — `observations.csv`, `photos.csv`, and `images/` for the iNaturalist source.
- `data/external/daily_weather.csv` — Open-Meteo daily metrics.
- `data/processed/`
  - `photos_cleaned.csv` — all photos with parsed species lists.
  - `photo_geospatial.csv` — photos with lat/lon for mapping.
  - `species_frequency.csv` — species occurrence counts.
  - `mushroom_daily.csv` — merged daily dataset with lagged rain + seasonality features.
  - `photo_colors.csv` — cached per-photo color for the Seasons viz, keyed by
    `(photo id, algorithm version)`. Stores the isolated subject color, the
    whole-frame fallback, and the confidence/separation that choose between
    them, so the fallback thresholds can be retuned without re-decoding images.
- `data/outputs/` — model summaries (`*_model_summary.csv`), trace plots (`*_trace.png`),
  `spatial_daily_counts.csv`, and standalone folium H3 maps.

## Project structure

```
texas-mushrooms/
├── config/
│   └── mushroom_filter.yaml      # taxonomy filter (genus blacklist, species whitelist, keywords)
├── data/
│   ├── raw/                      # scraped data + images (+ inaturalist/)
│   ├── external/                 # Open-Meteo weather
│   ├── processed/                # cleaned & feature-engineered datasets
│   └── outputs/                  # model summaries, traces, H3 maps
├── notebooks/
│   ├── EDA.ipynb
│   └── spatial_analysis.ipynb
├── scripts/
│   ├── prepare_datasets.py       # stage 3
│   ├── run_spatial_analysis.py   # stage 4
│   ├── export_web_assets.py      # stage 5  -> map + photos data
│   ├── export_season_assets.py   # stage 5b -> seasons data
│   ├── run_web.ps1               # proxy + dev server launcher
│   └── run_pipeline.py           # compat shim -> prepare_datasets.py
├── src/texas_mushrooms/
│   ├── cli.py                    # `crawl` and `inat` subcommands
│   ├── web_proxy.py              # local image server + hotlink-bypass proxy
│   ├── config/filter_config.py   # SpatialFilter, MushroomFilter
│   ├── scrape/
│   │   ├── core.py               # HTTP/HTML, robots.txt, KMZ parsing
│   │   ├── inaturalist.py        # separate iNaturalist API source
│   │   └── schemas.py            # DayPage, PhotoRecord, SpeciesRef (dataclasses)
│   ├── pipeline/
│   │   ├── processing.py         # preprocessing + modeling dataset
│   │   ├── weather.py            # Open-Meteo fetch
│   │   ├── spatial.py            # H3 binning, elevation
│   │   └── filters.py
│   └── modeling/
│       └── bayesian.py           # PyMC Poisson / ZIP models
├── tests/
└── web/                          # React site (see web/README.md)
```

## Geolocation details

- Each day page often links a KMZ file (`date-loc/YYYY-MM-DD.kmz`) — a zipped KML track and point set.
- All `<Placemark>` entries with `<Point>` geometry are parsed into a name → coordinate mapping.
- Photo filenames follow `.../archives/YYYY/ROLL/jpeg/NNb.jpg`; the derived key `ROLL-NN` (e.g.
  `3642-24`) is matched against Placemark names.
- On a match, the photo gets exact coordinates. Otherwise the day's first point (or the first
  coordinate found anywhere in the KML) is used as a fallback.
- `<LineString>` tracks are only used as a fallback for the day-level coordinate when no points exist.
- If a KMZ is missing or malformed, latitude/longitude are `null` for that day and its photos.
- Coordinate precision is preserved as provided; no smoothing or map-matching is applied.

## Conventions & gotchas

- **Filtering is layered and repeated.** Three orthogonal filters recur across stages: **year**
  (2018–2024, the best-coverage window), **spatial bbox** (Houston / Big Thicket), and **taxonomy**.
  Each has its own opt-out flag.
- **The taxonomy filter is config-driven** by `config/mushroom_filter.yaml` — it keeps stalked
  mushrooms and drops crusts, slime molds, shelf fungi, and lichens via a genus blacklist plus
  species whitelist overrides and caption keywords. The `taxonomy_reference` section at the bottom of
  that file is documentation only; nothing reads it.
- **Windows + PyMC:** always sample with `cores=1` — multiprocessing chains hang on Windows.
- **H3 resolution 7** is the standard for spatial binning; `spatial.py` supports both the new and old
  h3-py APIs.
- **iNaturalist data is intentionally excluded** from the Bayesian models and the Seasons viz — it
  exists as a comparison layer on the Map and Photos pages only.
- Be polite when scraping: keep `--delay` at 1.0 or higher.

## Development

```bash
pytest
pytest tests/test_scraper.py::test_parse_index_extracts_dates   # single test
mypy src          # strict
ruff check .      # line-length 88
pre-commit run --all-files
```

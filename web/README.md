# Texas Mushrooms Web

Vite + React + TypeScript site for exploring the datasets in this repo. It reads only the static
files in `public/data/`, so it runs entirely offline once those are exported — except for photo
images, which need the local proxy (see below).

## Prereqs

- Node.js (LTS), so you have `node` + `npm` on PATH.
- The Python side already run through stage 5 of the pipeline (see the root README).

## Quick start

From the repo root:

```powershell
./scripts/run_web.ps1
```

That starts the image proxy on port 8001 and the Vite dev server on **http://localhost:5173**, and
runs `npm install` first if `node_modules` is missing. Ctrl+C stops both.

Switches: `-Export` (regenerate `public/data/` first), `-SkipProxy`, `-ProxyPort 8002`.

## Manual start

### 1. Export the data

```powershell
cd d:\repos\texas-mushrooms
python scripts/export_web_assets.py      # h3_cells.geojson, photos_index.json (+ _inat variants)
python scripts/export_season_assets.py   # season_photos.json, season_weather.json
```

| File in `public/data/` | Used by | Produced by |
| --- | --- | --- |
| `h3_cells.geojson` | Map | `export_web_assets.py` |
| `photos_index.json` | Photos | `export_web_assets.py` |
| `h3_cells_inat.geojson` | Map (iNaturalist) | `export_web_assets.py` |
| `photos_index_inat.json` | Photos (iNaturalist) | `export_web_assets.py` |
| `season_photos.json` | Seasons | `export_season_assets.py` |
| `season_weather.json` | Seasons | `export_season_assets.py` |

Missing files degrade gracefully — the page renders an empty state prompting you to run the export.

### 2. Install and run

```powershell
cd d:\repos\texas-mushrooms\web
npm install
npm run dev
```

Production build: `npm run build` (output in `dist/`), then `npm run preview`.

Note: `vite.config.ts` sets `base: './'` for static hosting, but the app uses `BrowserRouter`, so a
static host needs a rewrite rule sending all paths to `index.html`.

## Making images load

The upstream image host uses hotlink protection, so direct `photo_url` loads fail. Run the repo's
image server, which serves local files from `data/raw/images/` and offers an allowlisted
`/proxy?url=&ref=` endpoint:

```powershell
cd d:\repos\texas-mushrooms
python -m texas_mushrooms.web_proxy --port 8001
```

Then set **Local Images Base URL** to `http://127.0.0.1:8001` in the Map/Photos pages (Seasons has
the same field labeled "proxy"). The Seasons page's hover previews require this.

## Pages

- **`/` — Map.** Leaflet map of H3 res-7 cells, colored by total photos or mean elevation, with a
  source toggle (texasmushrooms.org / iNaturalist / both).
- **`/seasons` — Seasons.** Single `<canvas>` rendering ~8,800 photos as dots positioned by
  day-of-year and painted the mushroom's own dominant color, over a wet/dry weather ribbon. Controls:
  ribbon mode (rain / temp / genera / off), year zoom, month filter, species filter, hover preview.
  Canvas rather than SVG because that many nodes won't render smoothly.
- **`/photos` — Photos.** Paginated grid (200 at a time) with source badges and links back to the
  original observation page.

## Stack

React 18, React Router 6, Leaflet + react-leaflet, d3-scale / d3-scale-chromatic for color scales.
No state library, no CSS framework — plain `src/styles.css`.

# Texas Mushrooms Web

Vite + React + TypeScript site for exploring the datasets in this repo. It reads only the static
files in `public/data/`, so it runs entirely offline once those are exported — except for photo
images, which need the local proxy (see below).

## Prereqs

- Node.js (LTS), so you have `node` + `npm` on PATH.
- The Python side already run through stage 5 of the pipeline (see [`docs/development.md`](../docs/development.md)).

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

Note: `vite.config.ts` sets `base: './'` so the built asset paths resolve under a project subpath
like `/texas-mushrooms/`. Combined with `HashRouter` (see Pages below), that means no rewrite rule
is needed on the static host.

## Making images load

The upstream image host uses hotlink protection, so direct `photo_url` loads fail. Run the repo's
image server, which serves local files from `data/raw/images/` and offers an allowlisted
`/proxy?url=&ref=` endpoint:

```powershell
cd d:\repos\texas-mushrooms
python -m texas_mushrooms.web_proxy --port 8001
```

Then set the **proxy** field on the Seasons page to `http://127.0.0.1:8001`. The Seasons page's
hover previews require this; it defaults to that address in dev and to `''` in production.

## Pages

- **`/` — Home.** Landing page: who the archive belongs to, what the Map answers, what Seasons
  shows, and the model results. Static — it fetches nothing. The hero dot field is 400 colors
  inlined in `src/heroDots.ts` rather than loaded from `season_photos.json` (3.4 MB), since this is
  the entry route.
- **`/map` — Map.** Leaflet map of H3 res-7 cells, colored by total photos or mean elevation, with a
  source toggle (texasmushrooms.org / iNaturalist / both).
- **`/seasons` — Seasons.** Single `<canvas>` rendering ~8,800 photos as dots positioned by
  day-of-year and painted the mushroom's own dominant color, over a wet/dry weather ribbon. Controls:
  ribbon mode (rain / temp / genera / off), year zoom, month filter, species filter, hover preview.
  Canvas rather than SVG because that many nodes won't render smoothly.

Routing uses `HashRouter` (`#/map`, `#/seasons`) so deep links survive a refresh on GitHub Pages, and
data files are fetched through `src/dataUrl.ts` so they resolve under a project subpath. Because the
router owns the URL hash, in-page `<a href="#section">` anchors do not work — use
`scrollIntoView` if section jumps are ever needed.

Map and Seasons are `React.lazy` chunks; only Home is in the entry bundle, and `leaflet/dist/leaflet.css`
is imported from `pages/Map.tsx` so it does not load for visitors who never open the map.

A Photos grid used to live at `/photos`; it was removed because the site cannot serve photographs
publicly — see the deployment notes in [`docs/development.md`](../docs/development.md).

## Stack

React 18, React Router 6, Leaflet + react-leaflet, d3-scale / d3-scale-chromatic for color scales.
No state library, no CSS framework — plain `src/styles.css`, whose `:root` block holds the dark
palette. Note that `Seasons.tsx` also hardcodes those same hex values as canvas `fillStyle` strings;
a 2D context cannot read CSS custom properties, so changing a palette value means grepping
`Seasons.tsx` for the old literal too.

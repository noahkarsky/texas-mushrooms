# Texas Mushrooms

**→ [noahkarsky.github.io/texas-mushrooms](https://noahkarsky.github.io/texas-mushrooms/)**

Seventeen years of one man's mushroom photographs, turned into a map and a calendar.

## The archive

The photographs and captions are **Alexey Sergeev's**. He teaches mathematics at Texas A&M and has
been publishing his mushroom walks at [texasmushrooms.org](https://www.texasmushrooms.org/) since
October 2007 — 19,156 pictures of 1,232 species by his own count, and still growing. He carries a
GPS and names the color, the place, and the substrate in his captions, which is the only reason any
of the analysis below is possible.

This project is built from that public archive. **It is not affiliated with him.** No photographs
are republished here: the source site uses hotlink protection, and a mirror should not push traffic
onto someone else's server. What the site shows is colors, coordinates, dates, and a link back to
the original page.

## What's on the site

- **[Map](https://noahkarsky.github.io/texas-mushrooms/#/map)** — 8,800 photographs binned into H3
  hexagons, by count or mean elevation, next to iNaturalist's coverage of the same area. The gap
  between them is the difference between a crowd and a route.
- **[Seasons](https://noahkarsky.github.io/texas-mushrooms/#/seasons)** — one dot per photograph by
  day of year, painted the mushroom's own [measured color](docs/color-measurement.md), over a
  wet/dry weather ribbon.
- **The numbers** — two zero-inflated Poisson models over 2,511 days of Open-Meteo weather. Soil
  moisture drives fruiting; last week's rainfall total, the folk-wisdom predictor, shows no
  detectable effect.

Figures on the site describe the 2018–2024 filtered analysis set (8,800 photos, 718 species,
46 cells), not the full archive.

## The repository

A polite scraper, a pandas/PyMC pipeline, and a static React site. Coordinates come from the KMZ
track logs attached to each day page; weather from Open-Meteo; a second, independent pull of
research-grade fungi from the iNaturalist API is kept alongside the scraped data and **never merged
with it**.

- [docs/development.md](docs/development.md) — setup, pipeline stages, running the site, deploying.
- [docs/color-measurement.md](docs/color-measurement.md) — how the dot colors are derived and scored.
- [web/README.md](web/README.md) — front-end specifics.

```bash
pip install -e .[dev]
```

MIT licensed (see [LICENSE](LICENSE)) — the code only. The photographs and captions belong to Alexey
Sergeev, and are neither copied nor relicensed here.

"""Export assets for the seasonal mushroom-color visualization.

Produces two files consumed by the web app's ``/seasons`` page:

* ``web/public/data/season_photos.json`` -- one record per photo, carrying the
  dominant color of the mushroom in the photographer's own image, its
  day-of-year, and the metadata needed to link back to the source page.
* ``web/public/data/season_weather.json`` -- per-year, per-day-of-year rainfall,
  soil moisture, and a wetness anomaly (SPI-like z-score) that forms the
  background "mushrooms follow the rain" story.

Color extraction is slow, so it is a standalone stage (not folded into the fast
``export_web_assets.py``). Results are cached in ``data/processed/photo_colors.csv``
keyed by ``(photo id, algorithm version)``, so a retuned extractor recomputes
rather than silently reusing stale rows.

The cache stores the subject color, the whole-frame fallback color, and the
confidence/separation that decide between them, so the fallback thresholds can be
retuned at export time without re-decoding a single image.

Run:
    python scripts/export_season_assets.py                # full run
    python scripts/export_season_assets.py --sample 200   # quick check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import date as date_cls
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from texas_mushrooms.pipeline.color import (  # noqa: E402
    ALGO_VERSION,
    ColorExtractionError,
    ColorParams,
    PhotoColor,
    apply_fallback,
    extract_photo_color,
)
from texas_mushrooms.pipeline.photo_assets import (  # noqa: E402
    genus_of,
    index_local_images,
    match_local_file,
    photo_id,
)

logger = logging.getLogger("export_season_assets")

DATA_DIR = REPO_ROOT / "data"
IMAGES_DIR = DATA_DIR / "raw" / "images"
WEB_PUBLIC_DATA_DIR = REPO_ROOT / "web" / "public" / "data"

PHOTOS_INDEX_JSON = WEB_PUBLIC_DATA_DIR / "photos_index.json"
PHOTOS_RAW_CSV = DATA_DIR / "raw" / "photos.csv"
PHOTOS_CLEANED_CSV = DATA_DIR / "processed" / "photos_cleaned.csv"
DAILY_WEATHER_CSV = DATA_DIR / "external" / "daily_weather.csv"
COLOR_CACHE_CSV = DATA_DIR / "processed" / "photo_colors.csv"

SEASON_PHOTOS_JSON = WEB_PUBLIC_DATA_DIR / "season_photos.json"
SEASON_WEATHER_JSON = WEB_PUBLIC_DATA_DIR / "season_weather.json"

# Display window used across the web app.
START_YEAR = 2018
END_YEAR = 2024

LEGACY_ALGO = "v1-octree"
CACHE_COLUMNS = [
    "id",
    "algo",
    "color",
    "subject_color",
    "frame_color",
    "swatches",
    "confidence",
    "separation",
    "background",
    "source",
]
CACHE_FLUSH_EVERY = 500


# --------------------------------------------------------------------------- #
# Color cache
# --------------------------------------------------------------------------- #
CacheKey = tuple[str, str]
CacheRow = dict[str, Any]


def _load_color_cache() -> dict[CacheKey, CacheRow]:
    """Load cached measurements, keyed by ``(photo id, algorithm version)``.

    Rows written before the cache carried a version are attributed to the legacy
    octree extractor, so they can still be selected with ``--algo`` but never
    satisfy a lookup for the current algorithm.
    """
    if not COLOR_CACHE_CSV.exists():
        return {}
    df = pd.read_csv(COLOR_CACHE_CSV)
    if "algo" not in df.columns:
        df["algo"] = LEGACY_ALGO
    df["algo"] = df["algo"].fillna(LEGACY_ALGO)

    cache: dict[CacheKey, CacheRow] = {}
    for row in df.to_dict("records"):
        key = (str(row["id"]), str(row["algo"]))
        cache[key] = {col: row.get(col) for col in CACHE_COLUMNS}
        cache[key]["id"], cache[key]["algo"] = key
    return cache


def _write_color_cache(cache: dict[CacheKey, CacheRow]) -> None:
    """Persist every algorithm's rows, so switching back with --algo is free."""
    COLOR_CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = [cache[key] for key in sorted(cache)]
    pd.DataFrame(rows, columns=CACHE_COLUMNS).to_csv(COLOR_CACHE_CSV, index=False)


def _cache_row(pid: str, result: PhotoColor) -> CacheRow:
    return {
        "id": pid,
        "algo": result.algo,
        "color": result.color,
        "subject_color": result.subject_color,
        "frame_color": result.frame_color,
        "swatches": "|".join(result.swatches),
        "confidence": round(result.confidence, 4),
        "separation": round(result.separation, 4),
        "background": result.background,
        "source": result.source,
    }


def _resolve_color(row: CacheRow, params: ColorParams) -> tuple[str, list[str]]:
    """Apply the fallback gate to a cached measurement.

    Legacy rows carry no confidence/separation, so they are used verbatim.
    """
    swatches = [s for s in str(row.get("swatches") or "").split("|") if s]
    subject = row.get("subject_color")
    frame = row.get("frame_color")
    if not isinstance(subject, str) or not isinstance(frame, str):
        return str(row.get("color")), swatches

    gated = apply_fallback(
        PhotoColor(
            color=str(row.get("color")),
            subject_color=subject,
            frame_color=frame,
            swatches=tuple(swatches),
            confidence=float(row.get("confidence") or 0.0),
            separation=float(row.get("separation") or 0.0),
            background=str(row.get("background") or ""),
            source=str(row.get("source") or "frame"),
        ),
        params,
    )
    return gated.color, swatches


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _extract_one(args: tuple[str, str]) -> tuple[str, CacheRow | None]:
    """Worker: (photo_id, path) -> cache row, or None if the file is unreadable."""
    pid, path_str = args
    try:
        return pid, _cache_row(pid, extract_photo_color(Path(path_str)))
    except ColorExtractionError as exc:
        logging.getLogger("export_season_assets").warning("%s", exc)
        return pid, None


def _work_items() -> list[tuple[str, Path]]:
    """Every downloaded image in the display window, as ``(photo id, path)``.

    Driven by the raw scrape rather than by ``photos_index.json`` so that photos
    currently excluded by the taxonomy and spatial filters are measured too --
    loosening those filters then costs nothing.
    """
    df = pd.read_csv(PHOTOS_RAW_CSV, usecols=["date", "photo_url"])
    df = df.dropna(subset=["date", "photo_url"])
    year = df["date"].astype(str).str.slice(0, 4)
    df = df[(year >= str(START_YEAR)) & (year <= str(END_YEAR))]

    by_date = index_local_images(IMAGES_DIR)
    items: dict[str, Path] = {}
    for row in df.itertuples(index=False):
        url = str(getattr(row, "photo_url"))
        local = match_local_file(str(getattr(row, "date")), url, by_date)
        if local is not None:
            items.setdefault(photo_id(url), local)
    return sorted(items.items())


def extract_colors(
    *,
    algo: str,
    limit: int | None,
    sample: int | None,
    force: bool,
    workers: int | None,
    seed: int,
) -> None:
    """Measure every image that has no cached result for the current algorithm."""
    if algo != ALGO_VERSION:
        logger.info("--algo %s selected; skipping extraction", algo)
        return

    cache = _load_color_cache()
    pending = [
        (pid, path)
        for pid, path in _work_items()
        if force or (pid, ALGO_VERSION) not in cache
    ]

    if sample is not None and sample < len(pending):
        rng = pd.Series(range(len(pending))).sample(n=sample, random_state=seed)
        pending = [pending[i] for i in sorted(rng)]
    elif limit is not None:
        pending = pending[:limit]

    if not pending:
        logger.info("All colors already cached for %s.", ALGO_VERSION)
        return

    logger.info("Extracting colors for %d images (%s)...", len(pending), ALGO_VERSION)
    jobs = [(pid, str(path)) for pid, path in pending]
    failures = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (pid, row) in enumerate(
            pool.map(_extract_one, jobs, chunksize=32), start=1
        ):
            if row is None:
                failures += 1
            else:
                cache[(pid, ALGO_VERSION)] = row
            # Flush periodically: a full run reads tens of GB and must survive
            # being interrupted partway through.
            if i % CACHE_FLUSH_EVERY == 0:
                _write_color_cache(cache)
                logger.info("  ...%d/%d", i, len(pending))
    _write_color_cache(cache)
    logger.info("Extraction complete (%d unreadable).", failures)


# --------------------------------------------------------------------------- #
# season_photos.json
# --------------------------------------------------------------------------- #
def _day_of_year(date_str: str) -> int | None:
    try:
        y, m, d = (int(x) for x in date_str.split("-"))
        return date_cls(y, m, d).timetuple().tm_yday
    except (ValueError, AttributeError):
        return None


def _photo_meta_lookup() -> dict[str, tuple[str | None, str | None]]:
    """Map photo_url -> (friendly name, genus) from the cleaned CSV."""
    if not PHOTOS_CLEANED_CSV.exists():
        return {}
    df = pd.read_csv(PHOTOS_CLEANED_CSV)
    lookup: dict[str, tuple[str | None, str | None]] = {}
    for row in df.itertuples(index=False):
        url = str(getattr(row, "photo_url", "") or "")
        if not url:
            continue
        common = str(getattr(row, "common_name", "") or "").strip()
        species = str(getattr(row, "first_species", "") or "").strip()
        name: str | None = common or species
        if not name or name.lower() == "nan":
            name = None
        genus = genus_of(species) if species.lower() != "nan" else None
        lookup[url] = (name, genus)
    return lookup


def export_season_photos(algo: str, params: ColorParams) -> None:
    photos: list[dict[str, Any]] = json.loads(
        PHOTOS_INDEX_JSON.read_text(encoding="utf-8")
    )
    cache = _load_color_cache()
    meta = _photo_meta_lookup()

    out_rows: list[dict[str, Any]] = []
    no_color = 0
    from_subject = 0
    for p in photos:
        pid = str(p["id"])
        date_str = str(p.get("date", ""))
        doy = _day_of_year(date_str)
        if doy is None:
            continue
        year = int(date_str[:4])
        if year < START_YEAR or year > END_YEAR:
            continue

        color: str | None = None
        swatches: list[str] = []
        row = cache.get((pid, algo))
        if row is None:
            no_color += 1
        else:
            color, swatches = _resolve_color(row, params)
            if row.get("source") == "subject":
                from_subject += 1

        species, genus = meta.get(str(p.get("photo_url", "")), (None, None))
        label = p.get("label_species")
        if label and str(label) != "Unidentified":
            if not species:
                species = str(label)
            if not genus:
                genus = genus_of(str(label))

        out_rows.append(
            {
                "id": pid,
                "date": date_str,
                "doy": doy,
                "year": year,
                "color": color,
                "swatches": swatches,
                "species": species,
                "genus": genus,
                "photo_url": p.get("photo_url"),
                "page_url": p.get("page_url"),
            }
        )

    SEASON_PHOTOS_JSON.parent.mkdir(parents=True, exist_ok=True)
    SEASON_PHOTOS_JSON.write_text(
        json.dumps(out_rows, separators=(",", ":")), encoding="utf-8"
    )
    logger.info(
        "Wrote %s (%d photos, %d without a local image, %d from subject isolation)",
        SEASON_PHOTOS_JSON,
        len(out_rows),
        no_color,
        from_subject,
    )


# --------------------------------------------------------------------------- #
# season_weather.json
# --------------------------------------------------------------------------- #
def export_season_weather() -> None:
    df = pd.read_csv(DAILY_WEATHER_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    rain = df["rain_sum"].fillna(0.0)
    # Trailing windows (bloom-relevant smoothing / accumulation).
    df["rain7"] = rain.rolling(7, min_periods=1).sum()
    df["rain30"] = rain.rolling(30, min_periods=1).sum()

    df["doy"] = df["date"].dt.dayofyear
    df["year"] = df["date"].dt.year

    # Climatology of the trailing-30-day rain total, per day-of-year, over the
    # full record. Smooth with a +/-7 day window because per-doy stats are noisy.
    clim = df.groupby("doy")["rain30"].agg(["mean", "std"]).reindex(range(1, 367))
    clim_mean = (
        clim["mean"].rolling(15, center=True, min_periods=1).mean().bfill().ffill()
    )
    clim_std = (
        clim["std"]
        .rolling(15, center=True, min_periods=1)
        .mean()
        .bfill()
        .ffill()
        .replace(0.0, pd.NA)
    )

    def anomaly(row: pd.Series) -> float | None:
        doy = int(row["doy"])
        mean = clim_mean.get(doy)
        std = clim_std.get(doy)
        if pd.isna(mean) or pd.isna(std) or std == 0:
            return None
        z = (float(row["rain30"]) - float(mean)) / float(std)
        return max(-3.0, min(3.0, z))

    df["anom"] = df.apply(anomaly, axis=1)

    doy_axis = list(range(1, 367))
    series: dict[str, dict[str, list[float | None]]] = {}
    years: list[int] = []
    for year in range(START_YEAR, END_YEAR + 1):
        year_df = df[df["year"] == year].set_index("doy")
        if year_df.empty:
            continue
        years.append(year)

        def col(name: str) -> list[float | None]:
            values: list[float | None] = []
            for doy in doy_axis:
                if doy in year_df.index:
                    v = year_df.at[doy, name]
                    values.append(None if pd.isna(v) else round(float(v), 2))
                else:
                    values.append(None)
            return values

        series[str(year)] = {
            "rain7": col("rain7"),
            "rain": col("rain_sum"),
            "tmean": col("temperature_mean"),
            "soil": col("soil_moisture_mean"),
            "anom": col("anom"),
        }

    payload = {
        "years": years,
        "doy": doy_axis,
        "series": series,
        "climatology": {
            "rain30_mean": [
                None if pd.isna(clim_mean.get(d)) else round(float(clim_mean[d]), 2)
                for d in doy_axis
            ]
        },
    }

    SEASON_WEATHER_JSON.parent.mkdir(parents=True, exist_ok=True)
    SEASON_WEATHER_JSON.write_text(
        json.dumps(payload, separators=(",", ":")), encoding="utf-8"
    )
    logger.info("Wrote %s (years %d-%d)", SEASON_WEATHER_JSON, years[0], years[-1])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="cap pending images")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="random subset of pending images (--limit is date-ordered, so it "
        "correlates with season and location)",
    )
    parser.add_argument("--force", action="store_true", help="re-extract cached images")
    parser.add_argument(
        "--algo", default=ALGO_VERSION, choices=[ALGO_VERSION, LEGACY_ALGO]
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-confidence", type=float, default=ColorParams().min_confidence
    )
    parser.add_argument(
        "--min-separation", type=float, default=ColorParams().min_separation
    )
    parser.add_argument("--photos-only", action="store_true")
    parser.add_argument("--weather-only", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    params = replace(
        ColorParams(),
        min_confidence=args.min_confidence,
        min_separation=args.min_separation,
    )

    if not args.weather_only:
        if not args.skip_extract:
            extract_colors(
                algo=args.algo,
                limit=args.limit,
                sample=args.sample,
                force=args.force,
                workers=args.workers,
                seed=args.seed,
            )
        export_season_photos(args.algo, params)
    if not args.photos_only:
        export_season_weather()


if __name__ == "__main__":
    main()

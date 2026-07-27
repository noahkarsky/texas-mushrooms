"""Export assets for the seasonal mushroom-color visualization.

Produces two files consumed by the web app's ``/seasons`` page:

* ``web/public/data/season_photos.json`` -- one record per photo, carrying the
  dominant color extracted from the photographer's own image, its day-of-year,
  and the metadata needed to link back to the source page.
* ``web/public/data/season_weather.json`` -- per-year, per-day-of-year rainfall,
  soil moisture, and a wetness anomaly (SPI-like z-score) that forms the
  background "mushrooms follow the rain" story.

Color extraction is slow, so this is a standalone stage (not folded into the
fast ``export_web_assets.py``). Results are cached in
``data/processed/photo_colors.csv`` so re-runs are near-instant.

Run:
    python scripts/export_season_assets.py
"""

from __future__ import annotations

import colorsys
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import date as date_cls
from pathlib import Path
from typing import Any, cast

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
IMAGES_DIR = DATA_DIR / "raw" / "images"
WEB_PUBLIC_DATA_DIR = REPO_ROOT / "web" / "public" / "data"

PHOTOS_INDEX_JSON = WEB_PUBLIC_DATA_DIR / "photos_index.json"
PHOTOS_CLEANED_CSV = DATA_DIR / "processed" / "photos_cleaned.csv"
DAILY_WEATHER_CSV = DATA_DIR / "external" / "daily_weather.csv"
COLOR_CACHE_CSV = DATA_DIR / "processed" / "photo_colors.csv"

SEASON_PHOTOS_JSON = WEB_PUBLIC_DATA_DIR / "season_photos.json"
SEASON_WEATHER_JSON = WEB_PUBLIC_DATA_DIR / "season_weather.json"

# Display window used across the web app.
START_YEAR = 2018
END_YEAR = 2024

# A swatch this unsaturated is probably leaf litter / sky, not the mushroom.
MIN_INTERESTING_SATURATION = 0.12
N_SWATCHES = 3


# --------------------------------------------------------------------------- #
# Color extraction
# --------------------------------------------------------------------------- #
def _hex(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


def _saturation(rgb: tuple[int, int, int]) -> float:
    r, g, b = (c / 255.0 for c in rgb)
    return colorsys.rgb_to_hsv(r, g, b)[1]


def _choose_color(swatches: list[tuple[int, int, int]]) -> str:
    """Pick the dot color: the most common swatch, unless it is drab.

    If the dominant swatch is desaturated (likely background), prefer the most
    saturated of the top swatches so the dot reflects the mushroom itself.
    """
    if not swatches:
        return "#888888"
    dominant = swatches[0]
    if _saturation(dominant) >= MIN_INTERESTING_SATURATION:
        return _hex(dominant)
    most_saturated = max(swatches, key=_saturation)
    return _hex(most_saturated)


def extract_colors(path: Path) -> tuple[str, list[str]]:
    """Return (chosen_color, [swatch_hex, ...]) for one image.

    Uses Pillow's fast octree quantizer on a 64px thumbnail: deterministic,
    dependency-light, and visually indistinguishable from k-means at this size.
    """
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        rgb.thumbnail((64, 64), Image.Resampling.LANCZOS)
        quantized = rgb.quantize(colors=N_SWATCHES, method=Image.Quantize.FASTOCTREE)

    palette = quantized.getpalette() or []
    # getcolors -> list of (count, palette_index); sort most-common first.
    counts = sorted(quantized.getcolors() or [], reverse=True)
    swatches: list[tuple[int, int, int]] = []
    for _, idx in counts[:N_SWATCHES]:
        base = cast(int, idx) * 3  # idx is a palette index for a quantized image
        swatches.append((palette[base], palette[base + 1], palette[base + 2]))

    chosen = _choose_color(swatches)
    return chosen, [_hex(s) for s in swatches]


def _extract_one(args: tuple[str, str]) -> tuple[str, str, str, str]:
    """Worker: (photo_id, path_str) -> (photo_id, color, swatches_str, path_str)."""
    photo_id, path_str = args
    color, swatches = extract_colors(Path(path_str))
    return photo_id, color, "|".join(swatches), path_str


# --------------------------------------------------------------------------- #
# Local image matching
# --------------------------------------------------------------------------- #
def _index_local_images() -> dict[str, list[Path]]:
    """Map each ``YYYY-MM-DD`` directory name to its image files."""
    by_date: dict[str, list[Path]] = defaultdict(list)
    if not IMAGES_DIR.exists():
        return by_date
    for day_dir in IMAGES_DIR.iterdir():
        if day_dir.is_dir():
            by_date[day_dir.name] = [
                p for p in day_dir.iterdir() if p.suffix.lower() == ".jpg"
            ]
    return by_date


def _match_local_file(
    date: str, photo_url: str, by_date: dict[str, list[Path]]
) -> Path | None:
    """Resolve a photo URL to its downloaded file.

    Disk files are prefixed (``018_12b.jpg``) while the URL basename is bare
    (``12b.jpg``); match exact basename first, else the ``_<basename>`` suffix.
    """
    candidates = by_date.get(date)
    if not candidates:
        return None
    basename = photo_url.rsplit("/", 1)[-1]
    if not basename:
        return None
    suffix = "_" + basename
    for path in candidates:
        if path.name == basename or path.name.endswith(suffix):
            return path
    return None


# --------------------------------------------------------------------------- #
# season_photos.json
# --------------------------------------------------------------------------- #
def _load_color_cache() -> dict[str, tuple[str, list[str]]]:
    if not COLOR_CACHE_CSV.exists():
        return {}
    df = pd.read_csv(COLOR_CACHE_CSV)
    cache: dict[str, tuple[str, list[str]]] = {}
    for row in df.itertuples(index=False):
        swatches_raw = str(getattr(row, "swatches", "") or "")
        swatches = [s for s in swatches_raw.split("|") if s]
        cache[str(getattr(row, "id"))] = (str(getattr(row, "color")), swatches)
    return cache


def _write_color_cache(cache: dict[str, tuple[str, list[str]]]) -> None:
    COLOR_CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": pid, "color": color, "swatches": "|".join(swatches)}
        for pid, (color, swatches) in sorted(cache.items())
    ]
    pd.DataFrame(rows, columns=["id", "color", "swatches"]).to_csv(
        COLOR_CACHE_CSV, index=False
    )


def _day_of_year(date_str: str) -> int | None:
    try:
        y, m, d = (int(x) for x in date_str.split("-"))
        return date_cls(y, m, d).timetuple().tm_yday
    except (ValueError, AttributeError):
        return None


# Tokens that look like a leading genus word but are not one.
_NON_GENUS = {"unidentified", "cf", "aff", "unknown", "sp", "spp"}


def _genus_of(scientific: str) -> str | None:
    """Derive a genus from a scientific name string.

    Takes the first whitespace-delimited token, accepts it only if it is a
    capitalized alphabetic word that is not a placeholder ("Unidentified", ...).
    """
    token = scientific.strip().split()[0] if scientific.strip() else ""
    token = token.strip(".,;:()[]")
    if not token or not token.isalpha() or not token[0].isupper():
        return None
    if token.lower() in _NON_GENUS:
        return None
    return token


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
        genus = _genus_of(species) if species.lower() != "nan" else None
        lookup[url] = (name, genus)
    return lookup


def export_season_photos() -> None:
    photos: list[dict[str, Any]] = json.loads(
        PHOTOS_INDEX_JSON.read_text(encoding="utf-8")
    )
    by_date = _index_local_images()
    cache = _load_color_cache()
    meta = _photo_meta_lookup()

    # Resolve which photos still need color extraction.
    pending: list[tuple[str, str]] = []
    matched_path: dict[str, Path] = {}
    for p in photos:
        pid = str(p["id"])
        if pid in cache:
            continue
        local = _match_local_file(
            str(p.get("date", "")), str(p.get("photo_url", "")), by_date
        )
        if local is not None:
            matched_path[pid] = local
            pending.append((pid, str(local)))

    if pending:
        print(f"Extracting colors for {len(pending)} images (parallel)...")
        with ProcessPoolExecutor() as pool:
            for i, (pid, dot_color, swatches_str, _path) in enumerate(
                pool.map(_extract_one, pending, chunksize=32), start=1
            ):
                sw = [s for s in swatches_str.split("|") if s]
                cache[pid] = (dot_color, sw)
                if i % 500 == 0:
                    print(f"  ...{i}/{len(pending)}")
        _write_color_cache(cache)
    else:
        print("All colors already cached.")

    out_rows: list[dict[str, Any]] = []
    no_color = 0
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
        if pid in cache:
            color, swatches = cache[pid]
        else:
            no_color += 1

        species, genus = meta.get(str(p.get("photo_url", "")), (None, None))
        label = p.get("label_species")
        if label and str(label) != "Unidentified":
            if not species:
                species = str(label)
            if not genus:
                genus = _genus_of(str(label))

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
    print(
        f"Wrote {SEASON_PHOTOS_JSON} "
        f"({len(out_rows)} photos, {no_color} without a local image)"
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
    print(f"Wrote {SEASON_WEATHER_JSON} (years {years[0]}-{years[-1]})")


def main() -> None:
    export_season_photos()
    export_season_weather()


if __name__ == "__main__":
    main()

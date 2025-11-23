from __future__ import annotations

# Preprocess and derive datasets from scraped data
# - Infers per-photo species from descriptions and day-level species lists
# - Exports derived artifacts into data/derived/

import re
import ast
import logging
from pathlib import Path
from typing import Iterable, Any
from collections import Counter

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ----------------------------
# Paths
# ----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DERIVED_DIR = DATA_DIR / "processed"
DERIVED_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Repo root: {REPO_ROOT}")
logger.info(f"Raw data dir: {RAW_DIR}")
logger.info(f"Processed dir: {DERIVED_DIR}")


# ----------------------------
# Helpers
# ----------------------------
def parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or value.strip() == "":
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


def normalize_species_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip()
    # Remove common qualifiers and punctuation variations
    # e.g., "Amanita muscaria cf.", "Amanita sp.", "aff.", trailing commas/periods
    s = re.sub(r"\b(cf\.|aff\.|var\.|ssp\.|spp?\.)\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[()\[\]{}]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def extract_weather_metrics(summary: str) -> dict[str, float | None]:
    pattern = re.compile(
        r"Temperature (?P<temp>[-+]?\d+\.\d)° .*?min\. (?P<min>[-+]?\d+)°, max\. (?P<max>[-+]?\d+)°.*?Precipitation (?P<precip>[-+]?\d+\.\d) mm",
        re.IGNORECASE,
    )
    if not isinstance(summary, str):
        return {"temp": None, "min": None, "max": None, "precip": None}
    m = pattern.search(summary)
    if not m:
        return {"temp": None, "min": None, "max": None, "precip": None}
    return {k: float(v) for k, v in m.groupdict().items()}


# ----------------------------
# Load data
# ----------------------------
logger.info("Loading data files...")
days_df = pd.read_csv(RAW_DIR / "days.csv")
photos_df = pd.read_csv(RAW_DIR / "photos.csv")
logger.info(f"Loaded {len(days_df)} day records and {len(photos_df)} photo records")
logger.debug(f"Days columns: {list(days_df.columns)}")
logger.debug(f"Photos columns: {list(photos_df.columns)}")

# Parse list columns if present
if "identified_species" in days_df.columns:
    logger.info("Parsing identified_species column in days...")
    days_df["identified_species_list"] = days_df["identified_species"].apply(parse_list)
else:
    logger.warning("Column 'identified_species' not found in days.csv")

if "species" in photos_df.columns:
    logger.info("Parsing species column in photos...")

    def parse_photo_species(val: Any) -> list[str]:
        lst = parse_list(val)
        # Handle list of dicts from new scraper format
        if lst and isinstance(lst, list) and len(lst) > 0 and isinstance(lst[0], dict):
            return [
                d.get("latin_name")
                for d in lst
                if isinstance(d, dict) and d.get("latin_name")
            ]
        return lst

    photos_df["species_list"] = photos_df["species"].apply(parse_photo_species)
else:
    logger.warning("Column 'species' not found in photos.csv")

# Derive convenience
photos_df["first_species"] = photos_df.get("species_list", []).apply(
    lambda xs: xs[0] if isinstance(xs, list) and xs else "Unidentified"
)
days_df["identified_species_count"] = days_df.get("identified_species_list", []).apply(
    lambda xs: len(xs) if isinstance(xs, list) else 0
)

logger.info(f"Days rows: {len(days_df)} | Photos rows: {len(photos_df)}")
logger.debug(
    f"Photos with unidentified species: {(photos_df['first_species'] == 'Unidentified').sum()}"
)
logger.debug(
    f"Days with species count: {(days_df['identified_species_count'] > 0).sum()}"
)


# ----------------------------
# Exports
# ----------------------------
logger.info("=" * 60)
logger.info("EXPORT PHASE")
logger.info("=" * 60)


def list_to_str(xs: Iterable[str]) -> str:
    return ";".join([x for x in xs if isinstance(x, str)])


# 1) Unified photos cleaned
photos_cleaned = photos_df.copy()
photos_cleaned["species_list_str"] = photos_cleaned.get("species_list", []).apply(
    list_to_str
)
# Use the first species from the list if available, else "Unidentified"
photos_cleaned["first_species"] = photos_cleaned["species_list"].apply(
    lambda xs: (xs[0] if isinstance(xs, list) and xs else "Unidentified")
)

out_photos_cleaned = DERIVED_DIR / "photos_cleaned.csv"
photos_cleaned.to_csv(out_photos_cleaned, index=False)
logger.info(f"✓ Exported {out_photos_cleaned.name:40s} (rows={len(photos_cleaned)})")


# 2) Geospatial export
photos_geo_export = photos_cleaned.dropna(subset=["latitude", "longitude"]).copy()
photos_geo_export["label_species"] = photos_geo_export["first_species"]
geo_cols = [
    "date",
    "page_url",
    "photo_url",
    "latitude",
    "longitude",
    "label_species",
]
out_geo = DERIVED_DIR / "photo_geospatial.csv"
photos_geo_export[geo_cols].to_csv(out_geo, index=False)
logger.info(f"✓ Exported {out_geo.name:40s} (rows={len(photos_geo_export)})")


# 3) Species frequency
counter: Counter[str] = Counter()
source = "photo-original"
if "species_list" in photos_df.columns:
    for xs in photos_df["species_list"]:
        counter.update(xs or [])
elif "identified_species_list" in days_df.columns:
    for xs in days_df["identified_species_list"]:
        counter.update(xs or [])
    source = "day-identified"

species_freq_df = pd.DataFrame(
    [(sp, n) for sp, n in counter.items()], columns=["species", "occurrences"]
).sort_values("occurrences", ascending=False)
species_freq_df["source"] = source

out_species_freq = DERIVED_DIR / "species_frequency.csv"
species_freq_df.to_csv(out_species_freq, index=False)
logger.info(f"✓ Exported {out_species_freq.name:40s} (rows={len(species_freq_df)})")
logger.debug(f"  Source: {source}")
logger.debug(f"  Top 5 species: {species_freq_df.head(5)['species'].tolist()}")


# 4) Weather enrichment from day-level weather_summary
if "weather_summary" in days_df.columns:
    logger.info("Extracting weather metrics...")
    weather_metrics = (
        days_df["weather_summary"].apply(extract_weather_metrics).apply(pd.Series)
    )
    weather_enriched_df = pd.concat(
        [days_df[["date", "photo_count"]], weather_metrics], axis=1
    )
    out_weather = DERIVED_DIR / "weather_enriched.csv"
    weather_enriched_df.to_csv(out_weather, index=False)
    logger.info(f"✓ Exported {out_weather.name:40s} (rows={len(weather_enriched_df)})")
else:
    logger.warning(
        "Column 'weather_summary' not found in days.csv; skipping weather_enriched.csv"
    )


logger.info("=" * 60)
logger.info("PRE-PROCESSING COMPLETE")
logger.info("=" * 60)
logger.info(f"Total exports: 4 files to {DERIVED_DIR}")

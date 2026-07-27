from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from texas_mushrooms.pipeline import spatial


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
WEB_PUBLIC_DATA_DIR = REPO_ROOT / "web" / "public" / "data"


def _h3_boundary_lonlat(h3_index: str) -> list[list[float]]:
    """Return a closed polygon ring as [[lon, lat], ...]."""
    import h3

    try:
        boundary = h3.cell_to_boundary(h3_index)
    except AttributeError:
        # Older h3-py
        boundary = h3.h3_to_geo_boundary(h3_index, geo_json=True)

    coords = [[float(lon), float(lat)] for (lat, lon) in boundary]
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def _feature_collection(features: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": list(features)}


def export_h3_cells_geojson(
    *,
    spatial_daily_counts_csv: Path,
    out_geojson: Path,
) -> None:
    df = pd.read_csv(spatial_daily_counts_csv)

    required = {"h3_index", "count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {spatial_daily_counts_csv}: {sorted(missing)}"
        )

    agg = (
        df.dropna(subset=["h3_index"])
        .groupby("h3_index", as_index=False)
        .agg(
            total_count=("count", "sum"),
            elevation=("elevation", "mean")
            if "elevation" in df.columns
            else ("count", "size"),
        )
    )

    if "elevation" not in df.columns:
        agg = agg.drop(columns=["elevation"])  # placeholder

    features: list[dict[str, Any]] = []
    for row in agg.itertuples(index=False):
        h3_index = str(getattr(row, "h3_index"))
        props: dict[str, Any] = {
            "h3_index": h3_index,
            "total_count": int(getattr(row, "total_count")),
            "source": "texasmushrooms",
        }
        if hasattr(row, "elevation") and pd.notna(getattr(row, "elevation")):
            props["elevation"] = float(getattr(row, "elevation"))

        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [_h3_boundary_lonlat(h3_index)],
                },
            }
        )

    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    out_geojson.write_text(json.dumps(_feature_collection(features)), encoding="utf-8")


def export_photos_index(
    *,
    photo_geospatial_csv: Path,
    out_json: Path,
    h3_resolution: int = 7,
) -> None:
    df = pd.read_csv(photo_geospatial_csv)

    required = {"date", "photo_url", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {photo_geospatial_csv}: {sorted(missing)}"
        )

    df = df.dropna(subset=["latitude", "longitude"]).copy()

    # Add h3_index for map-linked photo browsing.
    df = spatial.add_h3_indices(df, resolution=h3_resolution)

    def make_id(url: str) -> str:
        return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]

    def local_relpath(row: pd.Series) -> str:
        # Matches the repo's downloaded structure: data/raw/images/YYYY-MM-DD/<basename>
        date = str(row.get("date", ""))
        url = str(row.get("photo_url", ""))
        basename = url.split("/")[-1]
        return f"{date}/{basename}" if date and basename else ""

    out_rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        photo_url = str(r.get("photo_url", ""))
        if not photo_url:
            continue

        out_rows.append(
            {
                "id": make_id(photo_url),
                "date": str(r.get("date", "")),
                "label_species": None
                if pd.isna(r.get("label_species"))
                else str(r.get("label_species")),
                "photo_url": photo_url,
                "page_url": None
                if pd.isna(r.get("page_url"))
                else str(r.get("page_url")),
                "latitude": float(r["latitude"]),
                "longitude": float(r["longitude"]),
                "h3_index": None
                if pd.isna(r.get("h3_index"))
                else str(r.get("h3_index")),
                "local_relpath": local_relpath(r),
                "source": "texasmushrooms",
            }
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out_rows), encoding="utf-8")


def export_inat_h3_cells_geojson(
    *,
    inat_photos_csv: Path,
    out_geojson: Path,
    h3_resolution: int = 7,
) -> None:
    """Bin iNaturalist photo coordinates to H3 cells and write a GeoJSON.

    Unlike the texasmushrooms.org cells (which come from the modeling stage), the
    iNaturalist cells are computed directly from the fetched photo coordinates.
    """
    df = pd.read_csv(inat_photos_csv)

    required = {"latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {inat_photos_csv}: {sorted(missing)}"
        )

    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df = spatial.add_h3_indices(df, resolution=h3_resolution)
    df = df.dropna(subset=["h3_index"])

    counts = df.groupby("h3_index", as_index=False).size()
    counts = counts.rename(columns={"size": "total_count"})

    features: list[dict[str, Any]] = []
    for row in counts.itertuples(index=False):
        h3_index = str(getattr(row, "h3_index"))
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "h3_index": h3_index,
                    "total_count": int(getattr(row, "total_count")),
                    "source": "inaturalist",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [_h3_boundary_lonlat(h3_index)],
                },
            }
        )

    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    out_geojson.write_text(json.dumps(_feature_collection(features)), encoding="utf-8")


def export_inat_photos_index(
    *,
    inat_photos_csv: Path,
    out_json: Path,
    h3_resolution: int = 7,
) -> None:
    """Write an iNaturalist photos index matching the texasmushrooms schema.

    Adds ``source: "inaturalist"`` and a ``license`` field for provenance.
    """
    df = pd.read_csv(inat_photos_csv)

    required = {"date", "photo_url", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {inat_photos_csv}: {sorted(missing)}"
        )

    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df = spatial.add_h3_indices(df, resolution=h3_resolution)

    def make_id(url: str) -> str:
        return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]

    out_rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        photo_url = str(r.get("photo_url", ""))
        if not photo_url:
            continue

        local_relpath = (
            "" if pd.isna(r.get("local_relpath")) else str(r.get("local_relpath", ""))
        )

        out_rows.append(
            {
                "id": make_id(photo_url),
                "date": str(r.get("date", "")),
                "label_species": None
                if pd.isna(r.get("scientific_name"))
                else str(r.get("scientific_name")),
                "photo_url": photo_url,
                "page_url": None if pd.isna(r.get("uri")) else str(r.get("uri")),
                "latitude": float(r["latitude"]),
                "longitude": float(r["longitude"]),
                "h3_index": None
                if pd.isna(r.get("h3_index"))
                else str(r.get("h3_index")),
                "local_relpath": local_relpath,
                "source": "inaturalist",
                "license": None
                if pd.isna(r.get("photo_license"))
                else str(r.get("photo_license")),
            }
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out_rows), encoding="utf-8")


def main() -> None:
    export_h3_cells_geojson(
        spatial_daily_counts_csv=DATA_DIR / "outputs" / "spatial_daily_counts.csv",
        out_geojson=WEB_PUBLIC_DATA_DIR / "h3_cells.geojson",
    )

    export_photos_index(
        photo_geospatial_csv=DATA_DIR / "processed" / "photo_geospatial.csv",
        out_json=WEB_PUBLIC_DATA_DIR / "photos_index.json",
        h3_resolution=7,
    )

    print(f"Wrote {WEB_PUBLIC_DATA_DIR / 'h3_cells.geojson'}")
    print(f"Wrote {WEB_PUBLIC_DATA_DIR / 'photos_index.json'}")

    # iNaturalist assets (separate source) — only if the raw data has been fetched.
    inat_photos_csv = DATA_DIR / "raw" / "inaturalist" / "photos.csv"
    if inat_photos_csv.exists():
        export_inat_h3_cells_geojson(
            inat_photos_csv=inat_photos_csv,
            out_geojson=WEB_PUBLIC_DATA_DIR / "h3_cells_inat.geojson",
            h3_resolution=7,
        )
        export_inat_photos_index(
            inat_photos_csv=inat_photos_csv,
            out_json=WEB_PUBLIC_DATA_DIR / "photos_index_inat.json",
            h3_resolution=7,
        )
        print(f"Wrote {WEB_PUBLIC_DATA_DIR / 'h3_cells_inat.geojson'}")
        print(f"Wrote {WEB_PUBLIC_DATA_DIR / 'photos_index_inat.json'}")
    else:
        print(
            f"Skipping iNaturalist export: {inat_photos_csv} not found "
            "(run `python -m texas_mushrooms.cli inat` first)."
        )


if __name__ == "__main__":
    main()

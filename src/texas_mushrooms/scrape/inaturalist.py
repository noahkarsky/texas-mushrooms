"""
iNaturalist research-grade fungi fetcher for the Texas Mushrooms project.

This is a *parallel, separate* data source to the texasmushrooms.org scraper. It
pulls research-grade fungi observations from the public iNaturalist API for a
bounding box (default: the project's Houston/Big Thicket box) and normalizes them
into observation-level and photo-level records. Photos may optionally be
downloaded locally, but only when their license permits redistribution.

API docs: https://api.inaturalist.org/v1/docs/#!/Observations/get_observations
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests

from ..config.filter_config import SpatialFilter

logger = logging.getLogger(__name__)

API_URL = "https://api.inaturalist.org/v1/observations"
FUNGI_TAXON_ID = 47170  # Kingdom Fungi
PER_PAGE = 200  # iNaturalist max

# A descriptive User-Agent with contact info, per iNaturalist API etiquette.
USER_AGENT = (
    "texas-mushrooms-research/1.0 (https://github.com/noahkarsky/texas-mushrooms; "
    "contact noah.karsky8@gmail.com)"
)

# Licenses under which we are willing to download and redistribute a copy of a
# photo. All-rights-reserved (``None``/"C") photos keep their URL only.
DOWNLOADABLE_LICENSES = {
    "cc0",
    "cc-by",
    "cc-by-nc",
    "cc-by-sa",
    "cc-by-nd",
    "cc-by-nc-sa",
    "cc-by-nc-nd",
    "pd",
}


@dataclass
class ObservationRecord:
    """One iNaturalist observation."""

    observation_id: int
    observed_on: Optional[str]
    scientific_name: Optional[str]
    common_name: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    quality_grade: Optional[str]
    license_code: Optional[str]
    user_login: Optional[str]
    place_guess: Optional[str]
    uri: Optional[str]
    photo_count: int = 0


@dataclass
class PhotoRecord:
    """One photo attached to an iNaturalist observation."""

    observation_id: int
    date: Optional[str]
    scientific_name: Optional[str]
    common_name: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    photo_id: int
    photo_url: str
    photo_license: Optional[str]
    uri: Optional[str]
    local_relpath: str = ""


@dataclass
class FetchResult:
    observations: list[ObservationRecord] = field(default_factory=list)
    photos: list[PhotoRecord] = field(default_factory=list)


def get_session() -> requests.Session:
    """Return a requests session with the project's iNaturalist User-Agent."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _photo_url_at_size(url: str, size: str = "medium") -> str:
    """iNaturalist photo URLs embed the size token (square/small/medium/large).

    The API returns the ``square`` variant; swap it for a larger one.
    """
    for token in ("square", "small", "thumb"):
        if f"/{token}." in url:
            return url.replace(f"/{token}.", f"/{size}.", 1)
    return url


def _parse_location(obs: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    """Extract (lat, lon) from an observation's geojson or location string."""
    geo = obs.get("geojson")
    if isinstance(geo, dict):
        coords = geo.get("coordinates")
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            try:
                return float(coords[1]), float(coords[0])
            except (TypeError, ValueError):
                pass

    loc = obs.get("location")
    if isinstance(loc, str) and "," in loc:
        parts = loc.split(",")
        try:
            return float(parts[0]), float(parts[1])
        except (TypeError, ValueError):
            pass

    return None, None


def _normalize_observation(
    obs: dict[str, Any], bbox: SpatialFilter
) -> tuple[Optional[ObservationRecord], list[PhotoRecord]]:
    """Turn a raw API observation into typed records, or ``None`` if out of bbox."""
    lat, lon = _parse_location(obs)
    if lat is None or lon is None or not bbox.contains(lat, lon):
        return None, []

    taxon = obs.get("taxon") or {}
    scientific_name = taxon.get("name")
    common_name = taxon.get("preferred_common_name")
    user = obs.get("user") or {}
    observed_on = obs.get("observed_on")
    obs_id = int(obs["id"])
    uri = obs.get("uri")

    obs_photos = obs.get("observation_photos") or []
    photo_records: list[PhotoRecord] = []
    for op in obs_photos:
        photo = op.get("photo") or {}
        raw_url = photo.get("url")
        if not raw_url:
            continue
        photo_records.append(
            PhotoRecord(
                observation_id=obs_id,
                date=observed_on,
                scientific_name=scientific_name,
                common_name=common_name,
                latitude=lat,
                longitude=lon,
                photo_id=int(photo.get("id", 0)),
                photo_url=_photo_url_at_size(str(raw_url), "medium"),
                photo_license=photo.get("license_code"),
                uri=uri,
            )
        )

    record = ObservationRecord(
        observation_id=obs_id,
        observed_on=observed_on,
        scientific_name=scientific_name,
        common_name=common_name,
        latitude=lat,
        longitude=lon,
        quality_grade=obs.get("quality_grade"),
        license_code=obs.get("license_code"),
        user_login=user.get("login"),
        place_guess=obs.get("place_guess"),
        uri=uri,
        photo_count=len(photo_records),
    )
    return record, photo_records


def fetch_observations(
    bbox: Optional[SpatialFilter] = None,
    *,
    taxon_id: int = FUNGI_TAXON_ID,
    quality_grade: str = "research",
    delay: float = 1.0,
    max_pages: Optional[int] = None,
    session: Optional[requests.Session] = None,
) -> FetchResult:
    """Fetch research-grade observations for ``bbox`` via cursor pagination.

    Uses the ``id_above`` cursor (ordering by id ascending) to page beyond the
    API's 10,000-result window. Sleeps ``delay`` seconds between pages to stay
    within iNaturalist's rate limits.
    """
    if bbox is None:
        bbox = SpatialFilter.default()
    if session is None:
        session = get_session()

    result = FetchResult()
    id_above = 0
    page_num = 0

    while True:
        page_num += 1
        if max_pages is not None and page_num > max_pages:
            break

        params = {
            "taxon_id": taxon_id,
            "quality_grade": quality_grade,
            "swlat": bbox.min_lat,
            "swlng": bbox.min_lon,
            "nelat": bbox.max_lat,
            "nelng": bbox.max_lon,
            "photos": "true",
            "per_page": PER_PAGE,
            "order": "asc",
            "order_by": "id",
            "id_above": id_above,
        }
        params_str = {k: str(v) for k, v in params.items()}

        try:
            resp = session.get(API_URL, params=params_str, timeout=60)
        except requests.RequestException as e:
            raise ValueError(f"Failed to connect to iNaturalist API: {e}") from e

        if resp.status_code == 429:
            logger.warning("Rate limited (429); backing off 30s and retrying page.")
            page_num -= 1
            time.sleep(30)
            continue
        if resp.status_code != 200:
            raise ValueError(
                f"iNaturalist API returned error {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()
        results = data.get("results")
        if results is None:
            raise ValueError("Response JSON is missing required 'results' key")

        if not results:
            break

        for obs in results:
            record, photos = _normalize_observation(obs, bbox)
            if record is None:
                continue
            result.observations.append(record)
            result.photos.extend(photos)

        # Advance the cursor to the largest id seen on this page.
        id_above = max(int(o["id"]) for o in results)
        logger.info(
            "Fetched page %d: %d results (cumulative obs=%d, photos=%d), id_above=%d",
            page_num,
            len(results),
            len(result.observations),
            len(result.photos),
            id_above,
        )

        if len(results) < PER_PAGE:
            break

        time.sleep(delay)

    return result


def download_photo(
    session: requests.Session,
    url: str,
    dest_path: str,
) -> None:
    """Download a single photo to ``dest_path`` (skips if it already exists)."""
    if os.path.exists(dest_path):
        logger.debug("Skipping existing image: %s", dest_path)
        return

    logger.info("Downloading photo: %s", url)
    try:
        resp = session.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:  # noqa: BLE001 - best-effort download
        logger.error("Failed to download %s: %s", url, e)


def _is_downloadable(license_code: Optional[str]) -> bool:
    if not license_code:
        return False
    return license_code.lower() in DOWNLOADABLE_LICENSES


def download_photos(
    photos: list[PhotoRecord],
    image_dir: Path,
    *,
    delay: float = 1.0,
    session: Optional[requests.Session] = None,
) -> None:
    """Download license-permitting photos and set each record's ``local_relpath``.

    Files land at ``image_dir/YYYY-MM-DD/<obsid>_<photoid>.jpg``. Photos whose
    license does not permit redistribution are left with an empty ``local_relpath``.
    """
    if session is None:
        session = get_session()

    for photo in photos:
        if not _is_downloadable(photo.photo_license):
            logger.debug(
                "Skipping license-restricted photo %s (license=%s)",
                photo.photo_id,
                photo.photo_license,
            )
            continue

        date = photo.date or "unknown-date"
        filename = f"{photo.observation_id}_{photo.photo_id}.jpg"
        relpath = f"{date}/{filename}"
        dest = image_dir / date / filename

        download_photo(session, photo.photo_url, str(dest))
        if dest.exists():
            photo.local_relpath = relpath
        time.sleep(delay / 2)

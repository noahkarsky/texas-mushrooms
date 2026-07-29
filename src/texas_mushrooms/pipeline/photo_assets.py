"""Resolve scraped photo records to the image files and metadata on disk.

Shared by ``scripts/export_season_assets.py`` and ``scripts/eval_photo_colors.py``
so the URL-to-file matching rule lives in exactly one place.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

# Tokens that look like a leading genus word but are not one.
_NON_GENUS = {"unidentified", "cf", "aff", "unknown", "sp", "spp"}


def photo_id(photo_url: str) -> str:
    """Stable per-photo key. Must match ``scripts/export_web_assets.py``."""
    return hashlib.sha1(photo_url.encode("utf-8")).hexdigest()[:16]


def index_local_images(images_dir: Path) -> dict[str, list[Path]]:
    """Map each ``YYYY-MM-DD`` directory name to its downloaded image files."""
    by_date: dict[str, list[Path]] = defaultdict(list)
    if not images_dir.exists():
        return by_date
    for day_dir in images_dir.iterdir():
        if day_dir.is_dir():
            by_date[day_dir.name] = [
                p for p in day_dir.iterdir() if p.suffix.lower() == ".jpg"
            ]
    return by_date


def match_local_file(
    date: str, photo_url: str, by_date: dict[str, list[Path]]
) -> Path | None:
    """Resolve a photo URL to its downloaded file.

    Disk files carry an on-page ordinal prefix (``018_12b.jpg``) while the URL
    basename is bare (``12b.jpg``); match the exact basename first, else the
    ``_<basename>`` suffix.
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


def genus_of(scientific: str) -> str | None:
    """Derive a genus from a scientific name string.

    Takes the first whitespace-delimited token, accepting it only if it is a
    capitalized alphabetic word that is not a placeholder ("Unidentified", ...).
    """
    token = scientific.strip().split()[0] if scientific.strip() else ""
    token = token.strip(".,;:()[]")
    if not token or not token.isalpha() or not token[0].isupper():
        return None
    if token.lower() in _NON_GENUS:
        return None
    return token

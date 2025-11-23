from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional


@dataclass
class SpeciesRef:
    latin_name: str
    page_url: Optional[str] = None


@dataclass
class PhotoRecord:
    date: date
    page_url: str
    photo_url: str
    index_on_page: int

    full_caption: str

    common_name: Optional[str] = None
    species: List[SpeciesRef] = field(default_factory=list)
    location_text: Optional[str] = None

    latitude: Optional[float] = None
    longitude: Optional[float] = None


@dataclass
class DayPage:
    date: date
    url: str

    weather_summary: Optional[str] = None
    identified_species_text: Optional[str] = None
    identified_species: List[str] = field(default_factory=list)
    kmz_url: Optional[str] = None

    latitude: Optional[float] = None
    longitude: Optional[float] = None

    photos: List[PhotoRecord] = field(default_factory=list)

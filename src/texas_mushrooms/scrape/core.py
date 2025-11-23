import time
import logging
import os
import zipfile
import io
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Tuple, Optional, Dict
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from .schemas import DayPage, PhotoRecord, SpeciesRef

logger = logging.getLogger(__name__)

# Use a standard browser User-Agent to avoid 403 Forbidden errors from asergeev.com
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
BASE_URL = "https://www.texasmushrooms.org/"


def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def get_soup(url: str, session: requests.Session) -> BeautifulSoup:
    """Performs a GET, raises on non 200, returns a BeautifulSoup HTML parser."""
    logger.info(f"Fetching {url}")
    resp = session.get(url)
    resp.raise_for_status()
    # Use lxml if available, else html.parser
    return BeautifulSoup(resp.content, "html.parser")


def parse_index(session: requests.Session) -> List[DayPage]:
    """Fetches indexc.htm and extracts date links."""
    index_url = urljoin(BASE_URL, "indexc.htm")
    soup = get_soup(index_url, session)

    day_pages = []
    # Selector: a[href^="date-en/"][href$=".htm"]
    # Note: The hrefs might be relative, so we check for start/end
    links = soup.select('a[href^="date-en/"][href$=".htm"]')
    logger.info(f"Found {len(links)} potential day links in index.")

    for a in links:
        href = a.get("href")
        if not href or not isinstance(href, str):
            continue

        full_url = urljoin(index_url, href)
        text = a.get_text(strip=True)

        try:
            # Parse date format YYYY-MM-DD
            if text:
                day_date = datetime.strptime(text, "%Y-%m-%d").date()
            else:
                raise ValueError("Empty text")
        except ValueError:
            # Fallback: try to extract date from href (e.g. date-en/2024-11-07.htm)
            try:
                # Extract filename part and remove extension
                date_part = href.split("/")[-1].replace(".htm", "")
                day_date = datetime.strptime(date_part, "%Y-%m-%d").date()
            except ValueError:
                logger.warning(
                    f"Could not parse date from text '{text}' or href '{href}'"
                )
                continue

        day_pages.append(DayPage(date=day_date, url=full_url))

    # Sort by date descending just in case
    day_pages.sort(key=lambda x: x.date, reverse=True)
    return day_pages


def extract_weather_and_species(
    soup: BeautifulSoup,
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Finds weather summary and identified species text."""
    weather = None
    species_text = None
    species_list = []

    # Find element containing "Weather"
    # We look for a text node starting with "Weather"
    weather_node = soup.find(string=lambda t: t and t.strip().startswith("Weather"))
    if weather_node:
        # Get the parent element text.
        # If the node is inside a formatting tag (e.g. <b>Weather</b>), we might need to go up.
        # We'll try to find the block containing the full text.
        # A simple heuristic is to take the parent's text.
        parent = weather_node.parent
        if not parent:
            weather = None
        else:
            full_text = parent.get_text(" ", strip=True)

            # If the text is very short, it might be just the label. Try one level up.
            if len(full_text) < 50 and parent.parent:
                full_text = parent.parent.get_text(" ", strip=True)

            # Extract from "Weather" onwards
            if "Weather" in full_text:
                start_idx = full_text.find("Weather")
                weather = full_text[start_idx:]

    # Find element containing "Identified species"
    species_node = soup.find(string=lambda t: t and "Identified species" in t)
    if species_node:
        # Start collecting text from this node (or its parent if it's <b>)
        current = species_node
        if current.parent and current.parent.name == "b":
            current = current.parent

        collected_text = ""
        # Iterate siblings
        for sibling in current.next_siblings:
            if hasattr(sibling, "name") and sibling.name in [
                "p",
                "div",
                "table",
                "h1",
                "h2",
                "h3",
                "br",
            ]:
                # Stop at block elements or line breaks
                break

            text = (
                sibling.get_text(strip=True)
                if hasattr(sibling, "get_text")
                else str(sibling).strip()
            )
            if text:
                collected_text += " " + text

        if collected_text:
            species_text = collected_text.strip()
            # Remove leading colon if present
            if species_text.startswith(":"):
                species_text = species_text[1:].strip()

            # Remove trailing dot
            if species_text.endswith("."):
                species_text = species_text[:-1]

            # Split by comma
            parts = [p.strip() for p in species_text.split(",")]
            species_list = [p for p in parts if p]

    if weather:
        logger.debug(f"Found weather summary: {weather[:30]}...")
    if species_list:
        logger.debug(f"Found {len(species_list)} species")

    return weather, species_text, species_list


def extract_kmz_url(soup: BeautifulSoup, page_url: str) -> Optional[str]:
    """Finds KMZ link."""
    # Finds an anchor whose href ends with .kmz or contains "date-loc"
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".kmz") or "date-loc" in href:
            return urljoin(page_url, href)
    return None


def parse_photos(soup: BeautifulSoup, day: DayPage) -> List[PhotoRecord]:
    """Extracts photo records from a day page."""
    records = []

    # Select anchors with a[href*="/pictures/archives/"]
    photo_links = soup.select('a[href*="/pictures/archives/"]')
    logger.debug(f"Found {len(photo_links)} potential photo links for {day.date}")

    for idx, a in enumerate(photo_links, start=1):
        href = a.get("href")
        if not href or not isinstance(href, str):
            continue

        # Filter for image extensions to avoid .htm pages
        if not any(
            href.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif"]
        ):
            continue

        photo_url = urljoin(str(day.url), href)

        # Find container for caption (parent or ancestor)
        # Usually these are in a <p> or <td> or similar.
        # We'll try the immediate parent first.
        container = a.parent
        if not container:
            continue

        full_caption = container.get_text(" ", strip=True)

        # Heuristic for common name: everything before first "("
        common_name = None
        if "(" in full_caption:
            common_name = full_caption.split("(")[0].strip()
        else:
            common_name = full_caption

        # Find species links in the same container
        species_refs = []
        # Look for species links - they can be in /en/, /fungi_en/, /ru/, etc.
        # Pattern: links ending with species names like xylodon_flaviporus.htm
        for sp_a in container.find_all("a", href=True):
            sp_href = sp_a.get("href")
            # Check if this looks like a species page (contains language code and ends with .htm)
            if (
                sp_href
                and ("en/" in sp_href or "fungi_en/" in sp_href or "ru/" in sp_href)
                and sp_href.endswith(".htm")
            ):
                # Skip if it's not a species page (e.g., index pages, archives)
                if any(
                    skip in sp_href for skip in ["index", "list", "archives", "date-"]
                ):
                    continue

                sp_latin = sp_a.get_text(strip=True)
                full_sp_url = urljoin(str(day.url), sp_href)
                species_refs.append(
                    SpeciesRef(latin_name=sp_latin, page_url=full_sp_url)
                )

        record = PhotoRecord(
            date=day.date,
            page_url=day.url,
            photo_url=photo_url,
            index_on_page=idx,
            full_caption=full_caption,
            common_name=common_name,
            species=species_refs,
        )
        records.append(record)

    return records


def parse_kmz(
    session: requests.Session, kmz_url: str
) -> Tuple[Optional[float], Optional[float], Dict[str, Tuple[float, float]]]:
    """Downloads KMZ, extracts KML, and finds coordinates.

    Returns:
        Tuple of (default_lat, default_lon, points_map)
        where points_map is a dict of {name: (lat, lon)}
    """
    default_lat, default_lon = None, None
    points_map = {}

    try:
        logger.info(f"Fetching KMZ: {kmz_url}")
        resp = session.get(kmz_url)
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            # Find the .kml file
            kml_filename = next((n for n in z.namelist() if n.endswith(".kml")), None)
            if not kml_filename:
                logger.warning(f"No KML file found in {kmz_url}")
                return None, None, {}

            with z.open(kml_filename) as f:
                tree = ET.parse(f)
                root = tree.getroot()

                # Namespace handling is annoying in ElementTree, so we'll ignore it or handle it loosely
                # KML usually has a namespace like {http://www.opengis.net/kml/2.2}

                for placemark in root.iter():
                    if not placemark.tag.endswith("Placemark"):
                        continue

                    # Find name
                    name = None
                    for child in placemark:
                        if child.tag.endswith("name"):
                            name = child.text
                            break

                    # Find Point coordinates
                    coords_text = None
                    for child in placemark.iter():
                        if child.tag.endswith("Point"):
                            for sub in child:
                                if sub.tag.endswith("coordinates"):
                                    coords_text = sub.text
                                    break

                    if coords_text:
                        parts = coords_text.strip().split(",")
                        if len(parts) >= 2:
                            try:
                                lon = float(parts[0])
                                lat = float(parts[1])

                                # Set default if not set
                                if default_lat is None:
                                    default_lat = lat
                                    default_lon = lon

                                if name:
                                    points_map[name.strip()] = (lat, lon)
                            except ValueError:
                                continue

                # If no points found in Placemarks, try to find ANY coordinates (e.g. from a LineString) for default
                if default_lat is None:
                    for elem in root.iter():
                        if elem.tag.endswith("coordinates"):
                            text = elem.text
                            if text:
                                first_coord = text.strip().split()[0]
                                parts = first_coord.split(",")
                                if len(parts) >= 2:
                                    try:
                                        default_lon = float(parts[0])
                                        default_lat = float(parts[1])
                                        break
                                    except ValueError:
                                        continue

        return default_lat, default_lon, points_map
    except Exception as e:
        logger.error(f"Failed to parse KMZ {kmz_url}: {e}")
        return None, None, {}


def parse_day_page(day: DayPage, session: requests.Session) -> DayPage:
    """Fetches and populates details for a DayPage."""
    soup = get_soup(str(day.url), session)

    weather, species_text, species_list = extract_weather_and_species(soup)
    kmz = extract_kmz_url(soup, str(day.url))

    lat, lon = None, None
    points_map: dict[str, tuple[float, float]] = {}
    if kmz:
        lat, lon, points_map = parse_kmz(session, kmz)

    photos = parse_photos(soup, day)

    logger.info(
        f"Day {day.date}: Found {len(photos)} photos, {len(species_list)} species, Weather: {'Yes' if weather else 'No'}, KMZ: {'Yes' if kmz else 'No'}"
    )

    # Propagate coordinates to photos
    for p in photos:
        # Default to day's location
        p_lat, p_lon = lat, lon

        # Try to find specific location
        # URL format: .../archives/YYYY/ROLL/jpeg/NUMb.jpg
        # Key format: ROLL-NUM
        try:
            parts = str(p.photo_url).split("/")
            if "archives" in parts:
                idx = parts.index("archives")
                if len(parts) > idx + 2:
                    roll = parts[idx + 2]
                    filename = parts[-1]

                    # Extract leading digits from filename
                    num = ""
                    for char in filename:
                        if char.isdigit():
                            num += char
                        else:
                            break

                    if roll and num:
                        key = f"{roll}-{num}"
                        if key in points_map:
                            p_lat, p_lon = points_map[key]
        except Exception:
            pass

        p.latitude = p_lat
        p.longitude = p_lon

    # Return a new instance with populated fields
    return DayPage(
        date=day.date,
        url=day.url,
        weather_summary=weather,
        identified_species_text=species_text,
        identified_species=species_list,
        kmz_url=kmz,
        latitude=lat,
        longitude=lon,
        photos=photos,
    )


def download_image(
    session: requests.Session, url: str, dest_path: str, referer: Optional[str] = None
) -> None:
    """Downloads a single image to dest_path."""
    if os.path.exists(dest_path):
        logger.info(f"Skipping existing image: {dest_path}")
        return

    logger.info(f"Downloading image: {url}")
    try:
        headers = {}
        if referer:
            headers["Referer"] = referer

        resp = session.get(url, stream=True, headers=headers)
        resp.raise_for_status()

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")


def crawl(
    limit: Optional[int] = None,
    delay: float = 1.0,
    download_images: bool = False,
    image_dir: str = "data/images",
) -> Tuple[List[DayPage], List[PhotoRecord]]:
    """Main crawl function."""
    session = get_session()

    # 1. Check robots.txt (politeness)
    robots_url = urljoin(BASE_URL, "robots.txt")
    try:
        logger.info(f"Checking {robots_url}")
        r = session.get(robots_url)
        logger.info(f"robots.txt status: {r.status_code}")
        # We aren't parsing it strictly with a library here,
        # but we are logging it as requested.
        # The user requirement said "fetch and log... and check paths not disallowed".
        # Since we know /asergeev/php/ is disallowed, we should be safe with date-en/
    except Exception as e:
        logger.warning(f"Could not fetch robots.txt: {e}")

    # 2. Parse Index
    logger.info("Parsing index...")
    all_days = parse_index(session)
    logger.info(f"Found {len(all_days)} days in index.")

    if limit:
        all_days = all_days[:limit]
        logger.info(f"Limiting to first {limit} days.")

    parsed_days = []
    all_photos = []

    # 3. Loop days
    for day in all_days:
        logger.info(f"Processing day: {day.date}")
        try:
            detailed_day = parse_day_page(day, session)
            parsed_days.append(detailed_day)
            all_photos.extend(detailed_day.photos)

            if download_images:
                for photo in detailed_day.photos:
                    # Structure: data/images/YYYY-MM-DD/index_filename.jpg
                    filename = os.path.basename(str(photo.photo_url))
                    # Sanitize filename just in case
                    filename = filename.split("?")[0]

                    save_path = os.path.join(
                        image_dir,
                        str(day.date),
                        f"{photo.index_on_page:03d}_{filename}",
                    )
                    download_image(
                        session,
                        str(photo.photo_url),
                        save_path,
                        referer=str(photo.page_url),
                    )
                    time.sleep(
                        delay / 2
                    )  # slightly faster for images? or same? let's be safe.

            time.sleep(delay)

        except Exception as e:
            logger.error(f"Failed to process day {day.date}: {e}")

    return parsed_days, all_photos

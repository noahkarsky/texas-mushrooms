import argparse
import dataclasses
import logging
from pathlib import Path

import pandas as pd

from .config.filter_config import SpatialFilter
from .scrape.core import crawl


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _run_inat(args: argparse.Namespace) -> None:
    """Fetch iNaturalist research-grade fungi observations (separate source)."""
    from .scrape import inaturalist as inat

    if args.bbox:
        min_lon, min_lat, max_lon, max_lat = args.bbox
        bbox = SpatialFilter(
            min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat
        )
    else:
        bbox = SpatialFilter.default()

    logging.info("Fetching iNaturalist observations for %s", bbox)
    result = inat.fetch_observations(
        bbox=bbox,
        delay=args.delay,
        max_pages=args.max_pages,
    )
    logging.info(
        "Fetched %d observations, %d photos",
        len(result.observations),
        len(result.photos),
    )

    if args.download_images:
        image_dir = Path(args.image_dir)
        logging.info("Downloading license-permitting photos to %s", image_dir)
        inat.download_photos(result.photos, image_dir, delay=args.delay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_obs = pd.DataFrame([dataclasses.asdict(o) for o in result.observations])
    df_photos = pd.DataFrame([dataclasses.asdict(p) for p in result.photos])

    obs_path = out_dir / "observations.csv"
    photos_path = out_dir / "photos.csv"
    df_obs.to_csv(obs_path, index=False)
    df_photos.to_csv(photos_path, index=False)

    logging.info("Wrote %s (%d rows)", obs_path, len(df_obs))
    logging.info("Wrote %s (%d rows)", photos_path, len(df_photos))
    logging.info("Done!")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Texas Mushrooms Scraper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Crawl command
    crawl_parser = subparsers.add_parser(
        "crawl", help="Crawl the website and save data"
    )
    crawl_parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of days to process"
    )
    crawl_parser.add_argument(
        "--delay", type=float, default=1.0, help="Delay between requests in seconds"
    )
    crawl_parser.add_argument(
        "--out-dir", type=str, default="data/raw", help="Output directory for data"
    )
    crawl_parser.add_argument(
        "--download-images", action="store_true", help="Download images while crawling"
    )
    crawl_parser.add_argument(
        "--image-dir",
        type=str,
        default="data/raw/images",
        help="Directory to save downloaded images",
    )

    # iNaturalist command (separate, parallel data source)
    inat_parser = subparsers.add_parser(
        "inat", help="Fetch iNaturalist research-grade fungi observations"
    )
    inat_parser.add_argument(
        "--delay", type=float, default=1.0, help="Delay between API pages in seconds"
    )
    inat_parser.add_argument(
        "--out-dir",
        type=str,
        default="data/raw/inaturalist",
        help="Output directory for iNaturalist data",
    )
    inat_parser.add_argument(
        "--download-images",
        action="store_true",
        help="Download license-permitting photos locally",
    )
    inat_parser.add_argument(
        "--image-dir",
        type=str,
        default="data/raw/inaturalist/images",
        help="Directory to save downloaded photos",
    )
    inat_parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limit number of API pages fetched (200 obs/page)",
    )
    inat_parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        default=None,
        help="Bounding box (default: SpatialFilter.default(), Houston/Big Thicket)",
    )

    args = parser.parse_args()

    if args.command == "inat":
        _run_inat(args)
        return

    if args.command == "crawl":
        logging.info(f"Starting crawl with limit={args.limit}, delay={args.delay}")

        days, photos = crawl(
            limit=args.limit,
            delay=args.delay,
            download_images=args.download_images,
            image_dir=args.image_dir,
        )

        # Ensure output directory exists
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrames
        logging.info("Converting to DataFrames...")

        # Days DataFrame
        days_data = []
        for d in days:
            days_data.append(
                {
                    "date": d.date,
                    "url": str(d.url),
                    "weather_summary": d.weather_summary,
                    "identified_species_text": d.identified_species_text,
                    "identified_species": d.identified_species,
                    "kmz_url": str(d.kmz_url) if d.kmz_url else None,
                    "photo_count": len(d.photos),
                }
            )
        df_days = pd.DataFrame(days_data)

        # Photos DataFrame
        photos_data = []
        for p in photos:
            # Use dataclasses.asdict instead of model_dump
            p_dict = dataclasses.asdict(p)

            # URLs are already strings in the dataclass model, so no need to cast
            # But we can ensure they are strings just in case
            p_dict["page_url"] = str(p_dict["page_url"])
            p_dict["photo_url"] = str(p_dict["photo_url"])

            # Species is a list of dicts (from asdict)
            # URLs inside species are also strings now

            photos_data.append(p_dict)

        df_photos = pd.DataFrame(photos_data)

        # Save to CSV
        logging.info(f"Saving to {out_dir}")
        df_days.to_csv(out_dir / "days.csv", index=False)
        # Photos might have nested data (species), CSV is messy but useful for quick look
        df_photos.to_csv(out_dir / "photos.csv", index=False)

        logging.info("Done!")


if __name__ == "__main__":
    main()

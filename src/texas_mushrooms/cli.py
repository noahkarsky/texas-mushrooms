import argparse
import logging
import dataclasses
from pathlib import Path

import pandas as pd

from .scraper import crawl


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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
        "--out-dir", type=str, default="data", help="Output directory for data"
    )
    crawl_parser.add_argument(
        "--download-images", action="store_true", help="Download images while crawling"
    )
    crawl_parser.add_argument(
        "--image-dir",
        type=str,
        default="data/images",
        help="Directory to save downloaded images",
    )

    args = parser.parse_args()

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

        # Save to Parquet
        logging.info(f"Saving to {out_dir}")
        df_days.to_parquet(out_dir / "days.parquet", index=False)
        df_photos.to_parquet(out_dir / "photos.parquet", index=False)

        # Optional CSV
        df_days.to_csv(out_dir / "days.csv", index=False)
        # Photos might have nested data (species), CSV is messy but useful for quick look
        df_photos.to_csv(out_dir / "photos.csv", index=False)

        logging.info("Done!")


if __name__ == "__main__":
    main()

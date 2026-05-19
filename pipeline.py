"""
AEMO WEM Operational Demand — data ingestion pipeline.

Downloads annual CSV files from AEMO's public directory and saves them under data/.
Current year is always re-fetched (the file is updated throughout the year).

Usage
-----
# Refresh current year only (good for a daily cron job):
    python pipeline.py --current-only

# Download all years 2006 to present:
    python pipeline.py --all

# Download specific years:
    python pipeline.py --years 2022 2023 2024

# Force re-download even if file already exists:
    python pipeline.py --force

Cron example (run daily at 6 AM):
    0 6 * * * cd /path/to/project && .venv/bin/python pipeline.py --current-only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

BASE_URL = "https://data.wa.aemo.com.au/public/public-data/datafiles/operational-demand"
FILENAME_TMPL = "operational-demand-{year}.csv"
DATA_DIR = Path(__file__).parent / "data"
FIRST_YEAR = 2006
REQUEST_TIMEOUT = 120
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 5  # seconds between retries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _url_for_year(year: int) -> str:
    return f"{BASE_URL}/{FILENAME_TMPL.format(year=year)}"


def _path_for_year(year: int) -> Path:
    return DATA_DIR / FILENAME_TMPL.format(year=year)


def download_year(year: int, *, force: bool = False) -> bool:
    """Download a single year's CSV. Returns True if a new file was written."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _path_for_year(year)

    if path.exists() and not force:
        log.info("%d: already cached (%s)", year, path.name)
        return False

    url = _url_for_year(year)
    log.info("%d: fetching %s", year, url)

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            path.write_bytes(resp.content)
            log.info(
                "%d: saved %s bytes → %s",
                year,
                f"{len(resp.content):,}",
                path,
            )
            return True
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                log.warning("%d: 404 — file not available on AEMO server yet", year)
                return False
            log.warning("%d: HTTP error (attempt %d/%d): %s", year, attempt, RETRY_ATTEMPTS, exc)
        except requests.RequestException as exc:
            log.warning("%d: request error (attempt %d/%d): %s", year, attempt, RETRY_ATTEMPTS, exc)

        if attempt < RETRY_ATTEMPTS:
            log.info("Retrying in %ds…", RETRY_BACKOFF)
            time.sleep(RETRY_BACKOFF)

    log.error("%d: all %d attempts failed — skipping", year, RETRY_ATTEMPTS)
    return False


def run(years: list[int], *, force: bool = False) -> dict[str, int]:
    current_year = datetime.now().year
    updated, skipped, failed = 0, 0, 0

    for year in sorted(years):
        # Always force-refresh the current year: AEMO updates it in-place all year.
        is_current = year == current_year
        ok = download_year(year, force=force or is_current)
        if ok:
            updated += 1
        elif _path_for_year(year).exists():
            skipped += 1
        else:
            failed += 1

    log.info(
        "Finished: %d updated, %d already cached, %d failed",
        updated, skipped, failed,
    )
    return {"updated": updated, "skipped": skipped, "failed": failed}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AEMO WEM operational demand — data ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--current-only",
        action="store_true",
        help="Download/refresh the current year only (good for daily cron).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help=f"Download all available years ({FIRST_YEAR} – present).",
    )
    group.add_argument(
        "--years",
        nargs="+",
        type=int,
        metavar="YEAR",
        help="Explicit list of years to download.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists locally.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    current_year = datetime.now().year

    if args.current_only:
        years = [current_year]
    elif args.all:
        years = list(range(FIRST_YEAR, current_year + 1))
    elif args.years:
        years = args.years
    else:
        # Default: refresh current year + download previous year if missing.
        years = [current_year - 1, current_year]

    result = run(years, force=args.force)
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

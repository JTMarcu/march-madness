"""Download and extract Kaggle competition data into data/.

Usage:
    python setup.py                  # Downloads current year (2026)
    python setup.py --year 2025      # Downloads specific year
    python setup.py --year 2025 2026 # Downloads multiple years

Prerequisites:
    pip install kaggle
    Set up Kaggle API credentials: https://www.kaggle.com/docs/api#authentication
    Place kaggle.json in ~/.kaggle/ (Linux/Mac) or %USERPROFILE%\\.kaggle\\ (Windows)
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# Competition slug pattern — Kaggle names them by year
COMPETITION_TEMPLATE = "march-machine-learning-mania-{year}"


def download_competition(year: int) -> None:
    """Download and extract a single year's competition data.

    Args:
        year: Competition year (e.g., 2025, 2026).
    """
    competition = COMPETITION_TEMPLATE.format(year=year)
    zip_path = DATA_DIR / f"{competition}.zip"

    print(f"\n{'='*60}")
    print(f"Downloading: {competition}")
    print(f"{'='*60}")

    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)

    # Download via kaggle CLI
    # Try direct kaggle command first, fall back to python -m kaggle
    cmd = [
        "kaggle",
        "competitions", "download",
        "-c", competition,
        "-p", str(DATA_DIR),
    ]

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("\nERROR: kaggle CLI not found. Install it with:")
        print("  pip install kaggle")
        print("\nThen set up your API key:")
        print("  https://www.kaggle.com/docs/api#authentication")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Download failed (exit code {e.returncode})")
        print("Make sure you've accepted the competition rules on Kaggle.")
        sys.exit(1)

    # Extract the zip
    if zip_path.exists():
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)
        zip_path.unlink()  # Remove the zip after extraction
        print(f"Extracted to {DATA_DIR}/")
    else:
        # Sometimes kaggle downloads individual files directly
        print(f"Files downloaded to {DATA_DIR}/")

    # List what we got
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    print(f"\n{len(csv_files)} CSV files in data/:")
    for f in csv_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:45s} {size_mb:>7.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Download March Madness competition data from Kaggle"
    )
    parser.add_argument(
        "--year",
        type=int,
        nargs="+",
        default=[2026],
        help="Competition year(s) to download (default: 2026)",
    )
    args = parser.parse_args()

    for year in args.year:
        download_competition(year)

    print(f"\nDone! Data is ready in {DATA_DIR.resolve()}")


if __name__ == "__main__":
    main()

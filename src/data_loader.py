"""Load and merge raw Kaggle CSVs for the March Madness competition.

Combines men's and women's data into unified DataFrames with consistent
column naming. All raw CSVs stay untouched in data/.
"""

import subprocess
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

COMPETITION = "march-machine-learning-mania-2026"


def download_latest_data(data_dir: Optional[Path] = None) -> None:
    """Download the latest competition data from Kaggle and extract it.

    Requires the ``kaggle`` CLI to be installed and authenticated
    (``~/.kaggle/kaggle.json``).  Downloads the full dataset ZIP and
    extracts CSV files into *data_dir*, overwriting any existing files so
    the notebook always uses the freshest data.

    Args:
        data_dir: Target directory.  Defaults to ``data/``.
    """
    d = data_dir or DATA_DIR
    d.mkdir(parents=True, exist_ok=True)
    zip_path = d / f"{COMPETITION}.zip"

    print(f"Downloading latest data from Kaggle ({COMPETITION})...")
    result = subprocess.run(
        [
            "kaggle", "competitions", "download",
            "-c", COMPETITION,
            "-p", str(d),
        ],
        capture_output=True,
        text=True,
    )

    # Kaggle CLI writes progress bars to stderr, which is normal.
    # Only treat it as a real failure if the ZIP wasn't produced.
    if result.stdout:
        print(result.stdout.strip())

    if zip_path.exists():
        print("Extracting ZIP...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(d)
        zip_path.unlink()
        print(f"Done — {len(list(d.glob('*.csv')))} CSV files in {d}")
    elif result.returncode != 0:
        # Real failure — no ZIP and non-zero exit code
        err_msg = (result.stderr or "").strip()
        print(f"⚠️  Kaggle download failed (exit {result.returncode}): {err_msg}")
        print("Continuing with existing local data...")
    else:
        print("No ZIP found — files may already be up to date.")


def load_regular_season(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load and concatenate men's + women's regular season detailed results.

    Args:
        data_dir: Override path to data directory.

    Returns:
        Combined DataFrame with all regular season box scores (2003+ men, 2010+ women).
    """
    d = data_dir or DATA_DIR
    men = pd.read_csv(d / "MRegularSeasonDetailedResults.csv")
    women = pd.read_csv(d / "WRegularSeasonDetailedResults.csv")
    return pd.concat([men, women], ignore_index=True)


def load_tourney_results(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load and concatenate men's + women's NCAA tournament detailed results.

    Args:
        data_dir: Override path to data directory.

    Returns:
        Combined DataFrame with all tournament box scores.
    """
    d = data_dir or DATA_DIR
    men = pd.read_csv(d / "MNCAATourneyDetailedResults.csv")
    women = pd.read_csv(d / "WNCAATourneyDetailedResults.csv")
    return pd.concat([men, women], ignore_index=True)


def load_tourney_seeds(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load and concatenate men's + women's tournament seeds.

    Parses the seed string (e.g., 'W01a') into a numeric seed (1-16).

    Args:
        data_dir: Override path to data directory.

    Returns:
        DataFrame with Season, TeamID, Seed (string), seed (int 1-16).
    """
    d = data_dir or DATA_DIR
    men = pd.read_csv(d / "MNCAATourneySeeds.csv")
    women = pd.read_csv(d / "WNCAATourneySeeds.csv")
    seeds = pd.concat([men, women], ignore_index=True)
    seeds["seed"] = seeds["Seed"].apply(lambda x: int(x[1:3]))
    return seeds


def load_compact_results(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load men's + women's regular season compact results (scores only, 1985+).

    Args:
        data_dir: Override path to data directory.

    Returns:
        Combined compact results DataFrame.
    """
    d = data_dir or DATA_DIR
    men = pd.read_csv(d / "MRegularSeasonCompactResults.csv")
    women = pd.read_csv(d / "WRegularSeasonCompactResults.csv")
    return pd.concat([men, women], ignore_index=True)


def load_teams(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load men's + women's team ID-to-name mappings.

    Args:
        data_dir: Override path to data directory.

    Returns:
        Combined DataFrame with TeamID and TeamName.
    """
    d = data_dir or DATA_DIR
    men = pd.read_csv(d / "MTeams.csv")
    women = pd.read_csv(d / "WTeams.csv")
    return pd.concat([men, women], ignore_index=True)


def load_team_conferences(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load men's + women's team conference assignments per season.

    Args:
        data_dir: Override path to data directory.

    Returns:
        Combined DataFrame with Season, TeamID, ConfAbbrev.
    """
    d = data_dir or DATA_DIR
    men = pd.read_csv(d / "MTeamConferences.csv")
    women = pd.read_csv(d / "WTeamConferences.csv")
    return pd.concat([men, women], ignore_index=True)


def load_massey_ordinals(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load Massey Ordinals rankings (men's only — large file).

    Args:
        data_dir: Override path to data directory.

    Returns:
        DataFrame with Season, RankingDayNum, SystemName, TeamID, OrdinalRank.
    """
    d = data_dir or DATA_DIR
    return pd.read_csv(d / "MMasseyOrdinals.csv")


def load_coaches(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load men's coaching history.

    Args:
        data_dir: Override path to data directory.

    Returns:
        DataFrame with Season, TeamID, FirstDayNum, LastDayNum, CoachName.
    """
    d = data_dir or DATA_DIR
    return pd.read_csv(d / "MTeamCoaches.csv")


def load_sample_submission(
    stage: int = 1, data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Load a sample submission file.

    Args:
        stage: 1 for Stage 1 (historical validation), 2 for Stage 2 (current year).
        data_dir: Override path to data directory.

    Returns:
        DataFrame with ID and Pred columns.
    """
    d = data_dir or DATA_DIR
    return pd.read_csv(d / f"SampleSubmissionStage{stage}.csv")

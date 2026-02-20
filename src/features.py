"""Feature engineering for March Madness predictions.

Converts raw game-level data into per-season team statistics and matchup
features. Follows the convention of Team1ID < Team2ID and computes feature
*differences* (Team1 − Team2) for modeling.
"""

from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Data preparation — convert winner/loser format to symmetric Team1/Team2
# ---------------------------------------------------------------------------

def prepare_game_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert winner/loser format to symmetric Team1/Team2 rows.

    Each game produces TWO rows: one from each team's perspective.
    This lets us compute per-team averages by grouping on T1_TeamID.

    Args:
        df: Raw results DataFrame with W*/L* columns.

    Returns:
        DataFrame with T1_*/T2_* columns, a location encoding, and PointDiff.
    """
    # Create the swapped view (loser's perspective)
    dfswap = df[
        [
            "Season", "DayNum",
            "LTeamID", "LScore", "WTeamID", "WScore", "WLoc", "NumOT",
            "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR",
            "LAst", "LTO", "LStl", "LBlk", "LPF",
            "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR",
            "WAst", "WTO", "WStl", "WBlk", "WPF",
        ]
    ].copy()

    # Swap home/away for the loser's perspective
    dfswap.loc[df["WLoc"] == "H", "WLoc"] = "A"
    dfswap.loc[df["WLoc"] == "A", "WLoc"] = "H"

    df = df.copy()
    df.columns = df.columns.str.replace("WLoc", "location")
    dfswap.columns = dfswap.columns.str.replace("WLoc", "location")

    # Rename W→T1, L→T2
    df.columns = [c.replace("W", "T1_").replace("L", "T2_") for c in df.columns]
    dfswap.columns = [c.replace("L", "T1_").replace("W", "T2_") for c in df.columns]

    output = pd.concat([df, dfswap]).reset_index(drop=True)

    # Encode location as numeric
    output["location"] = output["location"].map({"N": 0, "H": 1, "A": -1}).fillna(0).astype(int)
    output["PointDiff"] = output["T1_Score"] - output["T2_Score"]

    return output


# ---------------------------------------------------------------------------
# Season-level team statistics
# ---------------------------------------------------------------------------

BOXSCORE_COLS = [
    "T1_FGM", "T1_FGA", "T1_FGM3", "T1_FGA3",
    "T1_OR", "T1_Ast", "T1_TO", "T1_Stl", "T1_PF",
    "T2_FGM", "T2_FGA", "T2_FGM3", "T2_FGA3",
    "T2_OR", "T2_Ast", "T2_TO", "T2_Stl", "T2_Blk",
    "PointDiff",
]


def compute_season_stats(regular_data: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team per-season average box score statistics.

    Args:
        regular_data: Output of prepare_game_data on regular season results.

    Returns:
        DataFrame indexed by (Season, T1_TeamID) with mean stats.
    """
    stats = (
        regular_data
        .groupby(["Season", "T1_TeamID"])[BOXSCORE_COLS]
        .mean()
        .reset_index()
    )
    stats.columns = ["".join(col).strip() for col in stats.columns]
    return stats


def compute_win_pct(regular_data: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team per-season win percentage.

    Args:
        regular_data: Output of prepare_game_data.

    Returns:
        DataFrame with Season, T1_TeamID, WinPct.
    """
    regular_data = regular_data.copy()
    regular_data["win"] = (regular_data["PointDiff"] > 0).astype(int)
    win_pct = (
        regular_data
        .groupby(["Season", "T1_TeamID"])["win"]
        .mean()
        .reset_index(name="WinPct")
    )
    return win_pct


def compute_efficiency(regular_data: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team per-season offensive & defensive efficiency.

    Efficiency = (Points / Possessions) * 100
    Possessions ≈ FGA - OR + TO + 0.475 * FTA

    Args:
        regular_data: Output of prepare_game_data.

    Returns:
        DataFrame with Season, T1_TeamID, OffEff, DefEff.
    """
    df = regular_data.copy()
    df["T1_Poss"] = df["T1_FGA"] - df["T1_OR"] + df["T1_TO"] + 0.475 * df["T1_FTA"]
    df["T2_Poss"] = df["T2_FGA"] - df["T2_OR"] + df["T2_TO"] + 0.475 * df["T2_FTA"]

    df["OffEff"] = (df["T1_Score"] / df["T1_Poss"]) * 100
    df["DefEff"] = (df["T2_Score"] / df["T2_Poss"]) * 100

    eff = (
        df.groupby(["Season", "T1_TeamID"])[["OffEff", "DefEff"]]
        .mean()
        .reset_index()
    )
    return eff


# ---------------------------------------------------------------------------
# Last-14-days momentum
# ---------------------------------------------------------------------------

def compute_last14_momentum(regular_data: pd.DataFrame, day_cutoff: int = 118) -> pd.DataFrame:
    """Compute win ratio in the last ~14 days of the regular season.

    Args:
        regular_data: Output of prepare_game_data.
        day_cutoff: DayNum threshold (games after this day).

    Returns:
        DataFrame with Season, T1_TeamID, win_ratio_14d.
    """
    late = regular_data.loc[regular_data["DayNum"] > day_cutoff].copy()
    late["win"] = (late["PointDiff"] > 0).astype(int)
    momentum = (
        late.groupby(["Season", "T1_TeamID"])["win"]
        .mean()
        .reset_index(name="win_ratio_14d")
    )
    return momentum


# ---------------------------------------------------------------------------
# GLM team quality metric
# ---------------------------------------------------------------------------

def compute_team_quality(
    regular_data: pd.DataFrame,
    seeds: pd.DataFrame,
    seasons: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Compute GLM-based team quality for tournament teams.

    Fits a logistic regression with team IDs as predictors on regular season
    win/loss outcomes, restricted to teams that appear in the tournament seeds.
    The resulting coefficients measure each team's quality.

    Args:
        regular_data: Output of prepare_game_data.
        seeds: Tournament seeds DataFrame with Season, TeamID columns.
        seasons: List of seasons to compute quality for. Defaults to all
                 available seasons in seeds (excluding 2020).

    Returns:
        DataFrame with TeamID, quality, Season columns.
    """
    # Filter to games involving tournament teams
    effects = regular_data[["Season", "T1_TeamID", "T2_TeamID", "PointDiff"]].copy()
    effects["T1_TeamID"] = effects["T1_TeamID"].astype(str)
    effects["T2_TeamID"] = effects["T2_TeamID"].astype(str)
    effects["win"] = (effects["PointDiff"] > 0).astype(int)

    # Build mart of all tournament team pairs
    march_madness = pd.merge(
        seeds[["Season", "TeamID"]],
        seeds[["Season", "TeamID"]],
        on="Season",
    )
    march_madness.columns = ["Season", "T1_TeamID", "T2_TeamID"]
    march_madness["T1_TeamID"] = march_madness["T1_TeamID"].astype(str)
    march_madness["T2_TeamID"] = march_madness["T2_TeamID"].astype(str)

    effects = pd.merge(effects, march_madness, on=["Season", "T1_TeamID", "T2_TeamID"])

    if seasons is None:
        seasons = sorted(effects["Season"].unique())
        seasons = [s for s in seasons if s != 2020]

    def _quality_for_season(season: int) -> pd.DataFrame:
        data = effects.loc[effects["Season"] == season]
        if len(data) == 0:
            return pd.DataFrame(columns=["TeamID", "quality", "Season"])
        try:
            glm = sm.GLM.from_formula(
                formula="win ~ -1 + T1_TeamID + T2_TeamID",
                data=data,
                family=sm.families.Binomial(),
            ).fit()
            quality = pd.DataFrame(glm.params).reset_index()
            quality.columns = ["TeamID", "quality"]
            quality["Season"] = season
            quality = quality.loc[quality["TeamID"].str.contains("T1_")].reset_index(drop=True)
            quality["TeamID"] = quality["TeamID"].apply(lambda x: x[10:14]).astype(int)
            return quality
        except Exception:
            return pd.DataFrame(columns=["TeamID", "quality", "Season"])

    return pd.concat([_quality_for_season(s) for s in seasons], ignore_index=True)


# ---------------------------------------------------------------------------
# Massey Ordinals feature
# ---------------------------------------------------------------------------

def compute_massey_features(
    massey_df: pd.DataFrame,
    top_systems: Optional[list[str]] = None,
    day_cutoff: int = 128,
) -> pd.DataFrame:
    """Extract end-of-season Massey Ordinal rankings for top systems.

    Args:
        massey_df: Raw MMasseyOrdinals.csv DataFrame.
        top_systems: List of system names to use. Defaults to top predictive
                     systems: POM, SAG, MOR, DOL, COL.
        day_cutoff: Use rankings from the latest day <= this cutoff.

    Returns:
        DataFrame with Season, TeamID, and one column per ranking system.
    """
    if top_systems is None:
        top_systems = ["POM", "SAG", "MOR", "DOL", "COL"]

    df = massey_df[massey_df["SystemName"].isin(top_systems)].copy()
    df = df[df["RankingDayNum"] <= day_cutoff]

    # Take the latest ranking per system per team per season
    latest = (
        df.groupby(["Season", "TeamID", "SystemName"])["RankingDayNum"]
        .max()
        .reset_index()
    )
    df = pd.merge(df, latest, on=["Season", "TeamID", "SystemName", "RankingDayNum"])

    # Pivot to wide format
    pivot = df.pivot_table(
        index=["Season", "TeamID"],
        columns="SystemName",
        values="OrdinalRank",
        aggfunc="mean",
    ).reset_index()

    pivot.columns.name = None
    # Rename system columns with prefix
    rename = {sys: f"Massey_{sys}" for sys in top_systems if sys in pivot.columns}
    pivot = pivot.rename(columns=rename)

    return pivot


# ---------------------------------------------------------------------------
# Build full feature set for matchups
# ---------------------------------------------------------------------------

def build_matchup_features(
    season_stats: pd.DataFrame,
    win_pct: pd.DataFrame,
    efficiency: pd.DataFrame,
    momentum: pd.DataFrame,
    seeds: pd.DataFrame,
    quality: Optional[pd.DataFrame] = None,
    massey: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge all per-team features into T1/T2 format for matchup merging.

    Call this function to produce the feature lookup tables that get merged
    onto tournament games or submission matchup pairs.

    Args:
        season_stats: Output of compute_season_stats.
        win_pct: Output of compute_win_pct.
        efficiency: Output of compute_efficiency.
        momentum: Output of compute_last14_momentum.
        seeds: Tournament seeds with Season, TeamID, seed columns.
        quality: Optional output of compute_team_quality.
        massey: Optional output of compute_massey_features.

    Returns:
        Tuple of (features_T1, features_T2) DataFrames ready for merge.
    """
    # Start with season stats
    features = season_stats.copy()

    # Merge additional per-team features
    features = pd.merge(features, win_pct, on=["Season", "T1_TeamID"], how="left")
    features = pd.merge(features, efficiency, on=["Season", "T1_TeamID"], how="left")
    features = pd.merge(features, momentum, on=["Season", "T1_TeamID"], how="left")

    # Seeds
    seed_cols = seeds[["Season", "TeamID", "seed"]].copy()
    seed_cols.columns = ["Season", "T1_TeamID", "T1_seed"]
    features = pd.merge(features, seed_cols, on=["Season", "T1_TeamID"], how="left")

    # Quality
    if quality is not None:
        qual = quality.copy()
        qual.columns = ["T1_TeamID", "T1_quality", "Season"]
        features = pd.merge(features, qual, on=["Season", "T1_TeamID"], how="left")

    # Massey
    if massey is not None:
        massey_t1 = massey.copy()
        massey_t1 = massey_t1.rename(columns={"TeamID": "T1_TeamID"})
        # Prefix massey columns with T1_
        massey_cols = [c for c in massey_t1.columns if c.startswith("Massey_")]
        massey_t1 = massey_t1.rename(columns={c: f"T1_{c}" for c in massey_cols})
        features = pd.merge(features, massey_t1, on=["Season", "T1_TeamID"], how="left")

    # Create T2 version
    features_T1 = features.copy()
    features_T2 = features.copy()

    # Rename for T2
    t2_rename = {}
    for col in features_T2.columns:
        if col in ("Season",):
            continue
        new_col = col.replace("T1_", "T2_").replace("T2_opponent_", "T1_opponent_")
        if not col.startswith("T2_") and col != "Season":
            new_col = "T2_" + col.lstrip("T1_") if col.startswith("T1_") else "T2_" + col
        t2_rename[col] = new_col

    features_T2.columns = ["Season"] + [
        c.replace("T1_", "T2_") for c in features_T2.columns if c != "Season"
    ]

    return features_T1, features_T2


def create_matchup_df(
    matchup_df: pd.DataFrame,
    features_T1: pd.DataFrame,
    features_T2: pd.DataFrame,
) -> pd.DataFrame:
    """Merge T1 and T2 features onto a matchup DataFrame.

    Args:
        matchup_df: DataFrame with Season, T1_TeamID, T2_TeamID columns.
        features_T1: T1 feature table from build_matchup_features.
        features_T2: T2 feature table from build_matchup_features.

    Returns:
        Enriched matchup DataFrame with all features merged.
    """
    result = matchup_df.copy()
    result = pd.merge(result, features_T1, on=["Season", "T1_TeamID"], how="left")
    result = pd.merge(result, features_T2, on=["Season", "T2_TeamID"], how="left")

    # Add seed difference
    if "T1_seed" in result.columns and "T2_seed" in result.columns:
        result["Seed_diff"] = result["T1_seed"] - result["T2_seed"]

    # Add quality difference
    if "T1_quality" in result.columns and "T2_quality" in result.columns:
        result["Quality_diff"] = result["T1_quality"] - result["T2_quality"]

    return result

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

# Team's own stats (from T1 perspective in symmetric data)
TEAM_STAT_COLS = [
    "T1_Score", "T1_FGM", "T1_FGA", "T1_FGM3", "T1_FGA3",
    "T1_FTM", "T1_FTA", "T1_OR", "T1_DR", "T1_Ast",
    "T1_TO", "T1_Stl", "T1_Blk", "T1_PF",
]

# Opponent stats (from T2 perspective)
OPP_STAT_COLS = [
    "T2_Score", "T2_FGM", "T2_FGA", "T2_FGM3", "T2_FGA3",
    "T2_FTM", "T2_FTA", "T2_OR", "T2_DR", "T2_Ast",
    "T2_TO", "T2_Stl", "T2_Blk", "T2_PF",
]


def compute_season_stats(regular_data: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team per-season average box score statistics.

    Averages the team's own stats AND opponent stats separately.
    Opponent columns are renamed with 'Opp_' prefix to avoid confusion.

    Args:
        regular_data: Output of prepare_game_data on regular season results.

    Returns:
        DataFrame with Season, TeamID, and per-game averages for
        the team's stats (FGM, FGA, ...) and opponent stats (Opp_FGM, ...).
    """
    agg_cols = TEAM_STAT_COLS + OPP_STAT_COLS + ["PointDiff"]
    # Only use columns that exist in the data
    agg_cols = [c for c in agg_cols if c in regular_data.columns]

    stats = (
        regular_data
        .groupby(["Season", "T1_TeamID"])[agg_cols]
        .mean()
        .reset_index()
    )

    # Clean column names: T1_Score -> Score, T2_Score -> Opp_Score, etc.
    rename = {"T1_TeamID": "TeamID"}
    for col in stats.columns:
        if col.startswith("T1_"):
            rename[col] = col[3:]  # T1_FGM -> FGM
        elif col.startswith("T2_"):
            rename[col] = "Opp_" + col[3:]  # T2_FGM -> Opp_FGM
    stats = stats.rename(columns=rename)

    return stats


def compute_shooting_pcts(season_stats: pd.DataFrame) -> pd.DataFrame:
    """Compute shooting percentage features from season averages.

    Derived from per-game average makes & attempts already in season_stats.
    These ratio features capture *efficiency* independent of pace.

    Args:
        season_stats: Output of compute_season_stats (per-game averages).

    Returns:
        DataFrame with Season, TeamID, FGPct, FG3Pct, FTPct.
    """
    df = season_stats[["Season", "TeamID"]].copy()

    # Team shooting percentages
    if "FGA" in season_stats.columns and "FGM" in season_stats.columns:
        df["FGPct"] = season_stats["FGM"] / season_stats["FGA"].replace(0, np.nan)
    if "FGA3" in season_stats.columns and "FGM3" in season_stats.columns:
        df["FG3Pct"] = season_stats["FGM3"] / season_stats["FGA3"].replace(0, np.nan)
    if "FTA" in season_stats.columns and "FTM" in season_stats.columns:
        df["FTPct"] = season_stats["FTM"] / season_stats["FTA"].replace(0, np.nan)

    return df


def compute_sos(
    regular_data: pd.DataFrame,
    win_pct: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Strength of Schedule as average opponent win percentage.

    A team that beats 20-win teams is stronger than one beating 5-win teams,
    even if both have the same record. SOS captures this.

    Args:
        regular_data: Output of prepare_game_data.
        win_pct: Output of compute_win_pct (Season, TeamID, WinPct).

    Returns:
        DataFrame with Season, TeamID, SOS.
    """
    df = regular_data[["Season", "T1_TeamID", "T2_TeamID"]].copy()
    # Look up each opponent's win percentage
    df = pd.merge(
        df,
        win_pct.rename(columns={"TeamID": "T2_TeamID", "WinPct": "Opp_WinPct"}),
        on=["Season", "T2_TeamID"],
        how="left",
    )
    # Average opponent win percentage = strength of schedule
    sos = (
        df.groupby(["Season", "T1_TeamID"])["Opp_WinPct"]
        .mean()
        .reset_index(name="SOS")
    )
    sos = sos.rename(columns={"T1_TeamID": "TeamID"})
    return sos


def compute_win_pct(regular_data: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team per-season win percentage.

    Args:
        regular_data: Output of prepare_game_data.

    Returns:
        DataFrame with Season, TeamID, WinPct.
    """
    regular_data = regular_data.copy()
    regular_data["win"] = (regular_data["PointDiff"] > 0).astype(int)
    win_pct = (
        regular_data
        .groupby(["Season", "T1_TeamID"])["win"]
        .mean()
        .reset_index(name="WinPct")
    )
    win_pct = win_pct.rename(columns={"T1_TeamID": "TeamID"})
    return win_pct


def compute_efficiency(regular_data: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team per-season offensive & defensive efficiency.

    Efficiency = (Points / Possessions) * 100
    Possessions ≈ FGA - OR + TO + 0.475 * FTA

    Args:
        regular_data: Output of prepare_game_data.

    Returns:
        DataFrame with Season, TeamID, OffEff, DefEff.
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
    eff = eff.rename(columns={"T1_TeamID": "TeamID"})
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
        DataFrame with Season, TeamID, win_ratio_14d.
    """
    late = regular_data.loc[regular_data["DayNum"] > day_cutoff].copy()
    late["win"] = (late["PointDiff"] > 0).astype(int)
    momentum = (
        late.groupby(["Season", "T1_TeamID"])["win"]
        .mean()
        .reset_index(name="win_ratio_14d")
    )
    momentum = momentum.rename(columns={"T1_TeamID": "TeamID"})
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
    win/loss outcomes, restricted to games between tournament teams.
    The resulting coefficients measure each team's quality.

    Args:
        regular_data: Output of prepare_game_data.
        seeds: Tournament seeds DataFrame with Season, TeamID columns.
        seasons: List of seasons to compute quality for. Defaults to all
                 available seasons in seeds (excluding 2020).

    Returns:
        DataFrame with Season, TeamID, quality columns.
    """
    effects = regular_data[["Season", "T1_TeamID", "T2_TeamID", "PointDiff"]].copy()
    effects["T1_TeamID"] = effects["T1_TeamID"].astype(str)
    effects["T2_TeamID"] = effects["T2_TeamID"].astype(str)
    effects["win"] = (effects["PointDiff"] > 0).astype(int)

    # Keep only games where BOTH teams made the tournament
    tourney_teams = seeds[["Season", "TeamID"]].copy()
    tourney_teams["TeamID"] = tourney_teams["TeamID"].astype(str)

    effects = pd.merge(
        effects,
        tourney_teams.rename(columns={"TeamID": "T1_TeamID"}),
        on=["Season", "T1_TeamID"],
    )
    effects = pd.merge(
        effects,
        tourney_teams.rename(columns={"TeamID": "T2_TeamID"}),
        on=["Season", "T2_TeamID"],
    )

    if seasons is None:
        seasons = sorted(effects["Season"].unique())
        seasons = [s for s in seasons if s != 2020]

    def _quality_for_season(season: int) -> pd.DataFrame:
        data = effects.loc[effects["Season"] == season]
        if len(data) == 0:
            return pd.DataFrame(columns=["Season", "TeamID", "quality"])
        try:
            # Use regularized GLM (L2 penalty prevents coefficient explosion)
            glm = sm.GLM.from_formula(
                formula="win ~ -1 + T1_TeamID + T2_TeamID",
                data=data,
                family=sm.families.Binomial(),
            ).fit_regularized(alpha=0.1, L1_wt=0.0, maxiter=100)
            quality = pd.DataFrame({"param": glm.params.index, "quality": glm.params.values})
            quality["quality"] = quality["quality"].astype(float)
            quality = quality.loc[quality["param"].str.startswith("T1_TeamID")]
            # Extract team ID — handles both [1103] and [T.1103] formats
            quality["TeamID"] = (
                quality["param"]
                .str.extract(r"T1_TeamID\[(?:T\.)?(\d+)\]")[0]
                .astype(int)
            )
            quality["Season"] = season
            # Normalize to z-scores so quality is comparable across seasons
            q = quality["quality"]
            if q.std() > 0:
                quality["quality"] = (q - q.mean()) / q.std()
            return quality[["Season", "TeamID", "quality"]].reset_index(drop=True)
        except Exception:
            return pd.DataFrame(columns=["Season", "TeamID", "quality"])

    return pd.concat([_quality_for_season(s) for s in seasons], ignore_index=True)


# ---------------------------------------------------------------------------
# Massey Ordinals feature
# ---------------------------------------------------------------------------

def compute_elo_ratings(
    compact_results: pd.DataFrame,
    K: float = 20.0,
    home_advantage: float = 100.0,
    initial_elo: float = 1500.0,
    carry_over: float = 0.75,
    min_season: int = 2003,
) -> pd.DataFrame:
    """Compute end-of-regular-season Elo ratings for all teams.

    Elo is updated game-by-game in chronological order. Unlike season
    averages, it captures *when* wins/losses happened and weighs recent
    performance more heavily. Ratings carry over between seasons (regressed
    toward the mean) so teams with sustained excellence get credit.

    Args:
        compact_results: Men's + women's compact results (Season, DayNum,
            WTeamID, WScore, LTeamID, LScore, WLoc).
        K: Elo K-factor — how much each game moves the rating.
        home_advantage: Elo bonus for the home team.
        initial_elo: Starting rating for new teams.
        carry_over: Fraction of prior season's Elo retained (rest regresses
            to initial_elo).
        min_season: Earliest season to return ratings for (earlier seasons
            are used to warm up Elo but not returned).

    Returns:
        DataFrame with Season, TeamID, Elo.
    """
    df = compact_results.sort_values(["Season", "DayNum"]).reset_index(drop=True)

    # Track current Elo for each team
    elo: dict[int, float] = {}
    # Store end-of-regular-season snapshot per (Season, TeamID)
    records: list[dict] = []

    prev_season = None

    for _, row in df.iterrows():
        season = row["Season"]

        # New season: carry over + regress toward mean
        if season != prev_season:
            # Snapshot end-of-last-season ratings
            if prev_season is not None and prev_season >= min_season:
                for tid, rating in elo.items():
                    records.append(
                        {"Season": prev_season, "TeamID": tid, "Elo": rating}
                    )
            # Regress all Elo toward initial_elo
            for tid in elo:
                elo[tid] = initial_elo + carry_over * (elo[tid] - initial_elo)
            prev_season = season

        w_id = int(row["WTeamID"])
        l_id = int(row["LTeamID"])

        # Initialize new teams
        if w_id not in elo:
            elo[w_id] = initial_elo
        if l_id not in elo:
            elo[l_id] = initial_elo

        # Home advantage
        w_elo = elo[w_id]
        l_elo = elo[l_id]
        loc = row.get("WLoc", "N")
        if loc == "H":
            w_elo += home_advantage
        elif loc == "A":
            l_elo += home_advantage

        # Expected scores
        e_w = 1.0 / (1.0 + 10.0 ** ((l_elo - w_elo) / 400.0))

        # Margin-of-victory multiplier (diminishing returns for blowouts)
        mov = abs(row["WScore"] - row["LScore"])
        mov_mult = np.log1p(mov) * (2.2 / ((elo[w_id] - elo[l_id]) * 0.001 + 2.2))
        mov_mult = max(mov_mult, 0.5)  # floor

        # Update
        elo[w_id] += K * mov_mult * (1.0 - e_w)
        elo[l_id] -= K * mov_mult * (1.0 - e_w)

    # Final season snapshot
    if prev_season is not None and prev_season >= min_season:
        for tid, rating in elo.items():
            records.append({"Season": prev_season, "TeamID": tid, "Elo": rating})

    result = pd.DataFrame(records)
    # Normalize per season to z-scores for comparability
    if len(result) > 0:
        result["Elo"] = result.groupby("Season")["Elo"].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
        )
    return result


def compute_coach_experience(
    coaches_df: pd.DataFrame,
    tourney_seeds: pd.DataFrame,
) -> pd.DataFrame:
    """Compute coach NCAA tournament experience (men only).

    Counts how many prior NCAA tournament appearances a team's current
    coach has made (across any team). Experienced tournament coaches
    consistently outperform their seeds.

    Args:
        coaches_df: MTeamCoaches.csv with Season, TeamID, CoachName,
            FirstDayNum, LastDayNum.
        tourney_seeds: Tournament seeds with Season and TeamID.

    Returns:
        DataFrame with Season, TeamID, CoachTourneyExp (count of prior
        tournament appearances for the coach).
    """
    # Identify which coach was leading the team at the end of the season
    # (LastDayNum == 154 typically corresponds to end of season / tournament)
    end_coaches = (
        coaches_df
        .sort_values(["Season", "TeamID", "LastDayNum"])
        .groupby(["Season", "TeamID"])
        .last()
        .reset_index()[["Season", "TeamID", "CoachName"]]
    )

    # Which of those coaches' teams made the tournament?
    tourney_teams = tourney_seeds[["Season", "TeamID"]].drop_duplicates()
    coach_tourney = pd.merge(end_coaches, tourney_teams, on=["Season", "TeamID"])

    # For each (Season, Coach), count prior tournament appearances
    # Sort by Season so cumcount gives the count of PRIOR appearances
    coach_tourney = coach_tourney.sort_values("Season")
    coach_tourney["CoachTourneyExp"] = (
        coach_tourney.groupby("CoachName").cumcount()
    )  # 0-indexed: first appearance = 0 prior appearances

    # Map back to ALL teams (not just tournament teams — some teams have
    # coaches with tournament experience from other schools)
    coach_exp = pd.merge(
        end_coaches,
        coach_tourney[["Season", "CoachName", "CoachTourneyExp"]].drop_duplicates(),
        on=["Season", "CoachName"],
        how="left",
    )
    # Coaches with no prior tournament experience get 0
    coach_exp["CoachTourneyExp"] = coach_exp["CoachTourneyExp"].fillna(0).astype(int)

    return coach_exp[["Season", "TeamID", "CoachTourneyExp"]]


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

def build_team_features(
    season_stats: pd.DataFrame,
    win_pct: pd.DataFrame,
    efficiency: pd.DataFrame,
    momentum: pd.DataFrame,
    seeds: pd.DataFrame,
    quality: Optional[pd.DataFrame] = None,
    massey: Optional[pd.DataFrame] = None,
    shooting: Optional[pd.DataFrame] = None,
    sos: Optional[pd.DataFrame] = None,
    elo: Optional[pd.DataFrame] = None,
    coach: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge all per-team features into a single lookup table.

    All inputs should be keyed on (Season, TeamID). The result is one row
    per team per season with all features consolidated.

    Args:
        season_stats: Output of compute_season_stats.
        win_pct: Output of compute_win_pct.
        efficiency: Output of compute_efficiency.
        momentum: Output of compute_last14_momentum.
        seeds: Tournament seeds with Season, TeamID, seed columns.
        quality: Optional output of compute_team_quality.
        massey: Optional output of compute_massey_features.
        shooting: Optional output of compute_shooting_pcts.
        sos: Optional output of compute_sos.
        elo: Optional output of compute_elo_ratings.
        coach: Optional output of compute_coach_experience.

    Returns:
        DataFrame keyed on (Season, TeamID) with all features.
    """
    features = season_stats.copy()

    # All these DataFrames are keyed on (Season, TeamID)
    features = pd.merge(features, win_pct, on=["Season", "TeamID"], how="left")
    features = pd.merge(features, efficiency, on=["Season", "TeamID"], how="left")
    features = pd.merge(features, momentum, on=["Season", "TeamID"], how="left")

    # Seeds
    seed_cols = seeds[["Season", "TeamID", "seed"]].drop_duplicates()
    features = pd.merge(features, seed_cols, on=["Season", "TeamID"], how="left")

    # Quality
    if quality is not None:
        features = pd.merge(
            features, quality[["Season", "TeamID", "quality"]],
            on=["Season", "TeamID"], how="left",
        )

    # Massey
    if massey is not None:
        features = pd.merge(features, massey, on=["Season", "TeamID"], how="left")

    # Shooting percentages
    if shooting is not None:
        features = pd.merge(features, shooting, on=["Season", "TeamID"], how="left")

    # Strength of schedule
    if sos is not None:
        features = pd.merge(features, sos, on=["Season", "TeamID"], how="left")

    # Elo ratings
    if elo is not None:
        features = pd.merge(features, elo, on=["Season", "TeamID"], how="left")

    # Coach tournament experience
    if coach is not None:
        features = pd.merge(features, coach, on=["Season", "TeamID"], how="left")

    return features


def create_matchup_df(
    matchup_df: pd.DataFrame,
    team_features: pd.DataFrame,
) -> pd.DataFrame:
    """Merge per-team features onto a matchup DataFrame for both teams.

    Joins team_features twice — once for T1 (Team1) and once for T2 (Team2)
    — prefixing columns with 'T1_' and 'T2_' respectively.

    Args:
        matchup_df: DataFrame with Season, T1_TeamID, T2_TeamID columns.
        team_features: Output of build_team_features.

    Returns:
        Matchup DataFrame with T1_* and T2_* feature columns.
    """
    result = matchup_df.copy()

    # Prepare T1 features: rename TeamID -> T1_TeamID, prefix stat cols with T1_
    feat_cols = [c for c in team_features.columns if c not in ("Season", "TeamID")]
    t1 = team_features.rename(
        columns={"TeamID": "T1_TeamID", **{c: f"T1_{c}" for c in feat_cols}}
    )
    t2 = team_features.rename(
        columns={"TeamID": "T2_TeamID", **{c: f"T2_{c}" for c in feat_cols}}
    )

    result = pd.merge(result, t1, on=["Season", "T1_TeamID"], how="left")
    result = pd.merge(result, t2, on=["Season", "T2_TeamID"], how="left")

    return result


def compute_difference_features(
    matchup_df: pd.DataFrame,
    exclude_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute Team1 - Team2 difference for all paired feature columns.

    Finds all columns with a 'T1_' prefix, looks for the matching 'T2_'
    column, and creates a 'Diff_' column = T1 - T2. Opponent stats (Opp_*)
    are excluded by default since they already represent the opponent.

    Args:
        matchup_df: Output of create_matchup_df.
        exclude_cols: Additional column suffixes to exclude from differencing.

    Returns:
        Tuple of (enriched DataFrame, list of difference feature names).
    """
    if exclude_cols is None:
        exclude_cols = []

    result = matchup_df.copy()
    diff_features = []

    t1_cols = [c for c in result.columns if c.startswith("T1_") and c != "T1_TeamID"]
    for t1_col in t1_cols:
        suffix = t1_col[3:]  # e.g., "FGM", "Opp_FGM", "seed", ...

        # Skip opponent stats — they'd create confusing double-negatives
        if suffix.startswith("Opp_"):
            continue
        if suffix in exclude_cols:
            continue

        t2_col = f"T2_{suffix}"
        if t2_col in result.columns:
            diff_col = f"Diff_{suffix}"
            result[diff_col] = result[t1_col] - result[t2_col]
            diff_features.append(diff_col)

    return result, diff_features

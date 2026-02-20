"""Utility functions for the March Madness pipeline.

Team name lookups, submission generation, plotting helpers,
and other shared utilities.
"""

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from . import data_loader


# ---------------------------------------------------------------------------
# Team name lookup
# ---------------------------------------------------------------------------

_TEAMS_CACHE: Optional[pd.DataFrame] = None


def get_teams() -> pd.DataFrame:
    """Load and cache team name mappings."""
    global _TEAMS_CACHE
    if _TEAMS_CACHE is None:
        _TEAMS_CACHE = data_loader.load_teams()
    return _TEAMS_CACHE


def team_name(team_id: int) -> str:
    """Look up a team's name by ID.

    Args:
        team_id: Numeric team ID (1xxx for men, 3xxx for women).

    Returns:
        Team name string, or "Unknown (ID)" if not found.
    """
    teams = get_teams()
    match = teams.loc[teams["TeamID"] == team_id, "TeamName"]
    if len(match) > 0:
        return match.values[0]
    return f"Unknown ({team_id})"


def team_id(name: str) -> Optional[int]:
    """Look up a team's ID by name (case-insensitive partial match).

    Args:
        name: Team name or partial name.

    Returns:
        TeamID or None if not found.
    """
    teams = get_teams()
    mask = teams["TeamName"].str.contains(name, case=False, na=False)
    if mask.sum() == 1:
        return teams.loc[mask, "TeamID"].values[0]
    if mask.sum() > 1:
        print(f"Multiple matches for '{name}':")
        print(teams.loc[mask, ["TeamID", "TeamName"]].to_string(index=False))
        return None
    return None


# ---------------------------------------------------------------------------
# Submission generation
# ---------------------------------------------------------------------------

def parse_submission_ids(submission: pd.DataFrame) -> pd.DataFrame:
    """Parse submission ID column into Season, T1_TeamID, T2_TeamID.

    Args:
        submission: DataFrame with ID column in format "Season_Team1_Team2".

    Returns:
        DataFrame with parsed columns added.
    """
    sub = submission.copy()
    parts = sub["ID"].str.split("_", expand=True)
    sub["Season"] = parts[0].astype(int)
    sub["T1_TeamID"] = parts[1].astype(int)
    sub["T2_TeamID"] = parts[2].astype(int)
    return sub


def generate_submission(
    predictions: np.ndarray,
    sample_sub: pd.DataFrame,
    output_path: str,
    clip_range: tuple[float, float] = (0.025, 0.975),
) -> pd.DataFrame:
    """Generate a Kaggle submission CSV.

    Args:
        predictions: Array of predicted probabilities (P(Team1 wins)).
        sample_sub: Sample submission DataFrame with ID column.
        output_path: Path to save the submission CSV.
        clip_range: Min/max probability bounds.

    Returns:
        Final submission DataFrame.
    """
    sub = sample_sub[["ID"]].copy()
    sub["Pred"] = np.clip(predictions, clip_range[0], clip_range[1])
    sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} ({len(sub)} rows)")
    print(f"Pred range: [{sub['Pred'].min():.4f}, {sub['Pred'].max():.4f}]")
    print(f"Pred mean: {sub['Pred'].mean():.4f}")
    return sub


# ---------------------------------------------------------------------------
# Matchup explorer
# ---------------------------------------------------------------------------

def lookup_matchup(
    submission: pd.DataFrame,
    team1: str,
    team2: str,
) -> Optional[pd.Series]:
    """Look up a specific matchup prediction from a submission.

    Args:
        submission: Submission DataFrame with ID and Pred columns.
        team1: First team name (partial match OK).
        team2: Second team name (partial match OK).

    Returns:
        Series with matchup details, or None if not found.
    """
    id1 = team_id(team1)
    id2 = team_id(team2)
    if id1 is None or id2 is None:
        return None

    # Ensure Team1ID < Team2ID
    low, high = min(id1, id2), max(id1, id2)
    flipped = low != id1

    mask = submission["ID"].str.contains(f"_{low}_{high}")
    if mask.sum() == 0:
        print(f"No matchup found for {team_name(id1)} vs {team_name(id2)}")
        return None

    row = submission.loc[mask].iloc[0]
    pred = row["Pred"]

    t1_name = team_name(low)
    t2_name = team_name(high)

    print(f"\n{'='*50}")
    print(f"{t1_name} vs {t2_name}")
    print(f"P({t1_name} wins) = {pred:.4f}")
    print(f"P({t2_name} wins) = {1 - pred:.4f}")
    winner = t1_name if pred > 0.5 else t2_name
    confidence = max(pred, 1 - pred)
    print(f"Predicted winner: {winner} ({confidence:.1%} confidence)")
    print(f"{'='*50}")

    return row


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_feature_importance(model, feature_names: list[str], top_n: int = 20) -> None:
    """Plot feature importance from an XGBoost model.

    Args:
        model: Trained XGBoost Booster.
        feature_names: List of feature names.
        top_n: Number of top features to show.
    """
    importance = model.get_score(importance_type="gain")
    imp_df = pd.DataFrame(
        [(feature_names[int(k[1:])], v) for k, v in importance.items()],
        columns=["Feature", "Importance"],
    )
    imp_df = imp_df.sort_values("Importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax)
    ax.set_title(f"Top {top_n} Feature Importance (Gain)")
    plt.tight_layout()
    plt.show()


def plot_calibration(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 20) -> None:
    """Plot a calibration curve.

    Args:
        y_true: True binary outcomes.
        y_pred: Predicted probabilities.
        n_bins: Number of bins.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = []
    bin_true = []

    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() > 0:
            bin_means.append(y_pred[mask].mean())
            bin_true.append(y_true[mask].mean())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(bin_means, bin_true, "o-", label="Model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration Plot")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Season filtering
# ---------------------------------------------------------------------------

SKIP_SEASONS = {2020}  # COVID — no tournament


def valid_seasons(min_season: int = 2010, max_season: int = 2099) -> list[int]:
    """Return list of valid training seasons.

    Args:
        min_season: Earliest season to include (2010 for detailed data).
        max_season: Latest season to include.

    Returns:
        Sorted list of valid season years.
    """
    return sorted(s for s in range(min_season, max_season + 1) if s not in SKIP_SEASONS)

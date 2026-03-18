"""Manage actual tournament results and compare against model predictions.

Provides utilities to:
- Load/save actual game results from a JSON file
- Compare model predictions to actual outcomes
- Calculate running performance metrics (MSE, accuracy, log loss)
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_FILE = DATA_DIR / "actual_results_2026.json"

ROUND_ORDER = {
    "Play-In": 0,
    "Round of 64": 1,
    "Round of 32": 2,
    "Sweet 16": 3,
    "Elite 8": 4,
    "Final Four": 5,
    "Championship": 6,
}


def _round_name(slot: str) -> str:
    """Map a bracket slot to a human-readable round name."""
    if not slot.startswith("R"):
        return "Play-In"
    mapping = {
        "R1": "Round of 64",
        "R2": "Round of 32",
        "R3": "Sweet 16",
        "R4": "Elite 8",
        "R5": "Final Four",
        "R6": "Championship",
    }
    return mapping.get(slot[:2], slot[:2])


def load_results() -> dict:
    """Load the actual results JSON file."""
    if not RESULTS_FILE.exists():
        return {
            "description": "Actual 2026 NCAA tournament results.",
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "men": {},
            "women": {},
        }
    with open(RESULTS_FILE) as f:
        return json.load(f)


def save_results(data: dict) -> None:
    """Save results to the JSON file."""
    data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def add_result(
    gender: str,
    slot: str,
    winner_id: int,
    winner_name: str,
    loser_id: int,
    loser_name: str,
    winner_score: int,
    loser_score: int,
) -> dict:
    """Add a single game result.

    Args:
        gender: 'M' or 'W'.
        slot: Bracket slot (e.g., 'Y16', 'R1W1').
        winner_id: TeamID of the winner.
        winner_name: Display name of winner.
        loser_id: TeamID of the loser.
        loser_name: Display name of loser.
        winner_score: Winner's final score.
        loser_score: Loser's final score.

    Returns:
        Updated results dict.
    """
    data = load_results()
    key = "men" if gender == "M" else "women"
    data[key][slot] = {
        "winner_id": winner_id,
        "winner_name": winner_name,
        "loser_id": loser_id,
        "loser_name": loser_name,
        "score": f"{winner_score}-{loser_score}",
        "round": _round_name(slot),
    }
    save_results(data)
    return data


def remove_result(gender: str, slot: str) -> dict:
    """Remove a result by slot."""
    data = load_results()
    key = "men" if gender == "M" else "women"
    data[key].pop(slot, None)
    save_results(data)
    return data


def get_prediction_for_matchup(
    submission_path: Path,
    season: int,
    team1_id: int,
    team2_id: int,
) -> Optional[float]:
    """Look up our model's predicted P(team1 wins) from a submission CSV.

    Args:
        submission_path: Path to submission CSV.
        season: Tournament season.
        team1_id: First team ID (will be sorted).
        team2_id: Second team ID.

    Returns:
        Predicted probability that the lower-ID team wins, or None.
    """
    lo, hi = min(team1_id, team2_id), max(team1_id, team2_id)
    target_id = f"{season}_{lo}_{hi}"
    try:
        sub = pd.read_csv(submission_path)
        row = sub[sub["ID"] == target_id]
        if len(row) > 0:
            return float(row.iloc[0]["Pred"])
    except Exception:
        pass
    return None


def build_performance_table(
    gender: str,
    submission_path: Path,
    season: int = 2026,
    teams_lookup: Optional[dict] = None,
) -> pd.DataFrame:
    """Build a DataFrame comparing predictions to actual results.

    Returns DataFrame with columns:
        Round, Slot, Winner, Loser, Score, Predicted, Actual, Correct, MSE_Contrib
    """
    data = load_results()
    key = "men" if gender == "M" else "women"
    results = data.get(key, {})

    if not results:
        return pd.DataFrame()

    rows = []
    for slot, info in sorted(results.items(),
                              key=lambda x: ROUND_ORDER.get(
                                  _round_name(x[0]), 99)):
        winner_id = info["winner_id"]
        loser_id = info.get("loser_id")
        winner_name = info.get("winner_name", str(winner_id))
        loser_name = info.get("loser_name", str(loser_id))
        score = info.get("score", "")
        round_name = info.get("round", _round_name(slot))

        # Get our prediction
        pred = None
        if loser_id is not None:
            pred = get_prediction_for_matchup(
                submission_path, season, winner_id, loser_id
            )

        # Determine if we predicted correctly
        actual_outcome = 1  # winner won
        if pred is not None:
            lo = min(winner_id, loser_id)
            # pred is P(lo wins). If lo == winner_id, actual=1, else actual=0
            if lo == winner_id:
                pred_winner_prob = pred
            else:
                pred_winner_prob = 1.0 - pred
            correct = pred_winner_prob > 0.5
            mse_contrib = (1.0 - pred_winner_prob) ** 2
        else:
            pred_winner_prob = None
            correct = None
            mse_contrib = None

        rows.append({
            "Round": round_name,
            "Slot": slot,
            "Winner": winner_name,
            "Loser": loser_name,
            "Score": score,
            "P(Winner)": pred_winner_prob,
            "Correct": correct,
            "MSE": mse_contrib,
        })

    return pd.DataFrame(rows)


def compute_metrics(perf_df: pd.DataFrame) -> dict:
    """Compute aggregate performance metrics from the performance table.

    Returns dict with keys: n_games, n_correct, accuracy, mse, log_loss.
    """
    if perf_df.empty:
        return {}

    valid = perf_df.dropna(subset=["P(Winner)", "Correct"])
    if valid.empty:
        return {"n_games": len(perf_df), "n_with_preds": 0}

    n = len(valid)
    n_correct = int(valid["Correct"].sum())
    mse = float(valid["MSE"].mean())

    # Log loss
    eps = 1e-15
    ll = -sum(
        math.log(max(min(p, 1 - eps), eps))
        for p in valid["P(Winner)"]
    ) / n

    return {
        "n_games": len(perf_df),
        "n_with_preds": n,
        "n_correct": n_correct,
        "accuracy": n_correct / n if n > 0 else 0,
        "mse": mse,
        "log_loss": ll,
    }


def compute_round_metrics(perf_df: pd.DataFrame) -> pd.DataFrame:
    """Break down metrics by round."""
    if perf_df.empty:
        return pd.DataFrame()

    valid = perf_df.dropna(subset=["P(Winner)", "Correct"])
    if valid.empty:
        return pd.DataFrame()

    groups = valid.groupby("Round").agg(
        Games=("Correct", "count"),
        Correct=("Correct", "sum"),
        MSE=("MSE", "mean"),
    ).reset_index()
    groups["Accuracy"] = groups["Correct"] / groups["Games"]
    groups["Correct"] = groups["Correct"].astype(int)
    # Sort by round order
    groups["_order"] = groups["Round"].map(ROUND_ORDER).fillna(99)
    groups = groups.sort_values("_order").drop(columns="_order")
    return groups

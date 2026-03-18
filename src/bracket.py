"""Bracket simulation engine for NCAA tournament predictions.

Given tournament seeds, team features, and trained models, this module
can simulate the entire bracket round by round. Supports both
deterministic (pick the favorite) and probabilistic (weighted random)
simulation modes.
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ACTUAL_RESULTS_FILE = Path(__file__).resolve().parent.parent / "data" / "actual_results_2026.json"


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Round display names
ROUND_NAMES = {
    "R1": "Round of 64",
    "R2": "Round of 32",
    "R3": "Sweet 16",
    "R4": "Elite 8",
    "R5": "Final Four",
    "R6": "Championship",
}

# Region display names — 2026 bracket mapping
# W=Duke(1), X=Florida(1), Y=Michigan(1), Z=Arizona(1)
REGION_NAMES = {
    "W": "East",
    "X": "South",
    "Y": "Midwest",
    "Z": "West",
}


class BracketPredictor:
    """Load models and predict matchup probabilities."""

    def __init__(self, model_dir: Path | None = None):
        """Load trained models and team features.

        Args:
            model_dir: Path to directory containing exported model artifacts.
        """
        model_dir = model_dir or MODELS_DIR

        with open(model_dir / "config.json") as f:
            self.config = json.load(f)

        with open(model_dir / "men_model.pkl", "rb") as f:
            self.men_model: LogisticRegression = pickle.load(f)
        with open(model_dir / "women_model.pkl", "rb") as f:
            self.women_model: LogisticRegression = pickle.load(f)
        with open(model_dir / "men_scaler.pkl", "rb") as f:
            self.men_scaler: StandardScaler = pickle.load(f)
        with open(model_dir / "women_scaler.pkl", "rb") as f:
            self.women_scaler: StandardScaler = pickle.load(f)

        self.team_features: pd.DataFrame = pd.read_pickle(model_dir / "team_features.pkl")
        self.teams: pd.DataFrame = pd.read_pickle(model_dir / "teams.pkl")
        self.features = self.config["features"]

    def team_name(self, team_id: int) -> str:
        """Look up team name by ID."""
        match = self.teams.loc[self.teams["TeamID"] == team_id, "TeamName"]
        if len(match) > 0:
            return match.values[0]
        return f"Unknown ({team_id})"

    def predict_matchup(
        self,
        team1_id: int,
        team2_id: int,
        season: int,
        gender: str = "M",
    ) -> float:
        """Predict P(team1 wins) for a single matchup.

        Args:
            team1_id: First team's ID.
            team2_id: Second team's ID.
            season: Season year.
            gender: 'M' for men's, 'W' for women's.

        Returns:
            Probability that team1 wins (0 to 1).
        """
        # Ensure canonical ordering (T1 < T2)
        if team1_id > team2_id:
            prob = self.predict_matchup(team2_id, team1_id, season, gender)
            return 1.0 - prob

        # Look up team features
        tf = self.team_features
        t1_feats = tf[(tf["Season"] == season) & (tf["TeamID"] == team1_id)]
        t2_feats = tf[(tf["Season"] == season) & (tf["TeamID"] == team2_id)]

        if len(t1_feats) == 0 or len(t2_feats) == 0:
            return 0.5  # No data — return neutral

        # Compute difference features (NaN/Inf → 0 so LogReg won't crash)
        feature_vals = []
        for feat in self.features:
            col = feat.replace("Diff_", "")
            t1_val = t1_feats[col].values[0] if col in t1_feats.columns else 0.0
            t2_val = t2_feats[col].values[0] if col in t2_feats.columns else 0.0
            try:
                diff = float(t1_val) - float(t2_val)
            except (TypeError, ValueError):
                diff = 0.0
            feature_vals.append(diff)

        X = np.nan_to_num(np.array([feature_vals], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

        # Select model based on gender
        if gender == "M":
            X_scaled = np.nan_to_num(self.men_scaler.transform(X), nan=0.0)
            prob = self.men_model.predict_proba(X_scaled)[0, 1]
        else:
            X_scaled = np.nan_to_num(self.women_scaler.transform(X), nan=0.0)
            prob = self.women_model.predict_proba(X_scaled)[0, 1]

        return float(np.clip(prob, 0.025, 0.975))

    def predict_score(
        self,
        team1_id: int,
        team2_id: int,
        season: int,
        prob_t1_wins: float,
    ) -> tuple[int, int]:
        """Predict a plausible score line for a matchup.

        Uses each team's avg Score and opponent's avg Opp_Score as a baseline,
        then adjusts the spread to be consistent with the win probability.

        Args:
            team1_id: First team's ID.
            team2_id: Second team's ID.
            season: Season year.
            prob_t1_wins: P(team1 wins) from predict_matchup.

        Returns:
            Tuple of (team1_score, team2_score) as rounded integers.
        """
        tf = self.team_features
        t1 = tf[(tf["Season"] == season) & (tf["TeamID"] == team1_id)]
        t2 = tf[(tf["Season"] == season) & (tf["TeamID"] == team2_id)]

        if len(t1) == 0 or len(t2) == 0:
            return (70, 70)

        # Baseline: average of team's offense and opponent's defense
        t1_score = (float(t1["Score"].values[0]) + float(t2["Opp_Score"].values[0])) / 2
        t2_score = (float(t2["Score"].values[0]) + float(t1["Opp_Score"].values[0])) / 2

        # Adjust spread to be consistent with win probability
        # Map prob to a spread: prob=0.5 → 0pt spread, prob=0.75 → ~8pt spread
        target_spread = 16.0 * (prob_t1_wins - 0.5)  # rough linear mapping
        current_spread = t1_score - t2_score
        adjustment = (target_spread - current_spread) / 2
        t1_score += adjustment
        t2_score -= adjustment

        # Tournament games are typically lower-scoring; nudge toward ~70
        avg = (t1_score + t2_score) / 2
        tourney_avg = 70.0
        scale = 0.7  # blend toward tournament average
        t1_score = t1_score - (avg - tourney_avg) * (1 - scale)
        t2_score = t2_score - (avg - tourney_avg) * (1 - scale)

        return (max(round(t1_score), 40), max(round(t2_score), 40))


class SubmissionPredictor:
    """Wraps BracketPredictor but uses pre-computed submission CSV for predictions.

    Falls back to the base predictor's live model when the lookup misses.
    """

    def __init__(
        self,
        base: BracketPredictor,
        submission_path: Path,
        season: int = 2026,
    ):
        self._base = base
        self._preds: dict[tuple[int, int], float] = {}
        self._load(submission_path, season)
        # Proxy attributes that downstream code relies on
        self.config = base.config
        self.team_features = base.team_features
        self.teams = base.teams
        self.features = base.features

    def _load(self, path: Path, season: int) -> None:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            parts = str(row["ID"]).split("_")
            if len(parts) != 3:
                continue
            s, t1, t2 = int(parts[0]), int(parts[1]), int(parts[2])
            if s == season:
                self._preds[(t1, t2)] = float(row["Pred"])

    def predict_matchup(
        self, team1_id: int, team2_id: int, season: int, gender: str = "M"
    ) -> float:
        if team1_id > team2_id:
            return 1.0 - self.predict_matchup(team2_id, team1_id, season, gender)
        key = (team1_id, team2_id)
        if key in self._preds:
            return self._preds[key]
        return self._base.predict_matchup(team1_id, team2_id, season, gender)

    def predict_score(
        self, team1_id: int, team2_id: int, season: int, prob_t1_wins: float
    ) -> tuple[int, int]:
        return self._base.predict_score(team1_id, team2_id, season, prob_t1_wins)

    def team_name(self, team_id: int) -> str:
        return self._base.team_name(team_id)


class BracketSimulator:
    """Simulate an NCAA tournament bracket."""

    def __init__(
        self,
        predictor: BracketPredictor,
        season: int,
        gender: str = "M",
        data_dir: Path | None = None,
    ):
        """Initialize bracket for a specific season and gender.

        Args:
            predictor: BracketPredictor instance with loaded models.
            season: Tournament season year.
            gender: 'M' for men's, 'W' for women's.
            data_dir: Path to data directory with seeds/slots CSVs.
        """
        self.predictor = predictor
        self.season = season
        self.gender = gender
        self.data_dir = data_dir or DATA_DIR

        # Load seeds and slots for this season
        self.seeds = self._load_seeds()
        self.slots = self._load_slots()

        # Map seed code → TeamID (e.g., "W01" → 1234)
        self.seed_to_team: dict[str, int] = {}
        for _, row in self.seeds.iterrows():
            self.seed_to_team[row["Seed"]] = row["TeamID"]

        # Results: slot → winning TeamID
        self.results: dict[str, int] = {}

        # Probabilities: slot → P(strong seed wins)
        self.probabilities: dict[str, float] = {}

        # User overrides: slot → TeamID (user-selected winner)
        self.overrides: dict[str, int] = {}

        # Actual results: slot → dict with winner_id, score, etc.
        # These are locked — users can't override them.
        self.actual_results: dict[str, dict] = {}
        self._load_actual_results()

    def _load_actual_results(self) -> None:
        """Load actual tournament results from JSON and apply them."""
        if not ACTUAL_RESULTS_FILE.exists():
            return
        try:
            with open(ACTUAL_RESULTS_FILE) as f:
                data = json.load(f)
            key = "men" if self.gender == "M" else "women"
            results = data.get(key, {})
            for slot, info in results.items():
                winner_id = info.get("winner_id")
                if winner_id is not None:
                    self.results[slot] = winner_id
                    self.actual_results[slot] = info
        except (json.JSONDecodeError, KeyError):
            pass

    def _load_seeds(self) -> pd.DataFrame:
        """Load tournament seeds for this season."""
        prefix = "M" if self.gender == "M" else "W"
        seeds = pd.read_csv(self.data_dir / f"{prefix}NCAATourneySeeds.csv")
        seeds = seeds[seeds["Season"] == self.season].copy()
        seeds["seed_num"] = seeds["Seed"].apply(lambda x: int(x[1:3]))
        return seeds

    def _load_slots(self) -> pd.DataFrame:
        """Load tournament bracket slots for this season."""
        prefix = "M" if self.gender == "M" else "W"
        slots = pd.read_csv(self.data_dir / f"{prefix}NCAATourneySlots.csv")
        slots = slots[slots["Season"] == self.season].copy()
        return slots

    def get_team_for_code(self, code: str) -> Optional[int]:
        """Resolve a seed code or slot reference to a TeamID.

        Handles:
        - Direct seed codes like "W01", "X16a"
        - Slot references like "R1W1" (resolved from prior results)

        Args:
            code: Seed code or slot name.

        Returns:
            TeamID or None if not yet resolved.
        """
        # Direct seed → team mapping
        if code in self.seed_to_team:
            return self.seed_to_team[code]

        # Slot reference → look up result
        if code in self.results:
            return self.results[code]

        return None

    def get_matchups_for_round(self, round_prefix: str) -> list[dict]:
        """Get all matchups for a specific round.

        Args:
            round_prefix: Round prefix like "R1", "R2", etc.

        Returns:
            List of dicts with slot, strong_code, weak_code, strong_team,
            weak_team, strong_name, weak_name, probability.
        """
        matchups = []
        round_slots = self.slots[self.slots["Slot"].str.startswith(round_prefix)].copy()
        round_slots = round_slots.sort_values("Slot")

        for _, row in round_slots.iterrows():
            slot = row["Slot"]
            strong_code = row["StrongSeed"]
            weak_code = row["WeakSeed"]

            strong_team = self.get_team_for_code(strong_code)
            weak_team = self.get_team_for_code(weak_code)

            prob = None
            if strong_team is not None and weak_team is not None:
                prob = self.predictor.predict_matchup(
                    strong_team, weak_team, self.season, self.gender
                )

            matchups.append({
                "slot": slot,
                "strong_code": strong_code,
                "weak_code": weak_code,
                "strong_team": strong_team,
                "weak_team": weak_team,
                "strong_name": self.predictor.team_name(strong_team) if strong_team else "TBD",
                "weak_name": self.predictor.team_name(weak_team) if weak_team else "TBD",
                "probability": prob,  # P(strong seed wins)
            })

        return matchups

    def simulate_game(self, slot: str, mode: str = "deterministic") -> Optional[int]:
        """Simulate a single game and record the result.

        Args:
            slot: Bracket slot name (e.g., "R1W1").
            mode: "deterministic" (pick favorite) or "probabilistic" (weighted random).

        Returns:
            Winning TeamID, or None if teams not yet resolved.
        """
        # Check for user override first
        if slot in self.overrides:
            winner = self.overrides[slot]
            self.results[slot] = winner
            return winner

        slot_row = self.slots[self.slots["Slot"] == slot]
        if len(slot_row) == 0:
            return None

        slot_row = slot_row.iloc[0]
        strong_team = self.get_team_for_code(slot_row["StrongSeed"])
        weak_team = self.get_team_for_code(slot_row["WeakSeed"])

        if strong_team is None or weak_team is None:
            return None

        prob = self.predictor.predict_matchup(
            strong_team, weak_team, self.season, self.gender
        )
        self.probabilities[slot] = prob

        if mode == "deterministic":
            winner = strong_team if prob >= 0.5 else weak_team
        else:
            winner = strong_team if np.random.random() < prob else weak_team

        self.results[slot] = winner
        return winner

    def simulate_full_bracket(self, mode: str = "deterministic") -> dict[str, int]:
        """Simulate the entire bracket from first round to championship.

        Processes rounds in order so each round's results feed the next.

        Args:
            mode: "deterministic" or "probabilistic".

        Returns:
            Dict mapping slot names to winning TeamIDs.
        """
        self.results.clear()
        self.probabilities.clear()

        # Process play-in games first (slots like "W16", "X11", etc.)
        playin_slots = self.slots[
            ~self.slots["Slot"].str.startswith("R")
        ].sort_values("Slot")
        for _, row in playin_slots.iterrows():
            self.simulate_game(row["Slot"], mode)

        # Process rounds R1 through R6
        for round_num in range(1, 7):
            prefix = f"R{round_num}"
            round_slots = self.slots[
                self.slots["Slot"].str.startswith(prefix)
            ].sort_values("Slot")
            for _, row in round_slots.iterrows():
                self.simulate_game(row["Slot"], mode)

        return self.results

    def get_bracket_summary(self) -> list[dict]:
        """Get a formatted summary of the simulated bracket.

        Returns:
            List of dicts with round, slot, team1, team2, winner, probability.
        """
        summary = []
        for _, row in self.slots.sort_values("Slot").iterrows():
            slot = row["Slot"]
            strong_team = self.get_team_for_code(row["StrongSeed"])
            weak_team = self.get_team_for_code(row["WeakSeed"])
            winner = self.results.get(slot)
            prob = self.probabilities.get(slot)

            # Determine round name
            if slot.startswith("R"):
                round_key = slot[:2]
                round_name = ROUND_NAMES.get(round_key, round_key)
            else:
                round_name = "Play-In"

            summary.append({
                "round": round_name,
                "slot": slot,
                "team1_id": strong_team,
                "team2_id": weak_team,
                "team1_name": self.predictor.team_name(strong_team) if strong_team else "TBD",
                "team2_name": self.predictor.team_name(weak_team) if weak_team else "TBD",
                "winner_id": winner,
                "winner_name": self.predictor.team_name(winner) if winner else "TBD",
                "probability": prob,
            })

        return summary

    def set_override(self, slot: str, team_id: int) -> None:
        """Set a user override for a specific game.

        Args:
            slot: Bracket slot name.
            team_id: TeamID of the user's pick.
        """
        self.overrides[slot] = team_id

    def clear_overrides(self) -> None:
        """Remove all user overrides."""
        self.overrides.clear()

    def get_champion(self) -> Optional[int]:
        """Get the predicted champion's TeamID."""
        # Championship slot varies — find the R6 slot
        r6_slots = self.slots[self.slots["Slot"].str.startswith("R6")]
        if len(r6_slots) > 0:
            champ_slot = r6_slots.iloc[0]["Slot"]
            return self.results.get(champ_slot)
        return None

"""Train final models and export artifacts for the bracket app.

Trains separate men's and women's logistic regression models on all
available tournament data (2010–2025), computes team features for the
current season, and saves everything to models/.

Usage:
    python -m src.export_models          # from repo root
    python src/export_models.py          # also works
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Allow running as script or module
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data_loader, features


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Best validated feature set (from experiment leaderboard)
BEST_FEATURES = ["Diff_seed", "Diff_PointDiff", "Diff_OffEff", "Diff_WinPct",
                 "Diff_Elo", "Diff_FGPct", "Diff_FTPct"]
MEN_C = 0.25
WOMEN_C = 0.15

CURRENT_SEASON = 2026


def models_are_fresh(
    model_dir: Path | None = None,
    max_age_hours: float = 24,
) -> bool:
    """Check whether exported models exist and are recent enough.

    Args:
        model_dir: Directory containing model artifacts.
        max_age_hours: Maximum age in hours before models are stale.

    Returns:
        True if models exist and were created within *max_age_hours*.
    """
    import time

    model_dir = model_dir or MODELS_DIR
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return False

    age_hours = (time.time() - config_path.stat().st_mtime) / 3600
    return age_hours < max_age_hours


def ensure_models(
    model_dir: Path | None = None,
    data_dir: Path | None = None,
    max_age_hours: float = 24,
    current_season: int = CURRENT_SEASON,
) -> dict:
    """Download fresh data and (re)train models if they're stale.

    Args:
        model_dir: Where to save model artifacts.
        data_dir: Where to download/find CSV data.
        max_age_hours: Staleness threshold.
        current_season: Season year for feature computation.

    Returns:
        Model config dict.
    """
    model_dir = model_dir or MODELS_DIR
    data_dir = data_dir or DATA_DIR

    if models_are_fresh(model_dir, max_age_hours):
        print(f"Models are fresh (< {max_age_hours}h old). Skipping retrain.")
        with open(model_dir / "config.json") as f:
            return json.load(f)

    # Try downloading latest data (non-fatal if Kaggle CLI unavailable)
    try:
        data_loader.download_latest_data(data_dir)
    except Exception as e:
        print(f"Warning: Could not download latest data: {e}")
        print("Using existing data files.")

    return train_and_export(model_dir, current_season)


def train_and_export(
    model_dir: Path | None = None,
    current_season: int = CURRENT_SEASON,
) -> dict:
    """Train final models and save all artifacts.

    Args:
        model_dir: Directory to save model artifacts. Defaults to models/.
        current_season: Season to compute team features for.

    Returns:
        Dict with model performance metrics and paths.
    """
    model_dir = model_dir or MODELS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  TRAINING FINAL MODELS FOR BRACKET APP")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading data...")
    regular_season = data_loader.load_regular_season()
    tourney_results = data_loader.load_tourney_results()
    seeds = data_loader.load_tourney_seeds()
    compact_results = data_loader.load_compact_results()
    teams = data_loader.load_teams()

    # Filter to 2010+
    regular_season = regular_season[regular_season["Season"] >= 2010].copy()
    tourney_results = tourney_results[tourney_results["Season"] >= 2010].copy()

    print(f"  Regular season: {len(regular_season):,} games")
    print(f"  Tournament: {len(tourney_results):,} games")
    print(f"  Seeds: {len(seeds):,} entries")

    # ------------------------------------------------------------------
    # 2. Compute features
    # ------------------------------------------------------------------
    print("\n[2/5] Computing features...")
    game_data = features.prepare_game_data(regular_season)
    season_stats = features.compute_season_stats(game_data)
    win_pct = features.compute_win_pct(game_data)
    efficiency = features.compute_efficiency(game_data)
    momentum = features.compute_last14_momentum(game_data)
    quality = features.compute_team_quality(game_data, seeds)
    elo = features.compute_elo_ratings(compact_results)
    shooting = features.compute_shooting_pcts(season_stats)

    tf = features.build_team_features(
        season_stats, win_pct, efficiency, momentum, seeds,
        quality=quality, elo=elo, shooting=shooting,
    )
    print(f"  Team features: {len(tf):,} team-seasons, {len(tf.columns)} columns")

    # ------------------------------------------------------------------
    # 3. Build tournament matchup data for training
    # ------------------------------------------------------------------
    print("\n[3/5] Building training matchups...")
    tourney_game_data = features.prepare_game_data(tourney_results)

    # Create matchup DataFrame (ensure T1 < T2)
    tourney_matchups = tourney_game_data[
        tourney_game_data["T1_TeamID"] < tourney_game_data["T2_TeamID"]
    ][["Season", "T1_TeamID", "T2_TeamID", "PointDiff"]].copy()
    tourney_matchups["T1_Win"] = (tourney_matchups["PointDiff"] > 0).astype(int)

    tourney_enriched = features.create_matchup_df(tourney_matchups, tf)
    tourney_enriched, diff_cols = features.compute_difference_features(tourney_enriched)

    # Split M/W
    tourney_men = tourney_enriched[tourney_enriched["T1_TeamID"] < 3000].copy()
    tourney_women = tourney_enriched[tourney_enriched["T1_TeamID"] >= 3000].copy()

    # Exclude 2020 (COVID)
    train_men = tourney_men[~tourney_men["Season"].isin([2020])].copy()
    train_women = tourney_women[~tourney_women["Season"].isin([2020])].copy()

    print(f"  Men's training games: {len(train_men)}")
    print(f"  Women's training games: {len(train_women)}")

    # ------------------------------------------------------------------
    # 4. Train models
    # ------------------------------------------------------------------
    print("\n[4/5] Training models...")

    # Men's model
    X_m = train_men[BEST_FEATURES].fillna(0).values
    y_m = train_men["T1_Win"].values
    men_scaler = StandardScaler()
    X_m_scaled = men_scaler.fit_transform(X_m)
    men_model = LogisticRegression(C=MEN_C, max_iter=1000, random_state=42)
    men_model.fit(X_m_scaled, y_m)

    # Women's model
    X_w = train_women[BEST_FEATURES].fillna(0).values
    y_w = train_women["T1_Win"].values
    women_scaler = StandardScaler()
    X_w_scaled = women_scaler.fit_transform(X_w)
    women_model = LogisticRegression(C=WOMEN_C, max_iter=1000, random_state=42)
    women_model.fit(X_w_scaled, y_w)

    print(f"  Men's model coefficients: {dict(zip(BEST_FEATURES, men_model.coef_[0].round(3)))}")
    print(f"  Women's model coefficients: {dict(zip(BEST_FEATURES, women_model.coef_[0].round(3)))}")

    # ------------------------------------------------------------------
    # 5. Save artifacts
    # ------------------------------------------------------------------
    print("\n[5/5] Saving artifacts...")

    # Models + scalers
    with open(model_dir / "men_model.pkl", "wb") as f:
        pickle.dump(men_model, f)
    with open(model_dir / "women_model.pkl", "wb") as f:
        pickle.dump(women_model, f)
    with open(model_dir / "men_scaler.pkl", "wb") as f:
        pickle.dump(men_scaler, f)
    with open(model_dir / "women_scaler.pkl", "wb") as f:
        pickle.dump(women_scaler, f)

    # Team features for current season (and all seasons for historical lookups)
    tf.to_pickle(model_dir / "team_features.pkl")

    # Team name lookup
    teams.to_pickle(model_dir / "teams.pkl")

    # Config
    config = {
        "features": BEST_FEATURES,
        "model_type": "logreg",
        "model_params": {"C_men": MEN_C, "C_women": WOMEN_C, "max_iter": 1000},
        "current_season": current_season,
        "training_seasons": sorted([s for s in train_men["Season"].unique().tolist()]),
        "n_men_games": len(train_men),
        "n_women_games": len(train_women),
        "clip_range": [0.025, 0.975],
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Saved to {model_dir}/:")
    for p in sorted(model_dir.glob("*")):
        if p.is_file():
            size_kb = p.stat().st_size / 1024
            print(f"    {p.name:30s} ({size_kb:.1f} KB)")

    print("\n" + "=" * 60)
    print("  EXPORT COMPLETE — ready for bracket app!")
    print("=" * 60)

    return config


if __name__ == "__main__":
    train_and_export()

"""Generate 5 diverse Kaggle submissions for March Madness 2026.

Strategies:
  1. Split M/W LogReg on 6 features (seed, PD, OffEff, WinPct, quality, Elo)
  2. Split M/W LogReg on top-4 features only (seed, PD, OffEff, WinPct)
  3. Split M/W shallow XGBoost classifier on 6 features
  4. Ensemble: average of #1 (LogReg-6) and #3 (XGB-6)
  5. Combined single LogReg on all data (no M/W split)

Usage:
    python generate_submissions.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import data_loader, features

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

CURRENT_SEASON = 2026
CLIP = (0.025, 0.975)

FEATURES_6 = ["Diff_seed", "Diff_PointDiff", "Diff_OffEff",
               "Diff_WinPct", "Diff_quality", "Diff_Elo"]
FEATURES_4 = ["Diff_seed", "Diff_PointDiff", "Diff_OffEff", "Diff_WinPct"]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_and_build_features():
    """Load all data, compute features, return team features + tourney matchups."""
    print("Loading data...")
    regular_season = data_loader.load_regular_season()
    tourney_results = data_loader.load_tourney_results()
    seeds = data_loader.load_tourney_seeds()
    compact_results = data_loader.load_compact_results()
    sample_sub = data_loader.load_sample_submission(stage=2)

    regular_season = regular_season[regular_season["Season"] >= 2010].copy()
    tourney_results = tourney_results[tourney_results["Season"] >= 2010].copy()

    print("Computing features...")
    game_data = features.prepare_game_data(regular_season)
    season_stats = features.compute_season_stats(game_data)
    win_pct = features.compute_win_pct(game_data)
    efficiency = features.compute_efficiency(game_data)
    momentum = features.compute_last14_momentum(game_data)
    quality = features.compute_team_quality(game_data, seeds)
    elo = features.compute_elo_ratings(compact_results)

    tf = features.build_team_features(
        season_stats, win_pct, efficiency, momentum, seeds,
        quality=quality, elo=elo,
    )

    # Build tournament training set
    tourney_game_data = features.prepare_game_data(tourney_results)
    tourney_matchups = tourney_game_data[
        tourney_game_data["T1_TeamID"] < tourney_game_data["T2_TeamID"]
    ][["Season", "T1_TeamID", "T2_TeamID", "PointDiff"]].copy()
    tourney_matchups["T1_Win"] = (tourney_matchups["PointDiff"] > 0).astype(int)

    tourney_enriched = features.create_matchup_df(tourney_matchups, tf)
    tourney_enriched, diff_cols = features.compute_difference_features(tourney_enriched)

    # Exclude 2020
    tourney_enriched = tourney_enriched[~tourney_enriched["Season"].isin([2020])].copy()

    # Build prediction set from sample submission
    sub_parsed = pd.DataFrame({"ID": sample_sub["ID"]})
    parts = sub_parsed["ID"].str.split("_", expand=True)
    sub_parsed["Season"] = parts[0].astype(int)
    sub_parsed["T1_TeamID"] = parts[1].astype(int)
    sub_parsed["T2_TeamID"] = parts[2].astype(int)

    pred_matchups = features.create_matchup_df(sub_parsed, tf)
    pred_matchups, _ = features.compute_difference_features(pred_matchups)

    return tourney_enriched, pred_matchups, sample_sub


def split_mw(df):
    """Split into men's and women's subsets."""
    men = df[df["T1_TeamID"] < 3000].copy()
    women = df[df["T1_TeamID"] >= 3000].copy()
    return men, women


def train_predict_split_logreg(train_df, pred_df, feat_cols, C=1.0):
    """Train split M/W LogReg and return predictions aligned to pred_df."""
    train_m, train_w = split_mw(train_df)
    pred_m, pred_w = split_mw(pred_df)

    preds = pd.Series(index=pred_df.index, dtype=float)

    for label, tr, pr in [("Men", train_m, pred_m), ("Women", train_w, pred_w)]:
        X_tr = tr[feat_cols].fillna(0).values
        y_tr = tr["T1_Win"].values
        X_pr = pr[feat_cols].fillna(0).values

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_pr_s = scaler.transform(X_pr)

        model = LogisticRegression(C=C, max_iter=1000, random_state=42)
        model.fit(X_tr_s, y_tr)
        p = model.predict_proba(X_pr_s)[:, 1]
        preds.loc[pr.index] = p
        print(f"  {label}: {len(tr)} train, {len(pr)} pred, coeffs={dict(zip(feat_cols, model.coef_[0].round(3)))}")

    return preds.values


def train_predict_split_xgb(train_df, pred_df, feat_cols):
    """Train split M/W shallow XGBoost classifier and return predictions."""
    train_m, train_w = split_mw(train_df)
    pred_m, pred_w = split_mw(pred_df)

    preds = pd.Series(index=pred_df.index, dtype=float)

    for label, tr, pr in [("Men", train_m, pred_m), ("Women", train_w, pred_w)]:
        X_tr = tr[feat_cols].fillna(0).values
        y_tr = tr["T1_Win"].values
        X_pr = pr[feat_cols].fillna(0).values

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_pr_s = scaler.transform(X_pr)

        model = XGBClassifier(
            max_depth=2,
            n_estimators=200,
            learning_rate=0.05,
            min_child_weight=30,
            gamma=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=3.0,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        model.fit(X_tr_s, y_tr)
        p = model.predict_proba(X_pr_s)[:, 1]
        preds.loc[pr.index] = p
        print(f"  {label}: {len(tr)} train, {len(pr)} pred")

    return preds.values


def train_predict_combined_logreg(train_df, pred_df, feat_cols, C=1.0):
    """Train a single combined LogReg on all data (no M/W split)."""
    X_tr = train_df[feat_cols].fillna(0).values
    y_tr = train_df["T1_Win"].values
    X_pr = pred_df[feat_cols].fillna(0).values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_pr_s = scaler.transform(X_pr)

    model = LogisticRegression(C=C, max_iter=1000, random_state=42)
    model.fit(X_tr_s, y_tr)
    p = model.predict_proba(X_pr_s)[:, 1]
    print(f"  Combined: {len(train_df)} train, {len(pred_df)} pred, coeffs={dict(zip(feat_cols, model.coef_[0].round(3)))}")
    return p


def save_submission(preds, sample_sub, path, desc):
    """Save a clipped submission CSV."""
    sub = sample_sub[["ID"]].copy()
    sub["Pred"] = np.clip(preds, CLIP[0], CLIP[1])
    sub.to_csv(path, index=False)
    print(f"  -> {path} | {len(sub)} rows | range [{sub['Pred'].min():.4f}, {sub['Pred'].max():.4f}] | mean {sub['Pred'].mean():.4f}")
    return sub


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    tourney, pred_df, sample_sub = load_and_build_features()
    print(f"\nTraining data: {len(tourney)} tournament games")
    print(f"Prediction set: {len(pred_df)} matchups\n")

    results = {}

    # --- Submission 1: Split LogReg, 6 features (current best) ---
    print("=" * 60)
    print("SUB 1: Split M/W LogReg — 6 features (seed+PD+OffEff+WinPct+quality+Elo)")
    print("=" * 60)
    p1 = train_predict_split_logreg(tourney, pred_df, FEATURES_6)
    s1 = save_submission(p1, sample_sub, OUTPUT_DIR / "sub1_split_lr6.csv",
                         "Split M/W LogReg, 6 features")
    results["sub1"] = p1

    # --- Submission 2: Split LogReg, top-4 features ---
    print("\n" + "=" * 60)
    print("SUB 2: Split M/W LogReg — 4 features (seed+PD+OffEff+WinPct)")
    print("=" * 60)
    p2 = train_predict_split_logreg(tourney, pred_df, FEATURES_4)
    s2 = save_submission(p2, sample_sub, OUTPUT_DIR / "sub2_split_lr4.csv",
                         "Split M/W LogReg, 4 features")
    results["sub2"] = p2

    # --- Submission 3: Split XGBoost, 6 features ---
    print("\n" + "=" * 60)
    print("SUB 3: Split M/W XGBoost — 6 features")
    print("=" * 60)
    p3 = train_predict_split_xgb(tourney, pred_df, FEATURES_6)
    s3 = save_submission(p3, sample_sub, OUTPUT_DIR / "sub3_split_xgb6.csv",
                         "Split M/W XGBoost, 6 features")
    results["sub3"] = p3

    # --- Submission 4: Ensemble (avg of LR-6 + XGB-6) ---
    print("\n" + "=" * 60)
    print("SUB 4: Ensemble — average of Sub1 (LR-6) + Sub3 (XGB-6)")
    print("=" * 60)
    p4 = (p1 + p3) / 2.0
    s4 = save_submission(p4, sample_sub, OUTPUT_DIR / "sub4_ensemble_lr_xgb.csv",
                         "Ensemble: LR-6 + XGB-6 average")
    results["sub4"] = p4

    # --- Submission 5: Combined single LogReg, 6 features ---
    print("\n" + "=" * 60)
    print("SUB 5: Combined (no M/W split) LogReg — 6 features")
    print("=" * 60)
    p5 = train_predict_combined_logreg(tourney, pred_df, FEATURES_6)
    s5 = save_submission(p5, sample_sub, OUTPUT_DIR / "sub5_combined_lr6.csv",
                         "Combined LogReg, 6 features")
    results["sub5"] = p5

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY — All 5 submissions generated")
    print("=" * 60)

    # Pairwise correlation between submissions
    corr_data = pd.DataFrame(results)
    print("\nPrediction correlations:")
    print(corr_data.corr().round(3).to_string())

    print("\nPrediction statistics:")
    for name, preds in results.items():
        clipped = np.clip(preds, CLIP[0], CLIP[1])
        print(f"  {name}: mean={clipped.mean():.4f}, std={clipped.std():.4f}, "
              f"min={clipped.min():.4f}, max={clipped.max():.4f}")

    print("\nFiles ready for upload:")
    for f in sorted(OUTPUT_DIR.glob("sub*.csv")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.0f} KB)")

    print("\nDone!")


if __name__ == "__main__":
    main()

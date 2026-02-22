# March Machine Learning Mania 2026 — Copilot Instructions

## Project Overview
Kaggle competition: **March Machine Learning Mania 2026**  
Goal: Predict win probabilities for every possible matchup in both the **Men's** and **Women's** 2026 NCAA basketball tournaments.  
Metric: **Mean Squared Error** (Brier score) — lower is better.  
Prize: $50,000 pool · Awards Points & Medals.  
GitHub: `JTMarcu/march-madness` (public)

## Competition Rules (Key Points)
- Submit **P(Team1 wins)** for every possible `Season_Team1ID_Team2ID` pair where `Team1ID < Team2ID`.
- Output file columns: `ID,Pred` — predictions must be probabilities in `[0, 1]`.
- Stage 1: Predict historical seasons for validation. Stage 2: Predict the actual 2026 tournament.
- Both men's (IDs 1xxx) and women's (IDs 3xxx) teams are combined in one submission file.
- Overconfident predictions are **heavily penalized** by MSE — prefer calibrated probabilities near 0.5 for uncertain matchups rather than 0/1 extremes.

## Repository Structure

```
Madness/
├── .github/
│   └── copilot-instructions.md      # THIS FILE — project rules and conventions
├── .gitignore
├── data/                             # Raw Kaggle-provided CSVs (DO NOT modify originals)
│   ├── M*.csv                        # Men's data files
│   ├── W*.csv                        # Women's data files
│   ├── SampleSubmission*.csv         # Kaggle submission templates
│   ├── Cities.csv, Conferences.csv   # Supplementary data
│   └── Processed_*.csv              # Intermediate processed data (regenerable)
├── notebooks/
│   ├── 03_modeling.ipynb             # Original 3-phase pipeline (reference)
│   └── 04_refined.ipynb             # PRIMARY — experiment tracking + multi-year validation
├── src/                              # Reusable Python modules
│   ├── __init__.py
│   ├── data_loader.py                # Load & merge raw CSVs (men + women)
│   ├── features.py                   # Feature engineering pipeline
│   ├── models.py                     # Model training & prediction helpers
│   └── utils.py                      # Team lookups, submission generation, plotting
├── models/                           # Saved trained models (.pkl, .cbm, .json)
├── output/                           # Submission CSVs
├── old/                              # Old notebooks from prior years (reference only)
├── results/                          # Evaluation results, plots
└── README.md
```

## Data Files Reference

### Men's Data (prefix `M`)
| File | Description | Key Columns |
|------|-------------|-------------|
| `MRegularSeasonDetailedResults.csv` | Game-level box scores (2003–current) | Season, WTeamID, LTeamID, scores, FG/3P/FT made & attempted, rebounds, assists, turnovers, steals, blocks |
| `MNCAATourneyDetailedResults.csv` | Tournament game box scores (2003–current) | Same as above |
| `MRegularSeasonCompactResults.csv` | Game scores only (1985–current) | Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc |
| `MNCAATourneyCompactResults.csv` | Tournament scores only (1985–current) | Same as above |
| `MNCAATourneySeeds.csv` | Tournament seeds (1985–current) | Season, Seed (e.g., "W01"), TeamID |
| `MNCAATourneySlots.csv` | Bracket structure | Season, Slot, StrongSeed, WeakSeed |
| `MMasseyOrdinals.csv` | 3rd-party ranking systems (**men only**) | Season, RankingDayNum, SystemName, TeamID, OrdinalRank |
| `MTeamCoaches.csv` | Coach history | Season, TeamID, FirstDayNum, LastDayNum, CoachName |
| `MTeamConferences.csv` | Conference memberships | Season, TeamID, ConfAbbrev |
| `MTeams.csv` | Team ID ↔ name mapping | TeamID, TeamName |

### Women's Data (prefix `W`)
Same structure as men's data. Women's detailed results start from **2010**.  
Women's TeamIDs are in the **3xxx** range.  
**Note:** Massey Ordinals are NOT available for women's teams — use seed-based proxies instead.

### Supplementary
| File | Description |
|------|-------------|
| `Cities.csv` | City ID ↔ name/state mapping |
| `Conferences.csv` | Conference abbreviation ↔ full name |
| `MGameCities.csv` / `WGameCities.csv` | Which games were played in which cities |

## Coding Conventions

### Python & Libraries
- **Python 3.11** at `C:\Users\JonMa\AppData\Local\Programs\Python\Python311\python.exe`
- Virtual env at `.venv/`
- Primary libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`
- Type hints on all function signatures in `src/` modules
- Docstrings (Google style) on all public functions

### Data Processing Rules
1. **Never modify raw CSVs** in `data/` — always create derived DataFrames in memory or save to `output/`.
2. **Filter to 2010+ seasons** for training — detailed stats not available before 2003; women's data starts 2010.
3. **Skip season 2020** — COVID cancellation, no tournament data.
4. **Sort TeamIDs** in matchup pairs — always ensure `Team1ID < Team2ID` to match submission format.
5. **Combine men's and women's** data into a single training set (they share the same statistical features).
6. Compute **per-game averages** for all box score stats, not totals.
7. Use **feature differences** (`Diff_X = T1_X − T2_X`) rather than raw per-team features — this cuts feature count in half and the model learns relative strength.
8. **Massey Ordinals are men-only** — women's teams have no Massey data. Never fill women's missing Massey values with 0 (it misleads the model). Either add an `is_mens` flag, train separate M/W models, or exclude Massey entirely.

## src/ Module API Reference

### `data_loader.py` — Data Loading
All functions accept an optional `data_dir` parameter (defaults to `data/`).
```python
load_regular_season()      → pd.DataFrame   # Men + Women detailed results (2010+)
load_tourney_results()     → pd.DataFrame   # Men + Women tournament detailed results
load_tourney_seeds()       → pd.DataFrame   # Seeds with parsed 'seed' int column
load_compact_results()     → pd.DataFrame   # Compact results (scores only)
load_teams()               → pd.DataFrame   # TeamID ↔ TeamName
load_team_conferences()    → pd.DataFrame   # Conference memberships
load_massey_ordinals()     → pd.DataFrame   # Massey rankings (MEN ONLY)
load_coaches()             → pd.DataFrame   # Coach history
load_sample_submission(stage=1|2) → pd.DataFrame  # Kaggle submission template
```

### `features.py` — Feature Engineering Pipeline
All feature functions are keyed on `(Season, TeamID)`. The pipeline flows:

```
Raw data → prepare_game_data() → compute_*() functions → build_team_features()
                                                              ↓
Tournament results → create_matchup_df(tourney, team_features)
                              ↓
                     compute_difference_features(matchup_df) → Diff_* columns
```

Key functions:
```python
prepare_game_data(df)                    → symmetric T1/T2 rows (each game = 2 rows)
compute_season_stats(game_data)          → per-game averages: FGM, Opp_FGM, Score, etc.
compute_shooting_pcts(season_stats)      → FGPct, FG3Pct, FTPct (ratio features)
compute_win_pct(game_data)               → WinPct, Games
compute_efficiency(game_data)            → OffEff, DefEff, Possessions
compute_last14_momentum(game_data)       → win_ratio_14d
compute_sos(game_data, win_pct)          → SOS (avg opponent WinPct)
compute_team_quality(game_data, seeds)   → quality (L2-regularized GLM + z-score)
compute_massey_features(massey_raw)      → Massey_POM, Massey_SAG, etc. (MEN ONLY)
build_team_features(stats, winpct, eff, momentum, seeds,
                    quality=None, massey=None, shooting=None, sos=None)
                                          → single table keyed on (Season, TeamID)
create_matchup_df(matchups, team_features) → T1_* and T2_* columns via join
compute_difference_features(matchup_df)  → (df_with_Diff_cols, list_of_diff_col_names)
```

**Column naming convention:**
- Team stats: `FGM`, `FGA`, `Score`, `PointDiff`
- Opponent stats: `Opp_FGM`, `Opp_FGA`, `Opp_Score`
- After matchup merge: `T1_FGM`, `T2_FGM`
- After difference: `Diff_FGM` (= T1_FGM − T2_FGM)

### `models.py` — Training & Prediction
```python
DEFAULT_XGB_PARAMS           # Conservative XGB params (eta=0.02, max_depth=3, etc.)
cauchyobj(preds, dtrain)     → Cauchy loss gradient/hessian for XGB
train_xgb_cv(X, y, seasons)  → models, scores, spline_calibrators
train_xgb_final(X, y)       → single xgb model
fit_spline_calibrators(preds, labels, n_bins=200) → dict of spline functions
predict_probabilities(X, models, calibrators) → calibrated predictions
train_rf_baseline(X, y)     → RandomForestClassifier
evaluate_predictions(y_true, y_pred) → dict with MSE, log_loss, accuracy
```

### `utils.py` — Helpers
```python
team_id(name, gender='M'|'W'|None)  # "Duke" → 1181 (M) or 3181 (W)
team_name(tid)                       # 1181 → "Duke"
parse_submission_ids(sample_sub)     # Extract Season, T1_TeamID, T2_TeamID from ID col
generate_submission(preds, sample_sub, path, clip=[0.025, 0.975])
lookup_matchup(t1_name, t2_name, submission_df, gender='M')
plot_feature_importance(model, feature_names)
plot_calibration(y_true, y_pred)
valid_seasons(min_season=2003, max_season=2026)  # Excludes 2020
```

**`team_id()` behavior:** Tries exact match first, then case-insensitive exact, then partial substring. Use `gender='M'` or `gender='W'` to disambiguate (e.g., Duke has both M and W teams). Note: "UConn" is stored as "Connecticut" in the data.

## Feature Importance (Empirically Validated)

Ranked by absolute importance from 2025 validation (logistic regression coefficients):

| Rank | Feature | Coefficient | Impact |
|------|---------|-------------|--------|
| 1 | `Diff_seed` | −1.10 | **Dominant** — lower seed = higher win prob |
| 2 | `Diff_PointDiff` | +0.70 | Point differential in regular season |
| 3 | `Diff_OffEff` | +0.40 | Offensive efficiency (points per possession) |
| 4 | `Diff_WinPct` | +0.35 | Win percentage |
| 5 | `Diff_DefEff` | −0.30 | Defensive efficiency (lower = better) |
| 6 | `Diff_win_ratio_14d` | +0.20 | Late-season momentum |
| 7 | `Diff_quality` | +0.15 | GLM team quality metric |

**Key insight:** The top 4 features (`Diff_seed`, `Diff_PointDiff`, `Diff_OffEff`, `Diff_WinPct`) capture nearly all the signal. Additional features provide diminishing returns and increase overfitting risk on ~130 tournament games per year.

## Modeling Guidelines (Updated with 2025 Results)

### Critical Finding: Simple Models Win + Split M/W
With only **~1,962 total tournament games** (2010–2025, M+W), complex models overfit:
- **Split M/W LogReg: MSE = 0.1613** (BEST full-submission)
- Combined LogReg: MSE = 0.1625
- XGBoost classifier: MSE = 0.1617
- Women-only LogReg: MSE = 0.1339 (remarkably predictable)

**Rule: Use separate M/W models. Start with logistic regression on 4–6 features. Only add complexity if it beats LogReg on ≥2 of 3 holdout years.**

### Validation Strategy
1. **Multi-year holdout** — validate on 2023, 2024, AND 2025 (not just one year). A change only counts if it helps on **at least 2 of 3** holdout years.
2. Train on all years before the holdout year (no future leakage).
3. **GroupKFold with Season** as group for cross-validation if needed.
4. Final model: train on ALL 2010–2025 data, predict 2026.

### Model Selection Hierarchy
1. **LogisticRegression** (C=1.0) on top 4–6 diff features → strong baseline
2. **Shallow XGBoost** (max_depth=2, min_child_weight=30, gamma=5) → test non-linear interactions
3. **LightGBM** (num_leaves=8, max_depth=2) → alternative to XGBoost
4. **Simple average of LogReg + shallow XGB** → only if both models are individually good

### Calibration & Clipping
- Clip final predictions to `[0.025, 0.975]` — never predict absolute certainty.
- Spline calibration on MSE metric is fragile with small data — prefer well-calibrated models (LogReg is already calibrated).
- For tree models, use Platt scaling or isotonic regression.

### GLM Team Quality
The `compute_team_quality()` function uses L2-regularized logistic regression on team indicators, then z-score normalizes. Quality values should range approximately [-3, +3]. If values exceed [-10, 10], something is wrong — check the regularization parameter (currently `alpha=0.1, L1_wt=0.0`).

### Massey Ordinals
- Only available for **men's teams**
- Systems used: POM, SAG, MOR, DOL, COL (top-5 most complete systems)
- For combined M+W models, either: (a) add `is_mens` flag, (b) train separate models, or (c) exclude Massey entirely
- **Never fill women's Massey values with 0** — it biases predictions

## Experiment Tracking Methodology

Use `notebooks/04_refined.ipynb` for systematic experimentation:

```python
# Every experiment follows this pattern:
run_experiment(
    name='Top4 (LR)',
    description='LogReg on seed + PointDiff + OffEff + WinPct',
    feature_names=['Diff_seed', 'Diff_PointDiff', 'Diff_OffEff', 'Diff_WinPct'],
    matchup_df=tourney_f,
    model_type='logreg',  # or 'xgb', 'lgbm'
)
# This trains on all data except each holdout year, reports MSE for 2023/2024/2025

show_leaderboard()  # See all experiments ranked by average MSE
```

### Adding New Experiments
1. Compute any new features in `src/features.py`
2. Add a cell in `04_refined.ipynb` calling `run_experiment()`
3. Check the leaderboard — does it beat the current best on 2+ holdout years?
4. If yes, update `FINAL_FEATURES` and `FINAL_MODEL` in the submission cell

## Workflow for 2026

### Phase 1: Data Update
1. `kaggle competitions download -c march-machine-learning-mania-2026 -p data/`
2. Verify 2025 tournament results and 2026 regular season data present
3. Update `SampleSubmission*.csv` files for 2026

### Phase 2: Feature Engineering & Experiments
1. Run `04_refined.ipynb` from top — it automatically loads data and builds features
2. Run all experiment cells — the leaderboard shows what works
3. Try new feature/model combos as additional experiments
4. Keep what beats the baseline on 2+ holdout years

### Phase 3: Final Submission
1. Set `FINAL_FEATURES`, `FINAL_MODEL`, `FINAL_PARAMS` in the submission cell
2. Model trains on ALL 2010–2025 data
3. Generates `output/submission_refined.csv`
4. Sanity-check marquee matchups

## Previous Year Results
- **2025 competition**: RandomForest + 13 diff features. No seeds, no Massey, no ensemble. Basic approach.
- **2023 competition** (`old/paris-madness-2023.ipynb`): XGBoost regression with Cauchy loss, GLM quality, seed features, spline calibration. Most sophisticated prior approach.
- **2026 iteration 1** (`notebooks/03_modeling.ipynb`): 3-phase pipeline. LogReg MSE=0.1447 beat XGBoost (0.1529) and ensemble (0.1523). GLM quality was overflowing (fixed). Massey handling was wrong for women (fixed).
- **2026 iteration 2** (`notebooks/04_refined.ipynb`): 25+ experiments. Split M/W strategy discovered — women far more predictable (0.134 vs 0.187 men-only). SOS and shooting % tested but don't beat top4+quality. Final: split LR with top4+quality.

## Key Learnings (Empirically Validated)
1. **Simple logistic regression beats complex models** on ~2K tournament games — the dataset is too small for deep trees
2. **Seed difference is the single most important feature** (coeff −1.10) — the selection committee already encodes team strength
3. **Top 4 features capture nearly all signal**: seed, point diff, offensive efficiency, win%
4. **Feature differences >>> raw features** — halves feature count, better signal
5. **Multi-year validation is essential** — a model that looks great on 2025 alone may have gotten lucky
6. **Overconfident predictions destroy MSE** — always clip to [0.025, 0.975]
7. **GLM team quality must be regularized** — unregularized GLM on team indicators overflows (1e16 values)
8. **Massey ordinals are men-only** — filling women's data with 0 is worse than excluding them
9. **Split M/W models beat combined** — women's tournament is more predictable (fewer upsets, larger talent gaps), so a tuned women-only model dramatically improves overall MSE
10. **Point differential → probability calibration outperforms direct classification** (but LogReg is simpler and nearly as good)
11. **Every new feature/model must earn its place** — test on 3 holdout years, keep only if it helps 2+ of them
12. **SOS and shooting % are redundant** — seed already encodes SOS; shooting % overlaps with OffEff. Neither beats top4+quality in multi-year validation
13. **Regularization strength (C=1.0) is near-optimal** — sweep from 0.01 to 5.0 shows diminishing returns from deviating

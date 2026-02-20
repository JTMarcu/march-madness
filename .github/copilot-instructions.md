# March Machine Learning Mania 2026 — Copilot Instructions

## Project Overview
Kaggle competition: **March Machine Learning Mania 2026**  
Goal: Predict win probabilities for every possible matchup in both the **Men's** and **Women's** 2026 NCAA basketball tournaments.  
Metric: **Mean Squared Error** (Brier score) — lower is better.  
Prize: $50,000 pool · Awards Points & Medals.

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
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb  # Feature pipeline
│   ├── 03_modeling.ipynb             # Model training & validation
│   ├── 04_submission.ipynb           # Generate final submission CSV
│   └── 05_bracket_analysis.ipynb     # Post-prediction bracket explorer
├── src/                              # Reusable Python modules
│   ├── __init__.py
│   ├── data_loader.py                # Load & merge raw CSVs
│   ├── features.py                   # Feature engineering functions
│   ├── models.py                     # Model training & prediction
│   └── utils.py                      # Helpers (team name lookup, plotting, etc.)
├── models/                           # Saved trained models (.pkl, .cbm, .json)
├── output/                           # Submission CSVs
├── archive/                          # Old notebooks from prior years (reference only)
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
| `MMasseyOrdinals.csv` | 3rd-party ranking systems | Season, RankingDayNum, SystemName, TeamID, OrdinalRank |
| `MTeamCoaches.csv` | Coach history | Season, TeamID, FirstDayNum, LastDayNum, CoachName |
| `MTeamConferences.csv` | Conference memberships | Season, TeamID, ConfAbbrev |
| `MTeams.csv` | Team ID ↔ name mapping | TeamID, TeamName |

### Women's Data (prefix `W`)
Same structure as men's data. Women's detailed results start from **2010**.  
Women's TeamIDs are in the **3xxx** range.

### Supplementary
| File | Description |
|------|-------------|
| `Cities.csv` | City ID ↔ name/state mapping |
| `Conferences.csv` | Conference abbreviation ↔ full name |
| `MGameCities.csv` / `WGameCities.csv` | Which games were played in which cities |

## Coding Conventions

### Python & Libraries
- **Python 3.11+** with venv at `.venv/`
- Primary libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `catboost`, `lightgbm`, `scipy`, `matplotlib`, `seaborn`
- Use `polars` for large data operations when performance matters
- Type hints on all function signatures in `src/` modules
- Docstrings (Google style) on all public functions

### Data Processing Rules
1. **Never modify raw CSVs** in `data/` — always create derived DataFrames in memory or save to `output/`.
2. **Filter to 2010+ seasons** for training — detailed stats not available before 2003; women's data starts 2010.
3. **Skip season 2020** — COVID cancellation, no tournament data.
4. **Sort TeamIDs** in matchup pairs — always ensure `Team1ID < Team2ID` to match submission format.
5. **Combine men's and women's** data into a single training set (they share the same statistical features).
6. Compute **per-game averages** for all box score stats, not totals.
7. Use **feature differences** (Team1_stat − Team2_stat) rather than raw per-team features — this cuts feature count in half and the model learns relative strength.

### Feature Engineering (Proven Features)
These features have been validated in prior years:

| Feature | Formula | Category |
|---------|---------|----------|
| WinPct | Wins / GamesPlayed | Record |
| AvgPointsFor | Mean points scored | Scoring |
| AvgPointsAgainst | Mean points allowed | Defense |
| ScoreMargin | PointsFor − PointsAgainst | Combined |
| FGP | FGM / FGA | Shooting |
| FG3P | FGM3 / FGA3 | Shooting |
| FTP | FTM / FTA | Shooting |
| AvgOR | Mean offensive rebounds | Rebounding |
| AvgDR | Mean defensive rebounds | Rebounding |
| AvgTO | Mean turnovers | Ball control |
| OffEff | (Score / Possessions) × 100 | Efficiency |
| DefEff | (OppScore / Possessions) × 100 | Efficiency |
| Possessions | FGA − OR + TO + 0.475 × FTA | Tempo |
| SeedNum | Parsed from seed string (1–16) | Seeding |
| Seed_diff | Team1Seed − Team2Seed | Seeding |
| TeamQuality | GLM coefficient (logistic reg on team IDs) | Advanced |
| Last14DaysWinRatio | Win% in final 14 days of regular season | Momentum |

### Features to Add for 2026 (Improvements)
- **Massey Ordinals**: Use top ranking systems (POM, SAG, MOR, etc.) as features or ensemble weights.
- **Conference strength**: Average win% of conference opponents.
- **Coach experience**: Years coaching + tournament appearances.
- **Travel distance**: Euclidean distance from team city to game city.
- **Strength of Schedule (SOS)**: Average opponent win%.
- **Adjusted efficiency margins** (KenPom-style): Tempo-free offensive/defensive ratings.
- **Recency weighting**: Weight recent games more heavily.
- **Historical tournament performance**: Team's historical tournament win rate.

### Modeling Guidelines
1. **Validation strategy**: Train on seasons 2010–(current−2), validate on season (current−1). Never leak tournament data from the prediction season.
2. **Cross-validation**: Use `GroupKFold` with `Season` as the group — prevents temporal leakage.
3. **Models to ensemble**:
   - XGBoost (regression on point differential → calibrate to probability)
   - CatBoost (handles categoricals natively)
   - LightGBM (fast, good with ordinals)
   - Random Forest (robust baseline)
4. **Calibration**: After training, calibrate probabilities using `CalibratedClassifierCV` or `scipy.interpolate.UnivariateSpline` on validation set.
5. **Clip predictions**: Clip final probabilities to `[0.025, 0.975]` — never predict absolute certainty.
6. **Seed-based priors**: For extreme seed mismatches (1 vs 16), blend model predictions with historical seed-based win rates.
7. **Log loss during training, MSE for final evaluation** — they optimize similar things but MSE is the competition metric.

### Submission Generation
```python
# Standard submission generation pattern
submission = sample_sub.copy()
submission['Pred'] = submission['ID'].apply(lambda x: predict_matchup(x, model, features))
submission['Pred'] = submission['Pred'].clip(0.025, 0.975)
submission.to_csv('output/submission_2026.csv', index=False)
```

## Workflow for 2026

### Phase 1: Data Update (When Kaggle releases 2026 data)
1. Download fresh CSVs from Kaggle into `data/`
2. Verify 2025 tournament results are now in the data
3. Verify 2026 regular season data is included
4. Update `SampleSubmission*.csv` files for 2026

### Phase 2: Feature Engineering
1. Run `src/features.py` to generate team stats for all seasons including 2026
2. Verify feature distributions haven't shifted dramatically
3. Add any new features (Massey ordinals, coach data, etc.)

### Phase 3: Model Training
1. Train on 2010–2024, validate on 2025
2. Evaluate MSE on 2025 validation (we know actual results now)
3. Tune hyperparameters
4. Train ensemble on 2010–2025, predict 2026

### Phase 4: Submission
1. Generate predictions for all 2026 matchups
2. Calibrate and clip probabilities
3. Sanity check: Do top seeds have high win probabilities?
4. Generate final `output/submission_2026.csv`

## Previous Year Results
- **2025 competition**: Used RandomForest with 13 difference features. Basic feature engineering with offensive/defensive efficiency. No seeds, no Massey ordinals, no ensemble.
- **2023 competition (paris-madness-2023.ipynb)**: Used XGBoost regression on point differential with custom Cauchy loss, GLM team quality metric, seed features, spline probability calibration. This was the most sophisticated approach.

## Key Learnings from Prior Years
1. Feature differences >>> raw features (halves feature count, better signal)
2. Seed information is extremely predictive — always include it
3. Point differential regression → probability calibration outperforms direct classification
4. Overconfident predictions destroy MSE scores — always clip/calibrate
5. Ensemble models beat any single model
6. Team quality metrics (GLM/Elo) add significant signal beyond box score stats
7. Recent form (last 14 days) captures injury/momentum effects
8. Don't over-engineer women's vs men's differences — a combined model works fine

# March Machine Learning Mania 2026

Kaggle competition to predict NCAA basketball tournament outcomes for both Men's and Women's 2026 tournaments.

**Competition**: [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)  
**Metric**: Mean Squared Error (Brier score) — lower is better  
**Prize**: $50,000 pool · Awards Points & Medals

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
# 1. Explore data
jupyter notebook notebooks/01_eda.ipynb

# 2. Engineer features & train
jupyter notebook notebooks/02_feature_engineering.ipynb
jupyter notebook notebooks/03_modeling.ipynb

# 3. Generate submission
jupyter notebook notebooks/04_submission.ipynb
```

## Repository Structure

```
Madness/
├── data/           # Raw Kaggle CSVs (download from competition)
├── notebooks/      # Ordered pipeline notebooks (01–05)
├── src/            # Reusable Python modules
├── models/         # Saved trained models
├── output/         # Submission CSVs
├── results/        # Evaluation plots & metrics
└── archive/        # Prior year notebooks (2023, 2025)
```

## Approach

### Features
- Box score averages (FG%, 3P%, FT%, rebounds, assists, turnovers)
- Offensive & defensive efficiency (points per 100 possessions)
- Tournament seed numbers & seed differences
- GLM team quality metric (logistic regression on team IDs)
- Last-14-days momentum (recent win rate)
- Massey Ordinal rankings (top systems)

### Models
- **XGBoost** regression on point differential with custom Cauchy loss
- **CatBoost** & **LightGBM** for ensemble diversity
- **Random Forest** baseline
- Spline-based probability calibration
- Predictions clipped to [0.025, 0.975]

## Data

Download data from [Kaggle competition page](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data) and place CSVs in `data/`.

## Prior Results
- **2025**: RandomForest with 13 difference features (basic approach)
- **2023**: XGBoost + GLM team quality + seeds + spline calibration (most sophisticated)
- **2026 goal**: Ensemble of multiple models with richer features

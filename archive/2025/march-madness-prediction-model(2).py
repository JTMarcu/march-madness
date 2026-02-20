# Define cross-validation strategy - stratified by gender and outcome
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# Dictionary to store results
results = {}

# Function to evaluate model performance
def evaluate_model(model, X_train, y_train, X_val, y_val, gender_train, gender_val):
    """Evaluate model performance with detailed metrics."""
    # Cross-validation using both gender and outcome for stratification
    strat_labels = np.array([f"{y}_{g}" for y, g in zip(y_train, gender_train)])
    cv_scores = []
    
    for train_idx, test_idx in cv.split(X_train, strat_labels):
        X_cv_train, X_cv_test = X_train[train_idx], X_train[test_idx]
        y_cv_train, y_cv_test = y_train[train_idx], y_train[test_idx]
        
        model.fit(X_cv_train, y_cv_train)
        y_cv_pred = model.predict_proba(X_cv_test)[:, 1]
        
        # Calculate Brier score
        brier = brier_score_loss(y_cv_test, y_cv_pred)
        cv_scores.append(brier)
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Predict on validation set
    y_val_pred = model.predict_proba(X_val)[:, 1]
    val_brier = brier_score_loss(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_pred)
    
    # Evaluate separately for men's and women's games
    men_idx = gender_val == 0
    women_idx = gender_val == 1
    
    men_brier = brier_score_loss(y_val[men_idx], y_val_pred[men_idx]) if sum(men_idx) > 0 else np.nan
    women_brier = brier_score_loss(y_val[women_idx], y_val_pred[women_idx]) if sum(women_idx) > 0 else np.nan
    
    return {
        'cv_brier_mean': cv_mean,
        'cv_brier_std': cv_std,
        'val_brier': val_brier,
        'val_auc': val_auc,
        'men_brier': men_brier,
        'women_brier': women_brier
    }

# Train and evaluate each model
for name, model in models.items():
    log_message(f"\nTraining {name}...")
    
    # Evaluate model
    metrics = evaluate_model(model, X_train_scaled, y_train, X_val_scaled, y_val, gender_train, gender_val)
    
    log_message(f"Cross-validation Brier score: {metrics['cv_brier_mean']:.4f} ± {metrics['cv_brier_std']:.4f}")
    log_message(f"Validation Brier score: {metrics['val_brier']:.4f}")
    log_message(f"Validation AUC: {metrics['val_auc']:.4f}")
    log_message(f"Men's Brier score: {metrics['men_brier']:.4f}")
    log_message(f"Women's Brier score: {metrics['women_brier']:.4f}")
    
    # Store results
    results[name] = {
        'model': model,
        'metrics': metrics
    }

# Find the best model based on validation Brier score
best_model_name = min(results, key=lambda k: results[k]['metrics']['val_brier'])
best_model = results[best_model_name]['model']
log_message(f"\nBest model: {best_model_name} with validation Brier score: {results[best_model_name]['metrics']['val_brier']:.4f}")

#############################################################
# STEP 7: HYPERPARAMETER TUNING
# Goal: Fine-tune the best performing model to maximize
# predictive performance.
#############################################################

log_message("\n7. PERFORMING HYPERPARAMETER TUNING...")

# Custom parameter grids for each model type
param_grids = {
    'Logistic Regression': {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    },
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 0.9, 1.0]
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    },
    'LightGBM': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'num_leaves': [31, 63, 127],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
}

# Get appropriate parameter grid
param_grid = param_grids.get(best_model_name, {})

if param_grid:
    log_message(f"Running grid search for {best_model_name} with {len(param_grid)} parameter sets...")
    
    # Define a custom stratification that combines gender and outcome
    strat_labels = np.array([f"{y}_{g}" for y, g in zip(y_train, gender_train)])
    
    # Set up grid search
    grid_search = GridSearchCV(
        estimator=clone(best_model),
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
        scoring='neg_brier_score',
        n_jobs=-1,
        verbose=1
    )
    
    # Run grid search
    grid_search.fit(X_train_scaled, y_train, groups=strat_labels)
    
    # Get best parameters and model
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Convert back to positive Brier score
    tuned_model = grid_search.best_estimator_
    
    log_message(f"\nBest parameters: {best_params}")
    log_message(f"Best cross-validation Brier score: {best_score:.4f}")
    
    # Evaluate tuned model on validation set
    y_val_pred_tuned = tuned_model.predict_proba(X_val_scaled)[:, 1]
    brier_tuned = brier_score_loss(y_val, y_val_pred_tuned)
    auc_tuned = roc_auc_score(y_val, y_val_pred_tuned)
    
    # Evaluate separately for men's and women's games
    men_idx = gender_val == 0
    women_idx = gender_val == 1
    
    men_brier_tuned = brier_score_loss(y_val[men_idx], y_val_pred_tuned[men_idx]) if sum(men_idx) > 0 else np.nan
    women_brier_tuned = brier_score_loss(y_val[women_idx], y_val_pred_tuned[women_idx]) if sum(women_idx) > 0 else np.nan
    
    log_message(f"Validation Brier score with tuned model: {brier_tuned:.4f}")
    log_message(f"Validation AUC with tuned model: {auc_tuned:.4f}")
    log_message(f"Men's Brier score: {men_brier_tuned:.4f}")
    log_message(f"Women's Brier score: {women_brier_tuned:.4f}")
    
    # Save the tuned model
    joblib.dump(tuned_model, os.path.join(OUTPUT_DIR, f'{best_model_name.replace(" ", "_").lower()}_tuned.pkl'))
    log_message(f"Tuned model saved to output directory")
    
else:
    log_message(f"No parameter grid defined for {best_model_name}. Skipping hyperparameter tuning.")
    tuned_model = best_model
    brier_tuned = results[best_model_name]['metrics']['val_brier']

#############################################################
# STEP 8: ANALYZE FEATURE IMPORTANCE
# Goal: Understand which factors are most important in 
# predicting game outcomes.
#############################################################

log_message("\n8. ANALYZING FEATURE IMPORTANCE...")

# Extract feature importance if available
if hasattr(tuned_model, 'feature_importances_'):
    feature_imp = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': tuned_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    log_message("Top 15 most important features:")
    for i, (feature, importance) in enumerate(zip(feature_imp['Feature'][:15], feature_imp['Importance'][:15])):
        log_message(f"{i+1}. {feature}: {importance:.4f}")
    
    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_imp.head(15))
    plt.title(f'Feature Importance ({best_model_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
    plt.close()
    
    # Save feature importance to CSV
    feature_imp.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)
    
elif hasattr(tuned_model, 'coef_'):
    feature_imp = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': tuned_model.coef_[0]
    })
    
    # Add absolute coefficient for sorting
    feature_imp['AbsCoefficient'] = np.abs(feature_imp['Coefficient'])
    feature_imp = feature_imp.sort_values('AbsCoefficient', ascending=False)
    
    log_message("Top 15 features by coefficient magnitude:")
    for i, row in feature_imp.head(15).iterrows():
        log_message(f"{i+1}. {row['Feature']}: {row['Coefficient']:.4f}")
    
    # Visualize coefficients
    plt.figure(figsize=(12, 8))
    top_features = feature_imp.head(15).copy()
    colors = ['blue' if c > 0 else 'red' for c in top_features['Coefficient']]
    sns.barplot(x='AbsCoefficient', y='Feature', data=top_features, palette=colors)
    plt.title(f'Feature Coefficients (Logistic Regression)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_coefficients.png'))
    plt.close()
    
    # Save feature importance to CSV
    feature_imp.to_csv(os.path.join(OUTPUT_DIR, 'feature_coefficients.csv'), index=False)

#############################################################
# STEP 9: TRAIN FINAL MODEL
# Goal: Train the final model on all available data for the
# best possible predictions.
#############################################################

log_message("\n9. TRAINING FINAL MODEL ON ALL DATA...")

# Scale all data
X_all_scaled = scaler.fit_transform(X_model)

# Train the final model
final_model = clone(tuned_model)
final_model.fit(X_all_scaled, y)

# Save the final model
joblib.dump(final_model, os.path.join(OUTPUT_DIR, 'final_model.pkl'))
log_message("Final model saved to output directory")

#############################################################
# STEP 10: GENERATE TOURNAMENT PREDICTIONS
# Goal: Create predictions for all possible matchups in the
# 2025 tournament using our final model.
#############################################################

log_message("\n10. GENERATING TOURNAMENT PREDICTIONS...")

# Prepare prediction features
submission_features, submission_gender = create_prediction_features(
    data['sample_submission'], m_team_stats, w_team_stats
)

# Remove ID and scale features
ids = submission_features['ID'].copy()
X_submission = submission_features.drop(['ID', 'Season'], axis=1, errors='ignore')

# Fill any missing values
X_submission = X_submission.fillna(0)

# Ensure all features are present and in the correct order
all_features = set(X_model.columns)
submission_features = set(X_submission.columns)

missing_cols = all_features - submission_features
extra_cols = submission_features - all_features

if missing_cols:
    log_message(f"Adding {len(missing_cols)} missing columns to submission features")
    for col in missing_cols:
        X_submission[col] = 0

if extra_cols:
    log_message(f"Removing {len(extra_cols)} extra columns from submission features")
    X_submission = X_submission.drop(extra_cols, axis=1)

# Ensure column order matches training data
X_submission = X_submission[X_model.columns]

# Scale features
X_submission_scaled = scaler.transform(X_submission)

# Predict probabilities
predictions = final_model.predict_proba(X_submission_scaled)[:, 1]

# Create submission file
submission = pd.DataFrame({
    'ID': ids,
    'Pred': predictions
})

# Save submission
submission_path = os.path.join(OUTPUT_DIR, 'march_madness_2025_predictions.csv')
submission.to_csv(submission_path, index=False)

log_message(f"\nSubmission file generated: {submission_path}")
log_message(f"Predictions summary: Min={predictions.min():.4f}, Max={predictions.max():.4f}, Mean={predictions.mean():.4f}")

#############################################################
# STEP 11: EXPLORE GENDER-SPECIFIC MODELS (OPTIONAL)
# Goal: Test if separate models for men's and women's games 
# perform better than a single combined model.
#############################################################

log_message("\n11. EXPLORING GENDER-SPECIFIC MODELS...")

# Split training data by gender
X_train_men = X_train_scaled[gender_train == 0]
y_train_men = y_train[gender_train == 0]
X_val_men = X_val_scaled[gender_val == 0]
y_val_men = y_val[gender_val == 0]

X_train_women = X_train_scaled[gender_train == 1]
y_train_women = y_train[gender_train == 1]
X_val_women = X_val_scaled[gender_val == 1]
y_val_women = y_val[gender_val == 1]

log_message(f"Men's training data: {X_train_men.shape[0]} samples")
log_message(f"Women's training data: {X_train_women.shape[0]} samples")

# Train men's model
men_model = clone(tuned_model)
men_model.fit(X_train_men, y_train_men)

# Evaluate men's model
y_val_men_pred = men_model.predict_proba(X_val_men)[:, 1]
men_brier = brier_score_loss(y_val_men, y_val_men_pred)
log_message(f"Men's model validation Brier score: {men_brier:.4f}")

# Train women's model
women_model = clone(tuned_model)
women_model.fit(X_train_women, y_train_women)

# Evaluate women's model
y_val_women_pred = women_model.predict_proba(X_val_women)[:, 1]
women_brier = brier_score_loss(y_val_women, y_val_women_pred)
log_message(f"Women's model validation Brier score: {women_brier:.4f}")

# Compare with combined model
y_val_pred_combined = tuned_model.predict_proba(X_val_scaled)[:, 1]
men_brier_combined = brier_score_loss(y_val_men, y_val_pred_combined[gender_val == 0])
women_brier_combined = brier_score_loss(y_val_women, y_val_pred_combined[gender_val == 1])

log_message(f"Combined model men's Brier score: {men_brier_combined:.4f}")
log_message(f"Combined model women's Brier score: {women_brier_combined:.4f}")

# Determine if separate models are better
men_improved = men_brier < men_brier_combined
women_improved = women_brier < women_brier_combined

log_message(f"Men's model improved: {men_improved} ({men_brier:.4f} vs {men_brier_combined:.4f})")
log_message(f"Women's model improved: {women_improved} ({women_brier:.4f} vs {women_brier_combined:.4f})")

if men_improved or women_improved:
    log_message("Using separate models for final predictions...")
    
    # Save models
    joblib.dump(men_model, os.path.join(OUTPUT_DIR, 'mens_model.pkl'))
    joblib.dump(women_model, os.path.join(OUTPUT_DIR, 'womens_model.pkl'))
    
    # Make predictions using separate models
    men_indices = submission_gender == 0
    women_indices = submission_gender == 1
    
    # Initialize predictions array
    separate_predictions = np.zeros(len(X_submission_scaled))
    
    # Predict men's games
    if sum(men_indices) > 0:
        separate_predictions[men_indices] = men_model.predict_proba(X_submission_scaled[men_indices])[:, 1]
    
    # Predict women's games
    if sum(women_indices) > 0:
        separate_predictions[women_indices] = women_model.predict_proba(X_submission_scaled[women_indices])[:, 1]
    
    # Create submission file
    separate_submission = pd.DataFrame({
        'ID': ids,
        'Pred': separate_predictions
    })
    
    # Save submission
    separate_path = os.path.join(OUTPUT_DIR, 'march_madness_2025_separate_models.csv')
    separate_submission.to_csv(separate_path, index=False)
    
    log_message(f"Separate models submission file generated: {separate_path}")
    log_message(f"Separate predictions summary: Min={separate_predictions.min():.4f}, Max={separate_predictions.max():.4f}, Mean={separate_predictions.mean():.4f}")

#############################################################
# STEP 12: CREATE ENSEMBLE MODEL (OPTIONAL)
# Goal: Combine multiple models into an ensemble to potentially
# achieve better performance than any single model alone.
#############################################################

log_message("\n12. CREATING ENSEMBLE MODEL...")# March Madness 2025 Prediction Model
# Full process to create predictions for men's and women's tournaments

#############################################################
# GOAL: This notebook builds a complete prediction model for 
# the 2025 NCAA basketball tournaments. We'll predict the 
# probability that one team beats another for all possible 
# matchups in both men's and women's tournaments. 
# The final output will be a submission file with predictions
# in the format required by the competition.
#############################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import clone
import xgboost as xgb
import os
import warnings
import joblib
from datetime import datetime
from tqdm import tqdm

# Try to import lightgbm, but make it optional
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Will skip LightGBM model.")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
def log_message(message, print_msg=True):
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    if print_msg:
        print(log_entry)
    return log_entry

# Set paths and parameters
DATA_DIR = 'data_dir'
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

#############################################################
# STEP 1: LOAD THE DATA
# Goal: Load all necessary files to get historical game 
# results, team information, and tournament data that will
# be the foundation for our prediction model.
#############################################################

log_message("1. LOADING DATA FILES...")

def load_data():
    """Load all necessary data files."""
    data = {}
    
    # Teams data
    data['m_teams'] = pd.read_csv(os.path.join(DATA_DIR, 'MTeams.csv'))
    data['w_teams'] = pd.read_csv(os.path.join(DATA_DIR, 'WTeams.csv'))
    
    # Regular season results
    data['m_reg_season'] = pd.read_csv(os.path.join(DATA_DIR, 'MRegularSeasonCompactResults.csv'))
    data['w_reg_season'] = pd.read_csv(os.path.join(DATA_DIR, 'WRegularSeasonCompactResults.csv'))
    
    # Tournament results
    data['m_tourney'] = pd.read_csv(os.path.join(DATA_DIR, 'MNCAATourneyCompactResults.csv'))
    data['w_tourney'] = pd.read_csv(os.path.join(DATA_DIR, 'WNCAATourneyCompactResults.csv'))
    
    # Tournament seeds
    data['m_seeds'] = pd.read_csv(os.path.join(DATA_DIR, 'MNCAATourneySeeds.csv'))
    data['w_seeds'] = pd.read_csv(os.path.join(DATA_DIR, 'WNCAATourneySeeds.csv'))
    
    # Team conferences
    data['m_conferences'] = pd.read_csv(os.path.join(DATA_DIR, 'MTeamConferences.csv'))
    data['w_conferences'] = pd.read_csv(os.path.join(DATA_DIR, 'WTeamConferences.csv'))
    
    # Detailed game statistics
    data['m_reg_detail'] = pd.read_csv(os.path.join(DATA_DIR, 'MRegularSeasonDetailedResults.csv'))
    data['w_reg_detail'] = pd.read_csv(os.path.join(DATA_DIR, 'WRegularSeasonDetailedResults.csv'))
    
    try:
        # Massey Ordinals (team rankings) - only for men's teams
        data['m_massey'] = pd.read_csv(os.path.join(DATA_DIR, 'MMasseyOrdinals.csv'))
    except FileNotFoundError:
        log_message("Warning: MMasseyOrdinals.csv not found. Continuing without rankings data.")
        data['m_massey'] = None
    
    # Sample submission file
    data['sample_submission'] = pd.read_csv(os.path.join(DATA_DIR, 'SampleSubmissionStage2.csv'))
    
    return data

# Load all data files
data = load_data()

# Display team counts
log_message(f"Men's teams: {len(data['m_teams'])}")
log_message(f"Women's teams: {len(data['w_teams'])}")

#############################################################
# STEP 2: DEFINE DATA PROCESSING FUNCTIONS
# Goal: Create utility functions that will transform the raw
# game data into useful features for our prediction model.
#############################################################

log_message("\n2. DEFINING DATA PROCESSING FUNCTIONS...")

def extract_seed_number(seed):
    """Extract numeric seed value from the seed string."""
    # Remove region and play-in indicators
    if seed and len(seed) > 2:
        return int(seed[1:3])
    return None

def create_season_team_stats(games_df, team_id_col='TeamID', opponent_id_col='OpponentID'):
    """
    Create season-level summary statistics for each team based on their game performance.
    
    Args:
        games_df: DataFrame containing game results
        team_id_col: Column name for the team ID
        opponent_id_col: Column name for the opponent ID
        
    Returns:
        DataFrame with season-level team statistics
    """
    # List to store team season stats
    team_seasons = []
    
    # Get unique seasons
    seasons = games_df['Season'].unique()
    
    for season in tqdm(seasons, desc="Processing seasons"):
        season_games = games_df[games_df['Season'] == season]
        
        # Get unique teams in this season
        teams = pd.unique(np.concatenate([
            season_games['WTeamID'].values, 
            season_games['LTeamID'].values
        ]))
        
        for team_id in teams:
            # Games where team won
            w_games = season_games[season_games['WTeamID'] == team_id]
            # Games where team lost
            l_games = season_games[season_games['LTeamID'] == team_id]
            
            # Calculate basic stats
            wins = len(w_games)
            losses = len(l_games)
            total_games = wins + losses
            
            if total_games == 0:
                continue  # Skip if no games played
                
            win_pct = wins / total_games
            
            # Points scored and allowed
            points_scored = w_games['WScore'].sum() + l_games['LScore'].sum()
            points_allowed = w_games['LScore'].sum() + l_games['WScore'].sum()
            
            # Average points per game
            avg_points_scored = points_scored / total_games
            avg_points_allowed = points_allowed / total_games
            
            # Calculate point differential
            point_diff = avg_points_scored - avg_points_allowed
            
            # Home, away, neutral record
            home_wins = len(w_games[w_games['WLoc'] == 'H'])
            home_losses = len(l_games[l_games['WLoc'] == 'A'])
            home_total = home_wins + home_losses
            home_win_pct = home_wins / home_total if home_total > 0 else 0
            
            away_wins = len(w_games[w_games['WLoc'] == 'A'])
            away_losses = len(l_games[l_games['WLoc'] == 'H'])
            away_total = away_wins + away_losses
            away_win_pct = away_wins / away_total if away_total > 0 else 0
            
            neutral_wins = len(w_games[w_games['WLoc'] == 'N'])
            neutral_losses = len(l_games[l_games['WLoc'] == 'N'])
            neutral_total = neutral_wins + neutral_losses
            neutral_win_pct = neutral_wins / neutral_total if neutral_total > 0 else 0
            
            # OT performance
            ot_games = len(w_games[w_games['NumOT'] > 0]) + len(l_games[l_games['NumOT'] > 0])
            ot_wins = len(w_games[w_games['NumOT'] > 0])
            ot_win_pct = ot_wins / ot_games if ot_games > 0 else 0
            
            # Close game performance (decided by 5 or fewer points)
            close_wins = len(w_games[w_games['WScore'] - w_games['LScore'] <= 5])
            close_losses = len(l_games[l_games['WScore'] - l_games['LScore'] <= 5])
            close_games = close_wins + close_losses
            close_win_pct = close_wins / close_games if close_games > 0 else 0
            
            # Blowout performance (decided by 15 or more points)
            blowout_wins = len(w_games[w_games['WScore'] - w_games['LScore'] >= 15])
            blowout_losses = len(l_games[l_games['WScore'] - l_games['LScore'] >= 15])
            blowout_win_pct = blowout_wins / (blowout_wins + blowout_losses) if (blowout_wins + blowout_losses) > 0 else 0
            
            # Recent form (last 5 games)
            recent_games = pd.concat([
                w_games[['Season', 'DayNum', 'WTeamID']].assign(Result=1),
                l_games[['Season', 'DayNum', 'LTeamID']].rename(columns={'LTeamID': 'WTeamID'}).assign(Result=0)
            ]).sort_values('DayNum', ascending=False).head(5)
            
            recent_win_pct = recent_games['Result'].mean() if len(recent_games) > 0 else win_pct
            
            # Win/Loss streaks
            if len(recent_games) > 0:
                current_streak = 0
                for result in recent_games['Result'].values:
                    if result == 1 and current_streak >= 0:
                        current_streak += 1
                    elif result == 1 and current_streak < 0:
                        current_streak = 1
                    elif result == 0 and current_streak <= 0:
                        current_streak -= 1
                    elif result == 0 and current_streak > 0:
                        current_streak = -1
            else:
                current_streak = 0
            
            # Add results to list
            team_seasons.append({
                'Season': season,
                'TeamID': team_id,
                'TotalGames': total_games,
                'Wins': wins,
                'Losses': losses,
                'WinPct': win_pct,
                'AvgPointsScored': avg_points_scored,
                'AvgPointsAllowed': avg_points_allowed,
                'PointDifferential': point_diff,
                'HomeWinPct': home_win_pct,
                'AwayWinPct': away_win_pct,
                'NeutralWinPct': neutral_win_pct,
                'OTWinPct': ot_win_pct,
                'CloseWinPct': close_win_pct,
                'BlowoutWinPct': blowout_win_pct,
                'RecentWinPct': recent_win_pct,
                'CurrentStreak': current_streak
            })
    
    return pd.DataFrame(team_seasons)

def create_advanced_team_stats(detailed_games_df):
    """
    Create advanced stats for each team based on detailed game data.
    
    Args:
        detailed_games_df: DataFrame containing detailed game results
        
    Returns:
        DataFrame with advanced team statistics
    """
    # List to store team season stats
    advanced_stats = []
    
    # Get unique seasons
    seasons = detailed_games_df['Season'].unique()
    
    for season in tqdm(seasons, desc="Processing advanced stats"):
        season_games = detailed_games_df[detailed_games_df['Season'] == season]
        
        # Get unique teams in this season
        teams = pd.unique(np.concatenate([
            season_games['WTeamID'].values, 
            season_games['LTeamID'].values
        ]))
        
        for team_id in teams:
            # Games where team won
            w_games = season_games[season_games['WTeamID'] == team_id]
            # Games where team lost
            l_games = season_games[season_games['LTeamID'] == team_id]
            
            # Calculate totals
            total_games = len(w_games) + len(l_games)
            
            if total_games == 0:
                continue  # Skip if no games
                
            # Offensive statistics (when team won)
            w_fgm = w_games['WFGM'].sum()
            w_fga = w_games['WFGA'].sum()
            w_fgm3 = w_games['WFGM3'].sum()
            w_fga3 = w_games['WFGA3'].sum()
            w_ftm = w_games['WFTM'].sum()
            w_fta = w_games['WFTA'].sum()
            w_or = w_games['WOR'].sum()
            w_dr = w_games['WDR'].sum()
            w_ast = w_games['WAst'].sum()
            w_to = w_games['WTO'].sum()
            w_stl = w_games['WStl'].sum()
            w_blk = w_games['WBlk'].sum()
            w_pf = w_games['WPF'].sum()
            
            # Offensive statistics (when team lost)
            l_fgm = l_games['LFGM'].sum()
            l_fga = l_games['LFGA'].sum()
            l_fgm3 = l_games['LFGM3'].sum()
            l_fga3 = l_games['LFGA3'].sum()
            l_ftm = l_games['LFTM'].sum()
            l_fta = l_games['LFTA'].sum()
            l_or = l_games['LOR'].sum()
            l_dr = l_games['LDR'].sum()
            l_ast = l_games['LAst'].sum()
            l_to = l_games['LTO'].sum()
            l_stl = l_games['LStl'].sum()
            l_blk = l_games['LBlk'].sum()
            l_pf = l_games['LPF'].sum()
            
            # Defensive statistics (when team won)
            def_w_fgm = w_games['LFGM'].sum()
            def_w_fga = w_games['LFGA'].sum()
            def_w_fgm3 = w_games['LFGM3'].sum()
            def_w_fga3 = w_games['LFGA3'].sum()
            def_w_ftm = w_games['LFTM'].sum()
            def_w_fta = w_games['LFTA'].sum()
            
            # Defensive statistics (when team lost)
            def_l_fgm = l_games['WFGM'].sum()
            def_l_fga = l_games['WFGA'].sum()
            def_l_fgm3 = l_games['WFGM3'].sum()
            def_l_fga3 = l_games['WFGA3'].sum()
            def_l_ftm = l_games['WFTM'].sum()
            def_l_fta = l_games['WFTA'].sum()
            
            # Combine stats
            fgm = w_fgm + l_fgm
            fga = w_fga + l_fga
            fgm3 = w_fgm3 + l_fgm3
            fga3 = w_fga3 + l_fga3
            ftm = w_ftm + l_ftm
            fta = w_fta + l_fta
            orb = w_or + l_or  # offensive rebounds
            drb = w_dr + l_dr  # defensive rebounds
            ast = w_ast + l_ast
            to = w_to + l_to
            stl = w_stl + l_stl
            blk = w_blk + l_blk
            pf = w_pf + l_pf
            
            # Defensive stats allowed
            def_fgm = def_w_fgm + def_l_fgm
            def_fga = def_w_fga + def_l_fga
            def_fgm3 = def_w_fgm3 + def_l_fgm3
            def_fga3 = def_w_fga3 + def_l_fga3
            def_ftm = def_w_ftm + def_l_ftm
            def_fta = def_w_fta + def_l_fta
            
            # Advanced stats calculation
            # Shooting percentages
            fg_pct = fgm / fga if fga > 0 else 0
            fg3_pct = fgm3 / fga3 if fga3 > 0 else 0
            ft_pct = ftm / fta if fta > 0 else 0
            effective_fg_pct = (fgm + 0.5 * fgm3) / fga if fga > 0 else 0
            
            # Defensive shooting percentages allowed
            def_fg_pct = def_fgm / def_fga if def_fga > 0 else 0
            def_fg3_pct = def_fgm3 / def_fga3 if def_fga3 > 0 else 0
            def_ft_pct = def_ftm / def_fta if def_fta > 0 else 0
            def_effective_fg_pct = (def_fgm + 0.5 * def_fgm3) / def_fga if def_fga > 0 else 0
            
            # Rebounding
            total_reb = orb + drb
            reb_per_game = total_reb / total_games
            orb_per_game = orb / total_games
            drb_per_game = drb / total_games
            
            # Assist to turnover ratio
            ast_to_ratio = ast / to if to > 0 else ast
            
            # Per game averages
            ast_per_game = ast / total_games
            to_per_game = to / total_games
            stl_per_game = stl / total_games
            blk_per_game = blk / total_games
            pf_per_game = pf / total_games
            
            # Offensive efficiency (points per 100 possessions)
            # Approximating possessions using FGA - OR + TO + (0.44 * FTA)
            poss = fga - orb + to + (0.44 * fta)
            points = (2 * (fgm - fgm3)) + (3 * fgm3) + ftm
            off_efficiency = 100 * points / poss if poss > 0 else 0
            
            # Defensive efficiency (points allowed per 100 possessions)
            def_poss = def_fga - orb + to + (0.44 * def_fta)
            def_points = (2 * (def_fgm - def_fgm3)) + (3 * def_fgm3) + def_ftm
            def_efficiency = 100 * def_points / def_poss if def_poss > 0 else 0
            
            # Net Rating (Offensive Efficiency - Defensive Efficiency)
            net_rating = off_efficiency - def_efficiency
            
            # True shooting percentage
            ts_pct = points / (2 * (fga + 0.44 * fta)) if (fga + 0.44 * fta) > 0 else 0
            
            # Add results to list
            advanced_stats.append({
                'Season': season,
                'TeamID': team_id,
                'FGPct': fg_pct,
                'FG3Pct': fg3_pct,
                'FTPct': ft_pct,
                'eFGPct': effective_fg_pct,
                'TSPct': ts_pct,
                'DefFGPct': def_fg_pct,
                'DefFG3Pct': def_fg3_pct,
                'DefFTPct': def_ft_pct,
                'DefEFGPct': def_effective_fg_pct,
                'RebPerGame': reb_per_game,
                'ORebPerGame': orb_per_game,
                'DRebPerGame': drb_per_game,
                'AstToRatio': ast_to_ratio,
                'AstPerGame': ast_per_game,
                'TOPerGame': to_per_game,
                'StlPerGame': stl_per_game,
                'BlkPerGame': blk_per_game,
                'PFPerGame': pf_per_game,
                'OffEfficiency': off_efficiency,
                'DefEfficiency': def_efficiency,
                'NetRating': net_rating
            })
    
    return pd.DataFrame(advanced_stats)

def get_massey_rankings(massey_df, ranking_systems=None):
    """
    Get team rankings from Massey Ordinals for multiple ranking systems.
    
    Args:
        massey_df: DataFrame containing Massey Ordinals
        ranking_systems: List of ranking systems to include
        
    Returns:
        DataFrame with team rankings for each season
    """
    if massey_df is None:
        return pd.DataFrame(columns=['Season', 'TeamID'])
        
    if ranking_systems is None:
        ranking_systems = ['POM', 'SAG', 'MOR', 'DOL', 'RPI']
    
    # Filter for specified systems and tournament time
    dfs = []
    
    for system in ranking_systems:
        system_df = massey_df[massey_df['SystemName'] == system].copy()
        if len(system_df) == 0:
            log_message(f"Warning: Ranking system {system} not found in Massey data")
            continue
            
        # Get the latest ranking day for each season (closest to tournament)
        max_days = system_df.groupby('Season')['RankingDayNum'].max().reset_index()
        max_days.rename(columns={'RankingDayNum': 'MaxRankingDay'}, inplace=True)
        
        system_df = system_df.merge(max_days, on='Season')
        system_df = system_df[system_df['RankingDayNum'] == system_df['MaxRankingDay']]
        
        # Keep only essential columns and rename
        system_df = system_df[['Season', 'TeamID', 'OrdinalRank']]
        system_df.rename(columns={'OrdinalRank': f'{system}Rank'}, inplace=True)
        
        dfs.append(system_df)
    
    # Merge all ranking systems
    if not dfs:
        return pd.DataFrame(columns=['Season', 'TeamID'])
        
    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on=['Season', 'TeamID'], how='outer')
    
    return result

def prepare_tournament_seeds(seeds_df):
    """
    Process tournament seeds.
    
    Args:
        seeds_df: DataFrame containing tournament seeds
        
    Returns:
        DataFrame with numeric seed values
    """
    seeds_numeric = seeds_df.copy()
    seeds_numeric['SeedValue'] = seeds_numeric['Seed'].apply(extract_seed_number)
    return seeds_numeric[['Season', 'TeamID', 'SeedValue']]

def add_conference_strength(team_stats_df, conferences_df, games_df):
    """
    Add conference strength metrics to team stats.
    
    Args:
        team_stats_df: DataFrame with team stats
        conferences_df: DataFrame with team conference affiliations
        games_df: DataFrame with game results
        
    Returns:
        DataFrame with conference strength metrics added
    """
    # Merge team stats with conference info
    merged_stats = team_stats_df.merge(
        conferences_df[['Season', 'TeamID', 'ConfAbbrev']], 
        on=['Season', 'TeamID'], 
        how='left'
    )
    
    # Calculate conference win percentages
    conf_stats = []
    
    for season in merged_stats['Season'].unique():
        season_games = games_df[games_df['Season'] == season]
        season_conferences = conferences_df[conferences_df['Season'] == season]
        
        # Add conference info to games
        games_with_conf = season_games.merge(
            season_conferences[['TeamID', 'ConfAbbrev']],
            left_on='WTeamID',
            right_on='TeamID',
            how='left'
        ).drop('TeamID', axis=1).rename(columns={'ConfAbbrev': 'WConf'})
        
        games_with_conf = games_with_conf.merge(
            season_conferences[['TeamID', 'ConfAbbrev']],
            left_on='LTeamID',
            right_on='TeamID',
            how='left'
        ).drop('TeamID', axis=1).rename(columns={'ConfAbbrev': 'LConf'})
        
        # Calculate non-conference win percentage for each conference
        for conf in season_conferences['ConfAbbrev'].unique():
            # Games where this conference won against another conference
            non_conf_wins = games_with_conf[
                (games_with_conf['WConf'] == conf) & 
                (games_with_conf['LConf'] != conf)
            ]
            
            # Games where this conference lost against another conference
            non_conf_losses = games_with_conf[
                (games_with_conf['LConf'] == conf) & 
                (games_with_conf['WConf'] != conf)
            ]
            
            non_conf_games = len(non_conf_wins) + len(non_conf_losses)
            conf_win_pct = len(non_conf_wins) / non_conf_games if non_conf_games > 0 else 0
            
            conf_stats.append({
                'Season': season,
                'ConfAbbrev': conf,
                'NonConfGames': non_conf_games,
                'NonConfWinPct': conf_win_pct
            })
    
    # Convert to DataFrame
    conf_strength_df = pd.DataFrame(conf_stats)
    
    # Merge conference strength back to team stats
    team_stats_with_conf = merged_stats.merge(
        conf_strength_df,
        on=['Season', 'ConfAbbrev'],
        how='left'
    )
    
    return team_stats_with_conf

def create_matchup_features(team1_stats, team2_stats):
    """
    Create features for a matchup between two teams.
    
    Args:
        team1_stats: Dictionary or Series of stats for the first team
        team2_stats: Dictionary or Series of stats for the second team
        
    Returns:
        Dictionary of matchup features
    """
    features = {}
    
    # Handle different input types
    if isinstance(team1_stats, pd.Series):
        team1_dict = team1_stats.to_dict()
        team2_dict = team2_stats.to_dict()
    else:
        team1_dict = team1_stats
        team2_dict = team2_stats
    
    # List of stats to compare
    basic_stats = [
        'WinPct', 'AvgPointsScored', 'AvgPointsAllowed', 'PointDifferential',
        'HomeWinPct', 'AwayWinPct', 'NeutralWinPct', 'OTWinPct',
        'CloseWinPct', 'BlowoutWinPct', 'RecentWinPct', 'CurrentStreak'
    ]
    
    # Advanced stats to compare
    advanced_stats = [
        'FGPct', 'FG3Pct', 'FTPct', 'eFGPct', 'TSPct',
        'DefFGPct', 'DefFG3Pct', 'DefFTPct', 'DefEFGPct',
        'RebPerGame', 'ORebPerGame', 'DRebPerGame', 'AstToRatio',
        'AstPerGame', 'TOPerGame', 'StlPerGame', 'BlkPerGame',
        'OffEfficiency', 'DefEfficiency', 'NetRating'
    ]
    
    # Conference strength
    conf_stats = ['NonConfWinPct']
    
    # Ranking stats (note: for rankings, lower is better)
    ranking_systems = ['POM', 'SAG', 'MOR', 'DOL', 'RPI']
    ranking_stats = [f'{sys}Rank' for sys in ranking_systems]
    
    # Add raw values for basic stats
    for stat in basic_stats:
        if stat in team1_dict and stat in team2_dict:
            features[f'Team1_{stat}'] = team1_dict[stat]
            features[f'Team2_{stat}'] = team2_dict[stat]
    
    # Add raw values for advanced stats
    for stat in advanced_stats:
        if stat in team1_dict and stat in team2_dict:
            features[f'Team1_{stat}'] = team1_dict[stat]
            features[f'Team2_{stat}'] = team2_dict[stat]
    
    # Add raw values for conference stats
    for stat in conf_stats:
        if stat in team1_dict and stat in team2_dict:
            features[f'Team1_{stat}'] = team1_dict[stat]
            features[f'Team2_{stat}'] = team2_dict[stat]
    
    # Add differences (team1 - team2)
    for stat in basic_stats:
        if stat in team1_dict and stat in team2_dict:
            features[f'{stat}_Diff'] = team1_dict[stat] - team2_dict[stat]
    
    for stat in advanced_stats:
        if stat in team1_dict and stat in team2_dict:
            features[f'{stat}_Diff'] = team1_dict[stat] - team2_dict[stat]
    
    for stat in conf_stats:
        if stat in team1_dict and stat in team2_dict:
            features[f'{stat}_Diff'] = team1_dict[stat] - team2_dict[stat]
    
    # Add ratios (team1 / team2) for key stats
    ratio_stats = ['WinPct', 'PointDifferential', 'OffEfficiency', 'DefEfficiency', 'NetRating']
    for stat in ratio_stats:
        if stat in team1_dict and stat in team2_dict:
            # Safely handle division by zero or negative values
            if stat in team1_dict and stat in team2_dict:
                t1_val = team1_dict[stat]
                t2_val = team2_dict[stat]
                
                # Skip if either value is missing
                if pd.isna(t1_val) or pd.isna(t2_val):
                    continue
                
                # Handle division and negative values safely
                if t1_val < 0 and t2_val < 0:
                    # Both negative, use ratio of absolute values, but invert
                    if abs(t1_val) > 0:
                        features[f'{stat}_Ratio'] = abs(t2_val) / abs(t1_val)
                    else:
                        features[f'{stat}_Ratio'] = 1.0  # Equal if both zero
                elif t1_val >= 0 and t2_val >= 0:
                    # Both positive
                    if t2_val > 0:
                        features[f'{stat}_Ratio'] = t1_val / t2_val
                    else:
                        features[f'{stat}_Ratio'] = 2.0 if t1_val > 0 else 1.0  # Arbitrary value when dividing by zero
                else:
                    # Mixed signs - use difference instead
                    features[f'{stat}_Ratio'] = t1_val - t2_val
    
    # Add rankings (note: for rankings, team2_rank - team1_rank is positive if team1 is better)
    for stat in ranking_stats:
        if stat in team1_dict and stat in team2_dict:
            # Only add if both values are not NaN
            if not pd.isna(team1_dict[stat]) and not pd.isna(team2_dict[stat]):
                features[f'{stat}_Raw1'] = team1_dict[stat]
                features[f'{stat}_Raw2'] = team2_dict[stat]
                features[f'{stat}_Diff'] = team2_dict[stat] - team1_dict[stat]  # Positive if team1 is better ranked
    
    # Add seed information if available
    if 'SeedValue' in team1_dict and 'SeedValue' in team2_dict:
        # Only add if both values are not NaN
        if not pd.isna(team1_dict['SeedValue']) and not pd.isna(team2_dict['SeedValue']):
            features['SeedValue_1'] = team1_dict['SeedValue']
            features['SeedValue_2'] = team2_dict['SeedValue']
            features['SeedValue_Diff'] = team1_dict['SeedValue'] - team2_dict['SeedValue']
            
            # Historical seed matchup performance (e.g., 5 vs 12 upsets)
            seed_gap = abs(team1_dict['SeedValue'] - team2_dict['SeedValue'])
            features['Seed_Gap'] = seed_gap
            
            # Is this a potential upset game? (lower seed vs higher seed)
            is_upset_scenario = 1 if team1_dict['SeedValue'] > team2_dict['SeedValue'] else 0
            features['Upset_Scenario'] = is_upset_scenario
    
    return features

def prepare_training_data(m_games, w_games, m_team_stats, w_team_stats, min_season=2010):
    """
    Prepare training data for model fitting.
    
    Args:
        m_games: Men's games DataFrame
        w_games: Women's games DataFrame
        m_team_stats: Men's team stats DataFrame
        w_team_stats: Women's team stats DataFrame
        min_season: Minimum season to include in training
        
    Returns:
        X: Feature matrix
        y: Target vector
        teams: Indicator for men's (0) or women's (1) game
    """
    # Lists to store features and targets
    X_features = []
    y_target = []
    gender_indicator = []  # 0 for men, 1 for women
    
    # Process men's games
    log_message("Processing men's games for training data...")
    for _, game in tqdm(m_games.iterrows(), total=len(m_games)):
        season = game['Season']
        
        # Skip older seasons
        if season < min_season:
            continue
            
        w_team = game['WTeamID']
        l_team = game['LTeamID']
        
        # Get team stats for this season
        w_team_stats_df = m_team_stats[
            (m_team_stats['Season'] == season) & 
            (m_team_stats['TeamID'] == w_team)
        ]
        
        l_team_stats_df = m_team_stats[
            (m_team_stats['Season'] == season) & 
            (m_team_stats['TeamID'] == l_team)
        ]
        
        # Skip if we don't have stats for either team
        if w_team_stats_df.empty or l_team_stats_df.empty:
            continue
            
        w_team_stats = w_team_stats_df.iloc[0]
        l_team_stats = l_team_stats_df.iloc[0]
        
        # Create matchup features (from winner's perspective)
        try:
            w_features = create_matchup_features(w_team_stats, l_team_stats)
            w_features['Season'] = season
            
            # Add to data
            X_features.append(w_features)
            y_target.append(1)  # 1 indicates the first team won
            gender_indicator.append(0)  # 0 for men's game
            
            # Create matchup features (from loser's perspective)
            l_features = create_matchup_features(l_team_stats, w_team_stats)
            l_features['Season'] = season
            
            # Add to data
            X_features.append(l_features)
            y_target.append(0)  # 0 indicates the first team lost
            gender_indicator.append(0)  # 0 for men's game
        except Exception as e:
            log_message(f"Error processing men's game {w_team} vs {l_team} (season {season}): {e}")
            continue
    
    # Process women's games
    log_message("Processing women's games for training data...")
    for _, game in tqdm(w_games.iterrows(), total=len(w_games)):
        season = game['Season']
        
        # Skip older seasons
        if season < min_season:
            continue
            
        w_team = game['WTeamID']
        l_team = game['LTeamID']
        
        # Get team stats for this season
        w_team_stats_df = w_team_stats[
            (w_team_stats['Season'] == season) & 
            (w_team_stats['TeamID'] == w_team)
        ]
        
        l_team_stats_df = w_team_stats[
            (w_team_stats['Season'] == season) & 
            (w_team_stats['TeamID'] == l_team)
        ]
        
        # Skip if we don't have stats for either team
        if w_team_stats_df.empty or l_team_stats_df.empty:
            continue
            
        w_team_stats = w_team_stats_df.iloc[0]
        l_team_stats = l_team_stats_df.iloc[0]
        
        # Create matchup features (from winner's perspective)
        try:
            w_features = create_matchup_features(w_team_stats, l_team_stats)
            w_features['Season'] = season
            
            # Add to data
            X_features.append(w_features)
            y_target.append(1)  # 1 indicates the first team won
            gender_indicator.append(1)  # 1 for women's game
            
            # Create matchup features (from loser's perspective)
            l_features = create_matchup_features(l_team_stats, w_team_stats)
            l_features['Season'] = season
            
            # Add to data
            X_features.append(l_features)
            y_target.append(0)  # 0 indicates the first team lost
            gender_indicator.append(1)  # 1 for women's game
        except Exception as e:
            log_message(f"Error processing women's game {w_team} vs {l_team} (season {season}): {e}")
            continue
    
    # Convert to DataFrame
    X = pd.DataFrame(X_features)
    y = np.array(y_target)
    gender = np.array(gender_indicator)
    
    return X, y, gender

def create_prediction_features(df_submission, m_team_stats, w_team_stats):
    """
    Create features for matchup predictions in the format of the submission file.
    
    Args:
        df_submission: Submission template DataFrame
        m_team_stats: Men's team stats DataFrame
        w_team_stats: Women's team stats DataFrame
        
    Returns:
        DataFrame with matchup features for predictions
    """
    features_list = []
    gender_indicator = []  # 0 for men, 1 for women
    
    # Get latest season for each dataset
    max_m_season = m_team_stats['Season'].max()
    max_w_season = w_team_stats['Season'].max()
    
    log_message(f"Creating prediction features for {len(df_submission)} matchups...")
    for _, row in tqdm(df_submission.iterrows(), total=len(df_submission)):
        # Parse the ID to get season and teams
        parts = row['ID'].split('_')
        season = int(parts[0])
        team1_id = int(parts[1])  # Lower TeamID
        team2_id = int(parts[2])  # Higher TeamID
        
        # Determine if men's or women's game based on TeamID
        if team1_id < 3000:  # Men's teams are 1000-1999
            team_stats = m_team_stats
            latest_season = max_m_season
            gender = 0  # Men's game
        else:  # Women's teams are 3000-3999
            team_stats = w_team_stats
            latest_season = max_w_season
            gender = 1  # Women's game
        
        # Use current season stats if available, otherwise use latest available
        team1_stat = team_stats[
            (team_stats['Season'] == season) & 
            (team_stats['TeamID'] == team1_id)
        ]
        
        if team1_stat.empty:
            # Try using latest season's stats
            team1_stat = team_stats[
                (team_stats['Season'] == latest_season) & 
                (team_stats['TeamID'] == team1_id)
            ]
        
        team2_stat = team_stats[
            (team_stats['Season'] == season) & 
            (team_stats['TeamID'] == team2_id)
        ]
        
        if team2_stat.empty:
            # Try using latest season's stats
            team2_stat = team_stats[
                (team_stats['Season'] == latest_season) & 
                (team_stats['TeamID'] == team2_id)
            ]
        
        # If we still don't have stats for either team, use average stats
        if team1_stat.empty or team2_stat.empty:
            # Calculate average stats for the gender and latest season
            avg_stats = team_stats[team_stats['Season'] == latest_season].mean(numeric_only=True)
            avg_stats = avg_stats.to_dict()
            avg_stats.update({'Season': latest_season, 'TeamID': -1})
            
            features = {'ID': row['ID'], 'Season': season}
            
            # Add an indicator for missing team data
            features['Missing_Team_Data'] = 1
            
            # Add zeros for all feature differences
            for col in team_stats.columns:
                if col not in ['Season', 'TeamID', 'ConfAbbrev']:
                    features[f'{col}_Diff'] = 0
                    
            features_list.append(features)
            gender_indicator.append(gender)
            continue
        
        # Use first row if multiple matches
        team1_stat = team1_stat.iloc[0]
        team2_stat = team2_stat.iloc[0]
        
        # Create matchup features
        try:
            features = create_matchup_features(team1_stat, team2_stat)
            features['ID'] = row['ID']
            features['Season'] = season
            
            # Add an indicator for complete team data
            features['Missing_Team_Data'] = 0
            
            features_list.append(features)
            gender_indicator.append(gender)
        except Exception as e:
            log_message(f"Error creating features for matchup {row['ID']}: {e}")
            
            # Create basic features with zeros
            features = {'ID': row['ID'], 'Season': season, 'Missing_Team_Data': 1}
            features_list.append(features)
            gender_indicator.append(gender)
    
    features_df = pd.DataFrame(features_list)
    gender_array = np.array(gender_indicator)
    
    return features_df, gender_array

#############################################################
# STEP 3: CREATE TEAM FEATURES
# Goal: Calculate comprehensive team statistics that will 
# serve as the foundation for our prediction model.
#############################################################

log_message("\n3. CREATING TEAM FEATURES...")

# 3a. Men's Teams Features
log_message("\nCreating men's team features...")

# Create basic season stats
log_message("Creating season stats...")
m_team_season_stats = create_season_team_stats(data['m_reg_season'])
log_message(f"Number of team-seasons: {len(m_team_season_stats)}")

# Create advanced stats
log_message("Creating advanced stats...")
m_advanced_stats = create_advanced_team_stats(data['m_reg_detail'])
log_message(f"Number of team-seasons with advanced stats: {len(m_advanced_stats)}")

# Merge basic and advanced stats
m_team_stats = m_team_season_stats.merge(
    m_advanced_stats, 
    on=['Season', 'TeamID'], 
    how='left'
)

# Get tournament seeds
m_numeric_seeds = prepare_tournament_seeds(data['m_seeds'])

# Merge seeds onto stats
m_team_stats = m_team_stats.merge(
    m_numeric_seeds,
    on=['Season', 'TeamID'],
    how='left'
)

# Add conference strength
m_team_stats = add_conference_strength(
    m_team_stats, 
    data['m_conferences'], 
    data['m_reg_season']
)

# Get rankings
log_message("Getting ranking data...")
try:
    # Get rankings from multiple systems
    m_rankings = get_massey_rankings(data['m_massey'])
    
    # Merge with team stats
    if not m_rankings.empty:
        m_team_stats = m_team_stats.merge(m_rankings, on=['Season', 'TeamID'], how='left')
        log_message("Rankings data added successfully")
    else:
        log_message("No rankings data available")
except Exception as e:
    log_message(f"Error getting rankings: {e}")

# 3b. Women's Teams Features
log_message("\nCreating women's team features...")

# Create basic season stats
log_message("Creating season stats...")
w_team_season_stats = create_season_team_stats(data['w_reg_season'])
log_message(f"Number of team-seasons: {len(w_team_season_stats)}")

# Create advanced stats
log_message("Creating advanced stats...")
w_advanced_stats = create_advanced_team_stats(data['w_reg_detail'])
log_message(f"Number of team-seasons with advanced stats: {len(w_advanced_stats)}")

# Merge basic and advanced stats
w_team_stats = w_team_season_stats.merge(
    w_advanced_stats, 
    on=['Season', 'TeamID'], 
    how='left'
)

# Get tournament seeds
w_numeric_seeds = prepare_tournament_seeds(data['w_seeds'])

# Merge seeds onto stats
w_team_stats = w_team_stats.merge(
    w_numeric_seeds,
    on=['Season', 'TeamID'],
    how='left'
)

# Add conference strength
w_team_stats = add_conference_strength(
    w_team_stats, 
    data['w_conferences'], 
    data['w_reg_season']
)

#############################################################
# STEP 4: PREPARE TRAINING DATA
# Goal: Create matchup features from historical games for
# model training.
#############################################################

log_message("\n4. PREPARING TRAINING DATA...")

# Combine regular season and tournament games for training
m_combined_games = pd.concat([data['m_reg_season'], data['m_tourney']])
w_combined_games = pd.concat([data['w_reg_season'], data['w_tourney']])

# Create training data from historical games
X, y, gender = prepare_training_data(
    m_combined_games, 
    w_combined_games,
    m_team_stats, 
    w_team_stats,
    min_season=2010  # Use more recent data for better relevance
)

log_message(f"Training data shape: {X.shape}")
log_message(f"Training label shape: {y.shape}")
log_message(f"Men's games: {np.sum(gender == 0)}, Women's games: {np.sum(gender == 1)}")

# Check features
feature_cols = [col for col in X.columns if col != 'Season' and col != 'ID']
log_message(f"\nNumber of features: {len(feature_cols)}")
log_message(f"Sample features: {feature_cols[:10]}...")

# Handle missing values
log_message("\nChecking for missing values...")
missing_pct = X[feature_cols].isnull().mean() * 100
features_with_missing = missing_pct[missing_pct > 0]

if not features_with_missing.empty:
    log_message("Features with missing values:")
    for feature, pct in features_with_missing.sort_values(ascending=False).items():
        log_message(f"  {feature}: {pct:.2f}%")
    
    # Fill missing values
    X = X.fillna(0)
    log_message("Missing values filled with zeros")
else:
    log_message("No missing values found")

#############################################################
# STEP 5: SPLIT DATA AND PREPARE FOR MODELING
# Goal: Divide data into training and validation sets, and
# scale features for optimal model performance.
#############################################################

log_message("\n5. SPLITTING AND SCALING DATA...")

# Remove non-feature columns
id_col = 'ID' in X.columns
X_model = X.drop(['Season'] + (['ID'] if id_col else []), axis=1, errors='ignore')

# Split data for validation - stratify by both outcome and gender
X_train, X_val, y_train, y_val, gender_train, gender_val = train_test_split(
    X_model, y, gender, test_size=0.2, random_state=RANDOM_SEED, stratify=np.column_stack((y, gender))
)

log_message(f"Training set: {X_train.shape[0]} samples")
log_message(f"Validation set: {X_val.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Save the scaler for later use
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'feature_scaler.pkl'))
log_message("Feature scaler saved to output directory")

#############################################################
# STEP 6: TRAIN MULTIPLE MODELS
# Goal: Train and evaluate different machine learning models
# to identify the most effective approach for predicting
# tournament outcomes.
#############################################################

log_message("\n6. TRAINING MULTIPLE MODELS...")

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000, C=0.1, solver='liblinear', random_state=RANDOM_SEED),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=10, random_state=RANDOM_SEED),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=RANDOM_SEED),
    'XGBoost': xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=RANDOM_SEED)
}

# Add LightGBM if available
if LIGHTGBM_AVAILABLE:
    models['LightGBM'] = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=RANDOM_SEED)
# March Madness Tournament Prediction Model

This notebook provides a comprehensive solution for predicting NCAA March Madness tournament outcomes. It includes:
- Data loading and processing
- Feature engineering
- Model training and evaluation
- Tournament simulation and prediction
- Visualization and analysis of results

## Setup

First, let's import the necessary libraries and set up our environment.

```python
# Import libraries
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import log_loss, roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from tqdm.notebook import tqdm

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Configuration
DATA_DIR = "data/"  # Directory containing NCAA CSV files
MODEL_DIR = "models/"  # Directory for saving models
RESULTS_DIR = "results/"  # Directory for saving results

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

print("Setup complete!")
```

# 1. Data Loading and Processing

Let's define functions to load and process the NCAA basketball data.

```python
def load_data(data_dir):
    """
    Load all necessary datasets from the provided directory.
    """
    # Load teams data
    teams = pd.read_csv(f"{data_dir}/MTeams.csv")
    
    # Load regular season results
    regular_season = pd.read_csv(f"{data_dir}/MRegularSeasonCompactResults.csv")
    detailed_regular_season = pd.read_csv(f"{data_dir}/MRegularSeasonDetailedResults.csv")
    
    # Load tournament data
    tourney_results = pd.read_csv(f"{data_dir}/MNCAATourneyCompactResults.csv")
    tourney_seeds = pd.read_csv(f"{data_dir}/MNCAATourneySeeds.csv")
    
    # Load rankings data
    rankings = pd.read_csv(f"{data_dir}/MMasseyOrdinals.csv")
    
    # Load conference data
    conferences = pd.read_csv(f"{data_dir}/MTeamConferences.csv")
    
    return {
        'teams': teams,
        'regular_season': regular_season,
        'detailed_regular_season': detailed_regular_season,
        'tourney_results': tourney_results,
        'tourney_seeds': tourney_seeds,
        'rankings': rankings,
        'conferences': conferences
    }

# Load the data
try:
    data = load_data(DATA_DIR)
    print("Data loaded successfully!")
    
    # Display basic information about the datasets
    print("\nDataset Information:")
    for name, df in data.items():
        print(f"{name}: {df.shape[0]} rows x {df.shape[1]} columns")
    
    # Show sample of teams data
    print("\nSample of teams data:")
    display(data['teams'].head())
    
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please make sure all required CSV files are in the data directory.")
```

# 2. Feature Engineering

Now let's define functions to extract and engineer features for our prediction model.

```python
def extract_seed_number(seed_string):
    """
    Extract numeric seed (1-16) from seed string like 'W01' or 'X16'
    """
    return int(seed_string[1:3])

def get_team_seed(tourney_seeds, season, team_id):
    """
    Get the seed for a team in a specific season.
    Returns the numeric seed (1-16) or 0 if not found.
    """
    seed_row = tourney_seeds[(tourney_seeds['Season'] == season) & 
                            (tourney_seeds['TeamID'] == team_id)]
    
    if len(seed_row) == 0:
        return 0
    
    seed_string = seed_row.iloc[0]['Seed']
    return extract_seed_number(seed_string)

def get_team_rankings(rankings, season, team_id, day_num=133):
    """
    Get the rankings for a team before the tournament in a specific season.
    Returns a dictionary of rankings from different systems.
    """
    # Filter for the given season, team, and before tournament
    team_rankings = rankings[(rankings['Season'] == season) & 
                           (rankings['TeamID'] == team_id) &
                           (rankings['RankingDayNum'] < day_num)]
    
    if len(team_rankings) == 0:
        return {'avg_rank': 0}
    
    # Get the latest rankings before the tournament
    max_day = team_rankings['RankingDayNum'].max()
    latest_rankings = team_rankings[team_rankings['RankingDayNum'] == max_day]
    
    # Calculate average ranking
    avg_rank = latest_rankings['OrdinalRank'].mean()
    
    # Prepare the result dictionary
    result = {'avg_rank': avg_rank}
    
    # Add rankings from specific systems if available
    for system in ['POM', 'SAG', 'MOR', 'DOK', 'RPI']:
        system_rank = latest_rankings[latest_rankings['SystemName'] == system]
        if len(system_rank) > 0:
            result[f'{system.lower()}_rank'] = system_rank.iloc[0]['OrdinalRank']
        else:
            result[f'{system.lower()}_rank'] = 0
    
    return result

def calculate_team_season_stats(regular_season, detailed_regular_season, season, team_id):
    """
    Calculate comprehensive statistics for a team in a specific season.
    """
    # Basic stats from compact results
    team_games = regular_season[(regular_season['Season'] == season) & 
                              ((regular_season['WTeamID'] == team_id) | 
                               (regular_season['LTeamID'] == team_id))]
    
    if len(team_games) == 0:
        return None
    
    # Calculate win-loss record
    wins = team_games[team_games['WTeamID'] == team_id]
    losses = team_games[team_games['LTeamID'] == team_id]
    
    num_games = len(team_games)
    num_wins = len(wins)
    win_pct = num_wins / num_games if num_games > 0 else 0
    
    # Calculate scoring statistics
    points_scored = wins['WScore'].sum() + losses['LScore'].sum()
    points_allowed = wins['LScore'].sum() + losses['WScore'].sum()
    
    avg_points_scored = points_scored / num_games if num_games > 0 else 0
    avg_points_allowed = points_allowed / num_games if num_games > 0 else 0
    scoring_margin = avg_points_scored - avg_points_allowed
    
    # Get detailed stats if available
    detailed_stats = {}
    if detailed_regular_season is not None:
        detailed_team_games = detailed_regular_season[
            (detailed_regular_season['Season'] == season) & 
            ((detailed_regular_season['WTeamID'] == team_id) | 
             (detailed_regular_season['LTeamID'] == team_id))
        ]
        
        if len(detailed_team_games) > 0:
            # Wins detailed stats
            detailed_wins = detailed_team_games[detailed_team_games['WTeamID'] == team_id]
            detailed_losses = detailed_team_games[detailed_team_games['LTeamID'] == team_id]
            
            # Offensive stats
            fg_made = detailed_wins['WFGM'].sum() + detailed_losses['LFGM'].sum()
            fg_attempted = detailed_wins['WFGA'].sum() + detailed_losses['LFGA'].sum()
            fg_pct = fg_made / fg_attempted if fg_attempted > 0 else 0
            
            fg3_made = detailed_wins['WFGM3'].sum() + detailed_losses['LFGM3'].sum()
            fg3_attempted = detailed_wins['WFGA3'].sum() + detailed_losses['LFGA3'].sum()
            fg3_pct = fg3_made / fg3_attempted if fg3_attempted > 0 else 0
            
            ft_made = detailed_wins['WFTM'].sum() + detailed_losses['LFTM'].sum()
            ft_attempted = detailed_wins['WFTA'].sum() + detailed_losses['LFTA'].sum()
            ft_pct = ft_made / ft_attempted if ft_attempted > 0 else 0
            
            # Rebounding
            off_rebounds = detailed_wins['WOR'].sum() + detailed_losses['LOR'].sum()
            def_rebounds = detailed_wins['WDR'].sum() + detailed_losses['LDR'].sum()
            total_rebounds = off_rebounds + def_rebounds
            
            # Other stats
            assists = detailed_wins['WAst'].sum() + detailed_losses['LAst'].sum()
            turnovers = detailed_wins['WTO'].sum() + detailed_losses['LTO'].sum()
            steals = detailed_wins['WStl'].sum() + detailed_losses['LStl'].sum()
            blocks = detailed_wins['WBlk'].sum() + detailed_losses['LBlk'].sum()
            
            # Opponent stats
            opp_fg_made = detailed_wins['LFGM'].sum() + detailed_losses['WFGM'].sum()
            opp_fg_attempted = detailed_wins['LFGA'].sum() + detailed_losses['WFGA'].sum()
            opp_fg_pct = opp_fg_made / opp_fg_attempted if opp_fg_attempted > 0 else 0
            
            opp_fg3_made = detailed_wins['LFGM3'].sum() + detailed_losses['WFGM3'].sum()
            opp_fg3_attempted = detailed_wins['LFGA3'].sum() + detailed_losses['WFGA3'].sum()
            opp_fg3_pct = opp_fg3_made / opp_fg3_attempted if opp_fg3_attempted > 0 else 0
            
            # Calculate per-game averages
            detailed_stats = {
                'fg_pct': fg_pct,
                'fg3_pct': fg3_pct,
                'ft_pct': ft_pct,
                'avg_off_rebounds': off_rebounds / num_games,
                'avg_def_rebounds': def_rebounds / num_games,
                'avg_total_rebounds': total_rebounds / num_games,
                'avg_assists': assists / num_games,
                'avg_turnovers': turnovers / num_games,
                'avg_steals': steals / num_games,
                'avg_blocks': blocks / num_games,
                'opp_fg_pct': opp_fg_pct,
                'opp_fg3_pct': opp_fg3_pct,
                'assist_to_turnover': assists / turnovers if turnovers > 0 else 0,
                'turnover_margin': (detailed_wins['LTO'].sum() + detailed_losses['WTO'].sum() - turnovers) / num_games
            }
            
            # Calculate advanced metrics
            possessions = (fg_attempted - off_rebounds + turnovers - 0.4 * ft_attempted) / num_games
            opp_possessions = (opp_fg_attempted - (detailed_wins['LOR'].sum() + detailed_losses['WOR'].sum()) + 
                              (detailed_wins['LTO'].sum() + detailed_losses['WTO'].sum()) - 
                              0.4 * (detailed_wins['LFTA'].sum() + detailed_losses['WFTA'].sum())) / num_games
            
            avg_possessions = (possessions + opp_possessions) / 2
            
            offensive_efficiency = 100 * (points_scored / (avg_possessions * num_games))
            defensive_efficiency = 100 * (points_allowed / (avg_possessions * num_games))
            
            detailed_stats.update({
                'offensive_efficiency': offensive_efficiency,
                'defensive_efficiency': defensive_efficiency,
                'efficiency_margin': offensive_efficiency - defensive_efficiency
            })
    
    # Combine all stats
    base_stats = {
        'team_id': team_id,
        'season': season,
        'num_games': num_games,
        'num_wins': num_wins,
        'win_pct': win_pct,
        'avg_points_scored': avg_points_scored,
        'avg_points_allowed': avg_points_allowed,
        'scoring_margin': scoring_margin
    }
    
    return {**base_stats, **detailed_stats}

def create_matchup_features(team_a_stats, team_a_ranking, team_a_seed,
                          team_b_stats, team_b_ranking, team_b_seed):
    """
    Create features for a matchup between two teams.
    """
    if team_a_stats is None or team_b_stats is None:
        return None
    
    # Basic differentials
    features = {
        'seed_diff': team_a_seed - team_b_seed,
        'win_pct_diff': team_a_stats['win_pct'] - team_b_stats['win_pct'],
        'scoring_margin_diff': team_a_stats['scoring_margin'] - team_b_stats['scoring_margin'],
        'avg_points_scored_diff': team_a_stats['avg_points_scored'] - team_b_stats['avg_points_scored'],
        'avg_points_allowed_diff': team_a_stats['avg_points_allowed'] - team_b_stats['avg_points_allowed'],
        'avg_rank_diff': team_a_ranking['avg_rank'] - team_b_ranking['avg_rank']
    }
    
    # Add detailed stats differentials if available
    for stat in ['fg_pct', 'fg3_pct', 'ft_pct', 'avg_off_rebounds', 'avg_def_rebounds',
                'avg_total_rebounds', 'avg_assists', 'avg_turnovers', 'avg_steals',
                'avg_blocks', 'opp_fg_pct', 'opp_fg3_pct', 'assist_to_turnover',
                'turnover_margin', 'offensive_efficiency', 'defensive_efficiency',
                'efficiency_margin']:
        if stat in team_a_stats and stat in team_b_stats:
            features[f'{stat}_diff'] = team_a_stats[stat] - team_b_stats[stat]
    
    # Add specific ranking differentials if available
    for system in ['pom', 'sag', 'mor', 'dok', 'rpi']:
        if f'{system}_rank' in team_a_ranking and f'{system}_rank' in team_b_ranking:
            features[f'{system}_rank_diff'] = team_a_ranking[f'{system}_rank'] - team_b_ranking[f'{system}_rank']
    
    # Calculate interaction features
    features['seed_win_pct_interaction'] = features['seed_diff'] * features['win_pct_diff']
    features['seed_margin_interaction'] = features['seed_diff'] * features['scoring_margin_diff']
    
    # Calculate seed-based features
    # Higher seeds (lower numbers) historically perform better
    features['higher_seed'] = 1 if team_a_seed < team_b_seed else 0
    features['seed_diff_squared'] = features['seed_diff'] ** 2
    
    # Calculate upset potential features
    if 'efficiency_margin' in team_a_stats and 'efficiency_margin' in team_b_stats:
        features['upset_potential'] = abs(features['seed_diff']) * (1 - abs(features['efficiency_margin_diff']))
    
    return features

def prepare_tournament_data(data, start_season=2010, end_season=2024):
    """
    Prepare data for tournament matchups from start_season to end_season.
    """
    tourney_results = data['tourney_results']
    tourney_seeds = data['tourney_seeds']
    rankings = data['rankings']
    regular_season = data['regular_season']
    detailed_regular_season = data['detailed_regular_season']
    
    # Filter for the relevant seasons
    tourney_games = tourney_results[(tourney_results['Season'] >= start_season) & 
                                    (tourney_results['Season'] <= end_season)]
    
    # Prepare features for each tournament game
    features_list = []
    for _, game in tourney_games.iterrows():
        season = game['Season']
        team_a_id = game['WTeamID']
        team_b_id = game['LTeamID']
        
        # Get team statistics
        team_a_stats = calculate_team_season_stats(regular_season, detailed_regular_season, season, team_a_id)
        team_b_stats = calculate_team_season_stats(regular_season, detailed_regular_season, season, team_b_id)
        
        # Get team rankings
        team_a_ranking = get_team_rankings(rankings, season, team_a_id)
        team_b_ranking = get_team_rankings(rankings, season, team_b_id)
        
        # Get team seeds
        team_a_seed = get_team_seed(tourney_seeds, season, team_a_id)
        team_b_seed = get_team_seed(tourney_seeds, season, team_b_id)
        
        # Create matchup features
        matchup_features = create_matchup_features(
            team_a_stats, team_a_ranking, team_a_seed,
            team_b_stats, team_b_ranking, team_b_seed
        )
        
        if matchup_features is not None:
            # Add game identifier and result
            matchup_features['season'] = season
            matchup_features['team_a_id'] = team_a_id
            matchup_features['team_b_id'] = team_b_id
            matchup_features['result'] = 1  # team_a won
            
            features_list.append(matchup_features)
            
            # Create reversed matchup features (symmetry for training)
            reversed_matchup_features = create_matchup_features(
                team_b_stats, team_b_ranking, team_b_seed,
                team_a_stats, team_a_ranking, team_a_seed
            )
            
            if reversed_matchup_features is not None:
                reversed_matchup_features['season'] = season
                reversed_matchup_features['team_a_id'] = team_b_id
                reversed_matchup_features['team_b_id'] = team_a_id
                reversed_matchup_features['result'] = 0  # team_a lost
                
                features_list.append(reversed_matchup_features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    
    return features_df
```

# 3. Model Training and Evaluation

Now let's define functions to train and evaluate our prediction model.

```python
def train_model(features_df, test_size=0.2, random_state=42):
    """
    Train a gradient boosting model on the provided features.
    """
    # Separate features and target
    X = features_df.drop(['season', 'team_a_id', 'team_b_id', 'result'], axis=1)
    y = features_df['result']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train gradient boosting model
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=random_state
    )
    
    gb_model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    train_probs = gb_model.predict_proba(X_train_scaled)[:, 1]
    test_probs = gb_model.predict_proba(X_test_scaled)[:, 1]
    
    train_loss = log_loss(y_train, train_probs)
    test_loss = log_loss(y_test, test_probs)
    
    print(f"Training Log Loss: {train_loss:.4f}")
    print(f"Testing Log Loss: {test_loss:.4f}")
    
    # Calculate ROC AUC
    from sklearn.metrics import roc_auc_score
    train_auc = roc_auc_score(y_train, train_probs)
    test_auc = roc_auc_score(y_test, test_probs)
    
    print(f"Training ROC AUC: {train_auc:.4f}")
    print(f"Testing ROC AUC: {test_auc:.4f}")
    
    # Identify most important features
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': gb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
    
    # Return model and associated objects
    return {
        'model': gb_model,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'feature_importance': feature_importance,
        'train_metrics': {'log_loss': train_loss, 'auc': train_auc},
        'test_metrics': {'log_loss': test_loss, 'auc': test_auc}
    }

def tune_model(features_df, param_grid=None, cv=5, random_state=42):
    """
    Perform hyperparameter tuning for the gradient boosting model.
    """
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.8, 1.0]
        }
    
    # Separate features and target
    X = features_df.drop(['season', 'team_a_id', 'team_b_id', 'result'], axis=1)
    y = features_df['result']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create base model
    gb_model = GradientBoostingClassifier(random_state=random_state)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=gb_model,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_log_loss',
        n_jobs=-1,
        verbose=1
    )
    
    print("Starting hyperparameter tuning...")
    grid_search.fit(X_scaled, y)
    
    # Print best parameters and score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best log loss: {-grid_search.best_score_:.4f}")
    
    # Train final model with best parameters
    best_model = GradientBoostingClassifier(
        **grid_search.best_params_,
        random_state=random_state
    )
    
    best_model.fit(X_scaled, y)
    
    # Return best model and scaler
    return {
        'model': best_model,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'best_params': grid_search.best_params_
    }

def save_model(model_dict, file_path):
    """
    Save the trained model and associated objects to a file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(model_dict, f)
    
    print(f"Model saved to {file_path}")

def load_model(file_path):
    """
    Load a trained model and associated objects from a file.
    """
    with open(file_path, 'rb') as f:
        model_dict = pickle.load(f)
    
    return model_dict

def evaluate_model_performance(model_dict, features_df):
    """
    Evaluate model performance on historical tournament data.
    """
    # Extract model components
    model = model_dict['model']
    scaler = model_dict['scaler']
    feature_names = model_dict['feature_names']
    
    # Split data into features and target
    X = features_df[feature_names]
    y = features_df['result']
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate log loss
    loss = log_loss(y, y_pred_proba)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Generate confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Generate classification report
    cr = classification_report(y, y_pred)
    
    # Print results
    print(f"Log Loss: {loss:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    
    # Create ROC curve plot
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    # Create confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    return {
        'log_loss': loss,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'classification_report': cr,
        'y_true': y,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
```

# 4. Tournament Prediction

Now let's define functions to generate predictions for the NCAA tournament.

```python
def prepare_tournament_matchups(data, season=2025):
    """
    Prepare potential matchups for the current tournament season.
    """
    # We'd need the teams and their seeds for the current tournament
    # This would typically be known right before the tournament starts
    
    # For now, we'll generate all possible matchups
    tourney_seeds = data['tourney_seeds']
    rankings = data['rankings']
    regular_season = data['regular_season']
    detailed_regular_season = data['detailed_regular_season']
    
    # Get teams in the tournament for the given season
    # In a real scenario, we'd have the actual 2025 seeds
    # For demonstration, we'll use the latest available season's seeds
    latest_season = tourney_seeds['Season'].max()
    prediction_season = latest_season
    print(f"Using seeds from {prediction_season} season for demonstration purposes")
    
    tourney_teams = tourney_seeds[tourney_seeds['Season'] == prediction_season]['TeamID'].unique()
    
    # Generate all possible matchups
    matchups = []
    for i, team_a_id in enumerate(tqdm(tourney_teams, desc="Generating matchups")):
        for team_b_id in tourney_teams[i+1:]:
            # Get team statistics
            team_a_stats = calculate_team_season_stats(regular_season, detailed_regular_season, season, team_a_id)
            team_b_stats = calculate_team_season_stats(regular_season, detailed_regular_season, season, team_b_id)
            
            # Get team rankings
            team_a_ranking = get_team_rankings(rankings, season, team_a_id)
            team_b_ranking = get_team_rankings(rankings, season, team_b_id)
            
            # Get team seeds
            team_a_seed = get_team_seed(tourney_seeds, prediction_season, team_a_id)
            team_b_seed = get_team_seed(tourney_seeds, prediction_season, team_b_id)
            
            # Create matchup features
            matchup_features = create_matchup_features(
                team_a_stats, team_a_ranking, team_a_seed,
                team_b_stats, team_b_ranking, team_b_seed
            )
            
            if matchup_features is not None:
                matchup_features['team_a_id'] = team_a_id
                matchup_features['team_b_id'] = team_b_id
                matchups.append(matchup_features)
    
    return pd.DataFrame(matchups)

def predict_matchup(model_dict, team_a_id, team_b_id, features_df):
    """
    Predict the outcome of a specific matchup.
    """
    # Extract the model and scaler
    model = model_dict['model']
    scaler = model_dict['scaler']
    feature_names = model_dict['feature_names']
    
    # Find the matchup in the features DataFrame
    matchup = features_df[(features_df['team_a_id'] == team_a_id) & 
                        (features_df['team_b_id'] == team_b_id)]
    
    if len(matchup) == 0:
        # Try the reverse matchup
        matchup = features_df[(features_df['team_a_id'] == team_b_id) & 
                            (features_df['team_b_id'] == team_a_id)]
        
        if len(matchup) == 0:
            print(f"Matchup not found: {team_a_id} vs {team_b_id}")
            return 0.5
        
        # If we found the reverse matchup, we need to invert the prediction
        matchup_features = matchup[feature_names].values
        matchup_features_scaled = scaler.transform(matchup_features)
        prob = 1 - model.predict_proba(matchup_features_scaled)[0, 1]
        
        return prob
    
    # Get the features for the matchup
    matchup_features = matchup[feature_names].values
    matchup_features_scaled = scaler.transform(matchup_features)
    
    # Predict the probability of team_a winning
    prob = model.predict_proba(matchup_features_scaled)[0, 1]
    
    return prob

def simulate_tournament(model_dict, tourney_structure, features_df, teams_df, num_simulations=1000):
    """
    Simulate the tournament multiple times to get expected outcomes.
    
    tourney_structure should be a dictionary mapping regions to lists of seeds.
    """
    # Create team name mapping
    team_id_to_name = dict(zip(teams_df['TeamID'], teams_df['TeamName']))
    
    # Initialize results
    team_win_counts = {}
    for team_id in features_df['team_a_id'].unique():
        team_win_counts[team_id] = {
            'round1': 0,
            'round2': 0,
            'sweet16': 0,
            'elite8': 0,
            'final4': 0,
            'championship': 0,
            'winner': 0
        }
    
    # Run simulations
    for sim in tqdm(range(num_simulations), desc="Simulating tournament"):
        # Initialize bracket with first round matchups
        bracket = {}
        for region, seeds in tourney_structure.items():
            bracket[region] = []
            for i in range(8):
                team1_seed = seeds[i]
                team2_seed = seeds[15-i]
                team1_rows = teams_df[(teams_df['Seed'] == f"{region}{team1_seed:02d}")]
                team2_rows = teams_df[(teams_df['Seed'] == f"{region}{team2_seed:02d}")]
                
                if len(team1_rows) == 0 or len(team2_rows) == 0:
                    print(f"Team not found for region {region}, seeds {team1_seed} and {team2_seed}")
                    continue
                    
                team1_id = team1_rows.iloc[0]['TeamID']
                team2_id = team2_rows.iloc[0]['TeamID']
                bracket[region].append((team1_id, team2_id))
        
        # Simulate each round
        for round_num in range(6):
            next_bracket = {}
            
            # First 4 rounds are within regions
            if round_num < 4:
                for region, matchups in bracket.items():
                    next_bracket[region] = []
                    for i in range(0, len(matchups), 2):
                        if i+1 < len(matchups):
                            team1_id, team2_id = matchups[i]
                            team3_id, team4_id = matchups[i+1]
                            
                            # Simulate first matchup
                            prob1 = predict_matchup(model_dict, team1_id, team2_id, features_df)
                            winner1 = team1_id if np.random.random() < prob1 else team2_id
                            
                            # Simulate second matchup
                            prob2 = predict_matchup(model_dict, team3_id, team4_id, features_df)
                            winner2 = team3_id if np.random.random() < prob2 else team4_id
                            
                            # Add winners to next round
                            next_bracket[region].append((winner1, winner2))
                            
                            # Update win counts
                            round_name = ['round1', 'round2', 'sweet16', 'elite8'][round_num]
                            team_win_counts[winner1][round_name] += 1
                            team_win_counts[winner2][round_name] += 1
            
            # Final Four
            elif round_num == 4:
                next_bracket['FinalFour'] = []
                regions = list(bracket.keys())
                
                # Matchup 1: regions[0] vs regions[1]
                team1_id, team2_id = bracket[regions[0]][0]
                prob1 = predict_matchup(model_dict, team1_id, team2_id, features_df)
                winner1 = team1_id if np.random.random() < prob1 else team2_id
                
                team3_id, team4_id = bracket[regions[1]][0]
                prob2 = predict_matchup(model_dict, team3_id, team4_id, features_df)
                winner2 = team3_id if np.random.random() < prob2 else team4_id
                
                # Matchup 2: regions[2] vs regions[3]
                team5_id, team6_id = bracket[regions[2]][0]
                prob3 = predict_matchup(model_dict, team5_id, team6_id, features_df)
                winner3 = team5_id if np.random.random() < prob3 else team6_id
                
                team7_id, team8_id = bracket[regions[3]][0]
                prob4 = predict_matchup(model_dict, team7_id, team8_id, features_df)
                winner4 = team7_id if np.random.random() < prob4 else team8_id
                
                # Add winners to championship
                next_bracket['FinalFour'].append((winner1, winner3))
                next_bracket['FinalFour'].append((winner2, winner4))
                
                # Update win counts
                team_win_counts[winner1]['final4'] += 1
                team_win_counts[winner2]['final4'] += 1
                team_win_counts[winner3]['final4'] += 1
                team_win_counts[winner4]['final4'] += 1
            
            # Championship
            else:
                team1_id, team2_id = next_bracket['FinalFour'][0]
                team3_id, team4_id = next_bracket['FinalFour'][1]
                
                # Simulate championship semifinal 1
                prob1 = predict_matchup(model_dict, team1_id, team2_id, features_df)
                finalist1 = team1_id if np.random.random() < prob1 else team2_id
                
                # Simulate championship semifinal 2
                prob2 = predict_matchup(model_dict, team3_id, team4_id, features_df)
                finalist2 = team3_id if np.random.random() < prob2 else team4_id
                
                # Simulate championship game
                prob_final = predict_matchup(model_dict, finalist1, finalist2, features_df)
                champion = finalist1 if np.random.random() < prob_final else finalist2
                
                # Update win counts
                team_win_counts[finalist1]['championship'] += 1
                team_win_counts[finalist2]['championship'] += 1
                team_win_counts[champion]['winner'] += 1
            
            # Update bracket for next round
            bracket = next_bracket
    
    # Convert win counts to probabilities
    team_win_probs = {}
    for team_id, rounds in team_win_counts.items():
        team_name = team_id_to_name.get(team_id, f"Team {team_id}")
        team_win_probs[team_name] = {
            round_name: count / num_simulations
            for round_name, count in rounds.items()
        }
        team_win_probs[team_name]['team_id'] = team_id
    
    return team_win_probs
```

# 5. Training the Model

Now let's prepare our training data and train the prediction model.

```python
# Prepare tournament data for training
print("Preparing tournament data for training...")
tournament_features = prepare_tournament_data(data, start_season=2010, end_season=2024)

print(f"\nDataset shape: {tournament_features.shape}")
print(f"Number of features: {tournament_features.shape[1] - 4}")
print(f"Number of games: {tournament_features.shape[0] // 2}")

# Display sample of the features
display(tournament_features.head())
```

```python
# Train the model
print("Training the prediction model...")
model_dict = train_model(tournament_features)

# Save the trained model
model_path = os.path.join(MODEL_DIR, "march_madness_model.pkl")
save_model(model_dict, model_path)

# Additional evaluation
print("\nEvaluating model performance on all data...")
eval_results = evaluate_model_performance(model_dict, tournament_features)
```

# 6. Hyperparameter Tuning (Optional)


```python
# Comment out this cell if you want to skip tuning
'''
# Define parameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5],
    'subsample': [0.8, 0.9, 1.0]
}

# Tune model
tuned_model_dict = tune_model(tournament_features, param_grid=param_grid, cv=5)

# Save tuned model
tuned_model_path = os.path.join(MODEL_DIR, "tuned_march_madness_model.pkl")
save_model(tuned_model_dict, tuned_model_path)

# Use tuned model for predictions
model_dict = tuned_model_dict
'''
```

# 7. Tournament Prediction

Now let's generate predictions for the current NCAA tournament season.

```python
# Prepare tournament matchups for prediction
print("Preparing tournament matchups for prediction...")
prediction_season = 2025  # Current season
matchups = prepare_tournament_matchups(data, season=prediction_season)
```

```python
# Define tournament structure
# UPDATE THIS WITH ACTUAL 2025 BRACKETS WHEN AVAILABLE

# Get the latest tournament seeds for demonstration
latest_season = data['tourney_seeds']['Season'].max()
latest_seeds = data['tourney_seeds'][data['tourney_seeds']['Season'] == latest_season]

# Extract region information
regions = set([seed[0] for seed in latest_seeds['Seed']])
print(f"Tournament regions: {regions}")

# Create tournament structure
tourney_structure = {}
for region in regions:
    tourney_structure[region] = list(range(1, 17))  # Seeds 1-16

print("Tournament structure defined for simulation")
```

```python
# Simulate tournament
num_simulations = 1000  # Increase for more stable results
print(f"Simulating tournament {num_simulations} times...")

# Use latest season's seeds for demonstration
teams_df = latest_seeds

# Run simulation
results = simulate_tournament(model_dict, tourney_structure, matchups, teams_df, num_simulations)
```

```python
# Display tournament prediction results
print("\nTOURNAMENT PREDICTION RESULTS")
print("=============================")

# Championship probabilities
championship_probs = pd.DataFrame([
    {'Team': team, 'Probability': probs['winner']} 
    for team, probs in results.items()
]).sort_values('Probability', ascending=False).reset_index(drop=True)

print("\nChampionship Probabilities:")
display(championship_probs.head(10))

# Plot championship probabilities
plt.figure(figsize=(12, 8))
sns.barplot(x='Probability', y='Team', data=championship_probs.head(20))
plt.title('Championship Probabilities (Top 20 Teams)')
plt.xlabel('Probability')
plt.ylabel('Team')
plt.tight_layout()
plt.show()

# Final Four probabilities
final_four_probs = pd.DataFrame([
    {'Team': team, 'Probability': probs['final4']} 
    for team, probs in results.items()
]).sort_values('Probability', ascending=False).reset_index(drop=True)

print("\nFinal Four Probabilities:")
display(final_four_probs.head(10))

# Combine probabilities for different rounds
all_rounds_probs = pd.DataFrame()
all_rounds_probs['Team'] = [team for team in championship_probs['Team']]
all_rounds_probs['Champion'] = [results[team]['winner'] for team in all_rounds_probs['Team']]
all_rounds_probs['Final Four'] = [results[team]['final4'] for team in all_rounds_probs['Team']]
all_rounds_probs['Elite Eight'] = [results[team]['elite8'] for team in all_rounds_probs['Team']]
all_rounds_probs['Sweet 16'] = [results[team]['sweet16'] for team in all_rounds_probs['Team']]
all_rounds_probs['Round of 32'] = [results[team]['round2'] for team in all_rounds_probs['Team']]
all_rounds_probs['Round of 64'] = [results[team]['round1'] for team in all_rounds_probs['Team']]

# Display comprehensive results
print("\nComprehensive Tournament Probabilities (Top 20 Teams):")
display(all_rounds_probs.head(20))

# Save results to CSV
results_file = os.path.join(RESULTS_DIR, f"mm2025_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
all_rounds_probs.to_csv(results_file, index=False)
print(f"\nDetailed results saved to: {results_file}")
```

# 8. Result Analysis and Visualization

Let's create visualizations to better understand the predictions.

```python
# Create a heatmap of round probabilities for top teams
top_teams = championship_probs.head(15)['Team'].tolist()
round_columns = ['Round of 64', 'Round of 32', 'Sweet 16', 'Elite Eight', 'Final Four', 'Champion']

# Filter for top teams
heatmap_data = all_rounds_probs[all_rounds_probs['Team'].isin(top_teams)].copy()
heatmap_data = heatmap_data.set_index('Team')[round_columns].sort_values('Champion', ascending=False)

# Create heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_data, annot=True, fmt='.1%', cmap='YlGnBu', linewidths=0.5)
plt.title('Tournament Advancement Probabilities (Top 15 Teams)')
plt.tight_layout()
plt.show()

# Create bracket visualization (simplified)
top_teams_by_region = {}
for region in regions:
    # Find teams in this region
    region_seeds = latest_seeds[latest_seeds['Seed'].str.startswith(region)]
    region_teams = [team_id_to_name.get(team_id, f"Team {team_id}") for team_id in region_seeds['TeamID']]
    # Keep only teams that are in our results
    region_teams = [team for team in region_teams if team in results]
    # Sort by championship probability
    region_teams = sorted(region_teams, key=lambda t: results[t]['winner'], reverse=True)
    # Keep top 4
    top_teams_by_region[region] = region_teams[:4]

# Plot region probabilities
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, (region, teams) in enumerate(top_teams_by_region.items()):
    data = []
    for team in teams:
        data.append({
            'Team': team,
            'Sweet 16': results[team]['sweet16'],
            'Elite Eight': results[team]['elite8'],
            'Final Four': results[team]['final4'],
            'Champion': results[team]['winner']
        })
    
    df = pd.DataFrame(data)
    df = df.set_index('Team').sort_values('Champion', ascending=False)
    df = df.T  # Transpose for better visualization
    
    ax = axes[i]
    sns.heatmap(df, annot=True, fmt='.1%', cmap='YlGnBu', linewidths=0.5, ax=ax)
    ax.set_title(f"{region} Region - Top Teams")
    ax.set_ylabel('Tournament Round')
    
plt.tight_layout()
plt.show()
```

```python
# Analyze upset potential
# For each team, compute an upset potential score
upset_potential = []

for team, probs in results.items():
    # Find the team's seed
    team_id = probs['team_id']
    team_seed_rows = latest_seeds[latest_seeds['TeamID'] == team_id]
    
    if len(team_seed_rows) > 0:
        seed_string = team_seed_rows.iloc[0]['Seed']
        seed_number = extract_seed_number(seed_string)
        
        # Calculate upset potential score
        # Higher seeds (8-16) with high Elite Eight or better probability
        if seed_number >= 8:
            elite_eight_plus = probs['elite8'] + probs['final4'] + probs['championship'] + probs['winner']
            upset_score = elite_eight_plus * seed_number / 8
            
            upset_potential.append({
                'Team': team,
                'Seed': seed_number,
                'Elite Eight+': elite_eight_plus,
                'Upset Score': upset_score
            })

# Sort by upset potential
upset_df = pd.DataFrame(upset_potential).sort_values('Upset Score', ascending=False).reset_index(drop=True)

# Display potential Cinderella teams
print("Potential Cinderella Teams (Upset Potential):")
display(upset_df.head(10))

# Plot upset potential
plt.figure(figsize=(12, 8))
sns.barplot(x='Upset Score', y='Team', hue='Seed', palette='YlOrRd', data=upset_df.head(10))
plt.title('Cinderella Potential (Top 10 Teams)')
plt.xlabel('Upset Potential Score')
plt.legend(title='Seed')
plt.tight_layout()
plt.show()
```

# 9. Final Bracket Prediction

Let's create a final bracket prediction based on the most likely outcome at each stage.

```python
def create_most_likely_bracket(results, tourney_structure, teams_df):
    """
    Create the most likely bracket based on head-to-head win probabilities.
    """
    team_id_to_name = dict(zip(teams_df['TeamID'], teams_df['TeamName']))
    seed_to_team_id = dict(zip(teams_df['Seed'], teams_df['TeamID']))
    
    # Initialize bracket with first round matchups
    bracket = {}
    for region, seeds in tourney_structure.items():
        bracket[region] = []
        for i in range(8):
            seed1 = f"{region}{seeds[i]:02d}"
            seed2 = f"{region}{seeds[15-i]:02d}"
            team1_id = seed_to_team_id.get(seed1)
            team2_id = seed_to_team_id.get(seed2)
            
            if team1_id is not None and team2_id is not None:
                team1_name = team_id_to_name.get(team1_id, f"Team {team1_id}")
                team2_name = team_id_to_name.get(team2_id, f"Team {team2_id}")
                bracket[region].append((team1_id, team1_name, team2_id, team2_name))
    
    # Build most likely bracket
    most_likely_bracket = {}
    for r in range(6):  # 6 rounds from first round to champion
        most_likely_bracket[f"Round {r+1}"] = []
    
    # Process first round
    most_likely_bracket["Round 1"] = bracket
    
    # Process subsequent rounds
    for round_num in range(1, 6):  # Rounds 2-6
        prev_round = most_likely_bracket[f"Round {round_num}"]
        next_round = {}
        
        # First 4 rounds are within regions
        if round_num < 4:
            for region, matchups in prev_round.items():
                next_round[region] = []
                for i in range(0, len(matchups), 2):
                    if i+1 < len(matchups):
                        team1_id, team1_name, team2_id, team2_name = matchups[i]
                        team3_id, team3_name, team4_id, team4_name = matchups[i+1]
                        
                        # First matchup most likely winner
                        team1_prob = results[team1_name]['winner'] if team1_name in results else 0
                        team2_prob = results[team2_name]['winner'] if team2_name in results else 0
                        winner1_id, winner1_name = (team1_id, team1_name) if team1_prob > team2_prob else (team2_id, team2_name)
                        
                        # Second matchup most likely winner
                        team3_prob = results[team3_name]['winner'] if team3_name in results else 0
                        team4_prob = results[team4_name]['winner'] if team4_name in results else 0
                        winner2_id, winner2_name = (team3_id, team3_name) if team3_prob > team4_prob else (team4_id, team4_name)
                        
                        next_round[region].append((winner1_id, winner1_name, winner2_id, winner2_name))
        
        # Final Four (round 4)
        elif round_num == 4:
            next_round["Final Four"] = []
            regions = list(prev_round.keys())
            
            # Get regional champions
            regional_champs = []
            for region in regions:
                if len(prev_round[region]) > 0:
                    team1_id, team1_name, team2_id, team2_name = prev_round[region][0]
                    team1_prob = results[team1_name]['winner'] if team1_name in results else 0
                    team2_prob = results[team2_name]['winner'] if team2_name in results else 0
                    winner_id, winner_name = (team1_id, team1_name) if team1_prob > team2_prob else (team2_id, team2_name)
                    regional_champs.append((winner_id, winner_name))
            
            # Create Final Four matchups
            if len(regional_champs) >= 4:
                next_round["Final Four"].append((regional_champs[0][0], regional_champs[0][1], 
                                            regional_champs[1][0], regional_champs[1][1]))
                next_round["Final Four"].append((regional_champs[2][0], regional_champs[2][1], 
                                            regional_champs[3][0], regional_champs[3][1]))
        
        # Championship (round 5)
        else:
            next_round["Championship"] = []
            if "Final Four" in prev_round and len(prev_round["Final Four"]) >= 2:
                team1_id, team1_name, team2_id, team2_name = prev_round["Final Four"][0]
                team3_id, team3_name, team4_id, team4_name = prev_round["Final Four"][1]
                
                # First semifinal most likely winner
                team1_prob = results[team1_name]['winner'] if team1_name in results else 0
                team2_prob = results[team2_name]['winner'] if team2_name in results else 0
                finalist1_id, finalist1_name = (team1_id, team1_name) if team1_prob > team2_prob else (team2_id, team2_name)
                
                # Second semifinal most likely winner
                team3_prob = results[team3_name]['winner'] if team3_name in results else 0
                team4_prob = results[team4_name]['winner'] if team4_name in results else 0
                finalist2_id, finalist2_name = (team3_id, team3_name) if team3_prob > team4_prob else (team4_id, team4_name)
                
                next_round["Championship"].append((finalist1_id, finalist1_name, finalist2_id, finalist2_name))
        
        most_likely_bracket[f"Round {round_num+1}"] = next_round
    
    # Determine champion
    champion = None
    if "Championship" in most_likely_bracket["Round 6"] and len(most_likely_bracket["Round 6"]["Championship"]) > 0:
        team1_id, team1_name, team2_id, team2_name = most_likely_bracket["Round 6"]["Championship"][0]
        team1_prob = results[team1_name]['winner'] if team1_name in results else 0
        team2_prob = results[team2_name]['winner'] if team2_name in results else 0
        champion = team1_name if team1_prob > team2_prob else team2_name
    
    return most_likely_bracket, champion

# Create most likely bracket
most_likely_bracket, champion = create_most_likely_bracket(results, tourney_structure, teams_df)

# Display final bracket prediction
print("\nMOST LIKELY BRACKET PREDICTION")
print("===============================")
print(f"Predicted Champion: {champion}")

# Display Final Four teams
print("\nFinal Four Teams:")
for matchup in most_likely_bracket["Round 5"]["Final Four"]:
    print(f"{matchup[1]} vs {matchup[3]}")

# Display Elite Eight teams by region
print("\nElite Eight Teams (by region):")
for region, matchups in most_likely_bracket["Round 4"].items():
    print(f"\n{region} Region:")
    for matchup in matchups:
        print(f"{matchup[1]} vs {matchup[3]}")
```

# 10. Performance Analysis of Previous Tournaments

Let's analyze how our model would have performed on previous tournaments.

```python
def evaluate_previous_tournament(data, model_dict, test_season=2024):
    """
    Evaluate model performance on a previous tournament.
    """
    # Get actual tournament results
    tourney_results = data['tourney_results']
    test_tourney = tourney_results[tourney_results['Season'] == test_season]
    
    # Create matchup features using data before the tournament
    matchups = prepare_tournament_matchups(data, season=test_season)
    
    # Extract model components
    model = model_dict['model']
    scaler = model_dict['scaler']
    feature_names = model_dict['feature_names']
    
    # Initialize counters
    correct_predictions = 0
    total_games = 0
    game_predictions = []
    
    # Evaluate each actual tournament game
    for _, game in test_tourney.iterrows():
        team_a_id = game['WTeamID']
        team_b_id = game['LTeamID']
        
        # Predict matchup
        pred_prob = predict_matchup(model_dict, team_a_id, team_b_id, matchups)
        pred_winner = team_a_id if pred_prob > 0.5 else team_b_id
        actual_winner = team_a_id
        
        # Update counters
        correct = (pred_winner == actual_winner)
        correct_predictions += int(correct)
        total_games += 1
        
        # Store prediction details
        team_a_name = data['teams'][data['teams']['TeamID'] == team_a_id].iloc[0]['TeamName'] \
            if len(data['teams'][data['teams']['TeamID'] == team_a_id]) > 0 else f"Team {team_a_id}"
        team_b_name = data['teams'][data['teams']['TeamID'] == team_b_id].iloc[0]['TeamName'] \
            if len(data['teams'][data['teams']['TeamID'] == team_b_id]) > 0 else f"Team {team_b_id}"
        
        # Get seeds
        team_a_seed = data['tourney_seeds'][(data['tourney_seeds']['Season'] == test_season) & 
                                          (data['tourney_seeds']['TeamID'] == team_a_id)].iloc[0]['Seed'] \
            if len(data['tourney_seeds'][(data['tourney_seeds']['Season'] == test_season) & 
                                       (data['tourney_seeds']['TeamID'] == team_a_id)]) > 0 else "?"
        team_b_seed = data['tourney_seeds'][(data['tourney_seeds']['Season'] == test_season) & 
                                          (data['tourney_seeds']['TeamID'] == team_b_id)].iloc[0]['Seed'] \
            if len(data['tourney_seeds'][(data['tourney_seeds']['Season'] == test_season) & 
                                       (data['tourney_seeds']['TeamID'] == team_b_id)]) > 0 else "?"
        
        game_predictions.append({
            'Team A': team_a_name,
            'Team B': team_b_name,
            'Seed A': team_a_seed,
            'Seed B': team_b_seed,
            'Predicted Prob': pred_prob,
            'Predicted Winner': 'Team A' if pred_winner == team_a_id else 'Team B',
            'Actual Winner': 'Team A',
            'Correct': correct
        })
    
    # Calculate accuracy
    accuracy = correct_predictions / total_games if total_games > 0 else 0
    
    # Print results
    print(f"Model Performance on {test_season} Tournament:")
    print(f"Accuracy: {accuracy:.2%} ({correct_predictions}/{total_games} correct predictions)")
    
    # Create and return DataFrame of predictions
    predictions_df = pd.DataFrame(game_predictions)
    return predictions_df, accuracy

# Evaluate on previous tournament
test_season = 2024  # Most recent tournament
predictions_df, accuracy = evaluate_previous_tournament(data, model_dict, test_season)

# Display predictions
print("\nGame Predictions:")
display(predictions_df)

# Plot prediction accuracy by seed difference
predictions_df['Seed A Num'] = predictions_df['Seed A'].apply(lambda x: int(x[1:3]) if isinstance(x, str) and len(x) >= 3 else 0)
predictions_df['Seed B Num'] = predictions_df['Seed B'].apply(lambda x: int(x[1:3]) if isinstance(x, str) and len(x) >= 3 else 0)
predictions_df['Seed Diff'] = predictions_df['Seed B Num'] - predictions_df['Seed A Num']

# Group by seed difference
seed_diff_accuracy = predictions_df.groupby('Seed Diff').agg(
    Games=('Correct', 'count'),
    Correct=('Correct', 'sum'),
    Accuracy=('Correct', 'mean')
).reset_index()

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x='Seed Diff', y='Accuracy', data=seed_diff_accuracy)
plt.title(f'Prediction Accuracy by Seed Difference ({test_season} Tournament)')
plt.xlabel('Seed Difference (Higher - Lower)')
plt.ylabel('Accuracy')
plt.axhline(y=accuracy, color='r', linestyle='--', label=f'Overall Accuracy: {accuracy:.2%}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

# Conclusion

In this notebook, we've built a comprehensive prediction model for the NCAA March Madness tournament. The model:

1. Processes historical NCAA basketball data
2. Engineers features from team statistics, rankings, and tournament seedings
3. Trains a gradient boosting model to predict game outcomes
4. Simulates the tournament to generate team advancement probabilities
5. Evaluates performance on historical tournaments

Key findings and insights:
- Seed difference remains one of the strongest predictors of tournament outcomes
- Team efficiency metrics (offensive and defensive) provide substantial predictive power
- The model can identify potential "Cinderella" teams with upset potential
- Tournament simulations provide more nuanced insights than simple game-by-game predictions

For the 2025 tournament, we've generated bracket predictions and team advancement probabilities that can guide your bracket selections. Remember that March Madness is inherently unpredictable, and even the best models can't account for all the factors that make the tournament so exciting!

### Next Steps

To improve the model further, you could:
1. Incorporate player-level statistics when available
2. Add recency bias to favor more recent game results
3. Include coaching experience factors
4. Add travel distance/fatigue effects
5. Incorporate injury information when available

When the actual 2025 tournament brackets are announced, update the tournament structure with the correct teams and seeds to generate your final predictions.
        
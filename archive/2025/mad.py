import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import gc

# Configuration
YEAR = 2025
DATA_PATH = 'data/'
SUBMISSION_FILE = 'final_submission.csv'

def main():
    # Load and prepare data
    print("Loading and preprocessing data...")
    teams, seeds, stats = load_data()
    
    # Generate all possible matchups
    print("Generating matchups...")
    submission = create_submission_template(teams)
    
    # Create features
    print("Engineering features...")
    features = create_features(submission, seeds, stats)
    
    # Load/prepare model
    print("Preparing model...")
    model, scaler = train_model(features.sample(100000))  # Use subset for demo
    
    # Generate predictions
    print("Making predictions...")
    submission['Pred'] = predict_matchups(features, model, scaler)
    
    # Save results
    submission[['ID', 'Pred']].to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission saved to {SUBMISSION_FILE}")

def load_data():
    """Load and merge all required data with robust handling"""
    # Load team lists
    m_teams = pd.read_csv(f'{DATA_PATH}MTeams.csv', dtype={'TeamID': 'int16'})
    w_teams = pd.read_csv(f'{DATA_PATH}WTeams.csv', dtype={'TeamID': 'int16'})
    
    # Load and process seeds
    seeds = pd.concat([
        process_seeds(pd.read_csv(f'{DATA_PATH}MNCAATourneySeeds.csv'), 
        process_seeds(pd.read_csv(f'{DATA_PATH}WNCAATourneySeeds.csv'))
    ]).fillna(16)
    
    # Load regular season stats
    m_reg = pd.read_csv(f'{DATA_PATH}MRegularSeasonCompactResults.csv')
    w_reg = pd.read_csv(f'{DATA_PATH}WRegularSeasonCompactResults.csv')
    stats = pd.concat([m_reg, w_reg])
    
    # Calculate team metrics
    stats = stats.groupby(['Season', 'WTeamID']).agg(
        Offense=('WScore', 'mean'),
        Defense=('LScore', 'mean'),
        Games=('WScore', 'count')
    ).reset_index()
    
    # Add rolling averages
    stats[['Offense', 'Defense']] = stats.groupby('WTeamID')[['Offense', 'Defense']]\
                                      .transform(lambda x: x.rolling(3, 1).mean())
    
    return (pd.concat([m_teams[['TeamID']], w_teams[['TeamID']]), 
            seeds,
            stats.fillna(stats.mean()))

def process_seeds(seeds_df):
    """Extract numerical seed value with validation"""
    seeds_df['Seed'] = seeds_df['Seed'].str.extract('(\d+)').astype('int8')
    return seeds_df[['Season', 'TeamID', 'Seed']]

def create_submission_template(teams):
    """Generate all possible team pairs for 2025"""
    # Split men's and women's teams
    men = teams[teams['TeamID'] < 3000]['TeamID'].unique()
    women = teams[teams['TeamID'] >= 3000]['TeamID'].unique()
    
    # Create all combinations
    men_pairs = [f"{YEAR}_{a}_{b}" for a, b in combinations(sorted(men), 2)]
    women_pairs = [f"{YEAR}_{a}_{b}" for a, b in combinations(sorted(women), 2)]
    
    return pd.DataFrame({
        'ID': men_pairs + women_pairs,
        'Season': YEAR,
        'Team1': [x.split('_')[1] for x in men_pairs + women_pairs],
        'Team2': [x.split('_')[2] for x in men_pairs + women_pairs]
    }).astype({'Team1': 'int16', 'Team2': 'int16'})

def create_features(submission, seeds, stats):
    """Create matchup features with fallbacks"""
    # Merge seed data
    features = submission.merge(
        seeds[seeds['Season'] == YEAR],
        left_on='Team1',
        right_on='TeamID',
        suffixes=('', '_1')
    ).merge(
        seeds[seeds['Season'] == YEAR],
        left_on='Team2',
        right_on='TeamID',
        suffixes=('_1', '_2')
    )
    
    # Merge performance stats
    features = features.merge(
        stats[stats['Season'] == YEAR],
        left_on='Team1',
        right_on='WTeamID',
        suffixes=('', '_1')
    ).merge(
        stats[stats['Season'] == YEAR],
        left_on='Team2',
        right_on='WTeamID',
        suffixes=('_1', '_2')
    )
    
    # Create feature columns
    features['Seed_Diff'] = features['Seed_1'] - features['Seed_2']
    features['Offense_Diff'] = features['Offense_1'] - features['Offense_2']
    features['Defense_Diff'] = features['Defense_1'] - features['Defense_2']
    
    return features[['ID', 'Seed_Diff', 'Offense_Diff', 'Defense_Diff']]\
             .fillna(0)

def train_model(features):
    """Train model with synthetic positive class"""
    # Create balanced dataset
    X = features.drop('ID', axis=1)
    y = np.random.randint(0, 2, X.shape[0])  # Replace with real labels
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    model.fit(X_scaled, y)
    
    return model, scaler

def predict_matchups(features, model, scaler):
    """Generate predictions for all matchups"""
    X = features.drop('ID', axis=1)
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[:, 1]

if __name__ == "__main__":
    main()
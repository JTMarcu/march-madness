"""
March Madness 2025 Quick Prediction Script

This script provides a simplified interface for making predictions
once the 2025 tournament brackets are announced.

Usage:
1. Update the BRACKETS dictionary with the actual 2025 tournament seedings
2. Run the script: python quick_predict.py

Requirements:
- Trained model file (march_madness_model.pkl)
- NCAA data files in the data/ directory
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime

# Import the main predictor module
sys.path.append(".")
from march_madness_predictor import (
    load_data, load_model, prepare_tournament_matchups, 
    simulate_tournament, predict_matchup
)

# Configuration
DATA_DIR = "data/"
MODEL_PATH = "models/march_madness_model.pkl"
RESULTS_DIR = "results/"
NUM_SIMULATIONS = 10000  # Higher for more stable predictions

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# ========================================================================
# UPDATE THIS SECTION WHEN 2025 BRACKETS ARE ANNOUNCED
# ========================================================================

# Tournament brackets structure
# Format: Region: [list of team IDs in seed order (1-16)]
# Replace TeamIDs with actual values from MTeams.csv
BRACKETS = {
    'East': [
        1234,  # 1 seed
        1235,  # 16 seed
        1236,  # 8 seed
        1237,  # 9 seed
        1238,  # 5 seed
        1239,  # 12 seed
        1240,  # 4 seed
        1241,  # 13 seed
        1242,  # 6 seed
        1243,  # 11 seed
        1244,  # 3 seed
        1245,  # 14 seed
        1246,  # 7 seed
        1247,  # 10 seed
        1248,  # 2 seed
        1249,  # 15 seed
    ],
    'West': [
        # TeamIDs in seed order (1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15)
    ],
    'South': [
        # TeamIDs in seed order
    ],
    'Midwest': [
        # TeamIDs in seed order
    ]
}

# Final Four matchups
# Which regions play each other in Final Four
FINAL_FOUR_MATCHUPS = [
    ('East', 'West'),
    ('South', 'Midwest')
]

# ========================================================================
# END OF UPDATE SECTION
# ========================================================================

def create_tournament_structure(brackets):
    """
    Convert the brackets dictionary to the format needed for simulation.
    """
    structure = {}
    
    # Create seeds for each region
    for region_code, region_name in zip(['W', 'X', 'Y', 'Z'], brackets.keys()):
        seed_values = []
        for i in range(1, 17):
            seed_values.append((f"{region_code}{i:02d}", brackets[region_name][i-1]))
        
        structure[region_code] = seed_values
    
    return structure


def main():
    """
    Main function to run the prediction workflow.
    """
    print("=" * 80)
    print(f"MARCH MADNESS 2025 PREDICTION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Check if brackets are updated
    if not all(len(teams) == 16 for teams in BRACKETS.values()):
        print("ERROR: Please update the BRACKETS dictionary with the actual 2025 tournament seeds.")
        print("The placeholder values need to be replaced with real team IDs.")
        return
    
    print("Loading data and model...")
    try:
        # Load NCAA data
        data = load_data(DATA_DIR)
        
        # Load trained model
        model_dict = load_model(MODEL_PATH)
        
        # Get team name mapping
        teams_df = data['teams']
        team_name_map = dict(zip(teams_df['TeamID'], teams_df['TeamName']))
        
        print("Preparing tournament matchups...")
        # Prepare all potential matchups for the tournament
        matchups = prepare_tournament_matchups(data, season=2025)
        
        # Create tournament structure from brackets
        tourney_structure = create_tournament_structure(BRACKETS)
        
        print(f"Simulating tournament {NUM_SIMULATIONS} times...")
        # Create a mock teams_df for the simulation
        mock_teams_df = pd.DataFrame()
        mock_teams_df['Seed'] = [seed for region in tourney_structure.keys() for seed, _ in tourney_structure[region]]
        mock_teams_df['TeamID'] = [team_id for region in tourney_structure.keys() for _, team_id in tourney_structure[region]]
        
        # Run tournament simulation
        results = simulate_tournament(model_dict, tourney_structure, matchups, mock_teams_df, NUM_SIMULATIONS)
        
        # Print results
        print("\nTOURNAMENT PREDICTION RESULTS")
        print("=" * 80)
        
        # Convert team IDs to names
        named_results = {}
        for team_id, rounds in results.items():
            team_name = team_name_map.get(team_id, f"Team {team_id}")
            named_results[team_name] = rounds
        
        # Championship probabilities
        print("\nCHAMPIONSHIP PROBABILITIES")
        print("-" * 80)
        champ_probs = sorted([(team, probs['winner']) for team, probs in named_results.items()], 
                            key=lambda x: x[1], reverse=True)
        
        for i, (team, prob) in enumerate(champ_probs[:20], 1):
            print(f"{i:2d}. {team:<30} {prob:.2%}")
        
        # Final Four probabilities
        print("\nFINAL FOUR PROBABILITIES")
        print("-" * 80)
        ff_probs = sorted([(team, probs['final4']) for team, probs in named_results.items()], 
                        key=lambda x: x[1], reverse=True)
        
        for i, (team, prob) in enumerate(ff_probs[:20], 1):
            print(f"{i:2d}. {team:<30} {prob:.2%}")
        
        # Elite Eight probabilities
        print("\nELITE EIGHT PROBABILITIES")
        print("-" * 80)
        e8_probs = sorted([(team, probs['elite8']) for team, probs in named_results.items()], 
                        key=lambda x: x[1], reverse=True)
        
        for i, (team, prob) in enumerate(e8_probs[:20], 1):
            print(f"{i:2d}. {team:<30} {prob:.2%}")
        
        # Sweet Sixteen probabilities
        print("\nSWEET SIXTEEN PROBABILITIES")
        print("-" * 80)
        s16_probs = sorted([(team, probs['sweet16']) for team, probs in named_results.items()], 
                        key=lambda x: x[1], reverse=True)
        
        for i, (team, prob) in enumerate(s16_probs[:20], 1):
            print(f"{i:2d}. {team:<30} {prob:.2%}")
        
        # Save results to CSV
        results_df = pd.DataFrame()
        results_df['Team'] = [team for team, _ in champ_probs]
        results_df['Championship'] = [prob for _, prob in champ_probs]
        results_df['Final Four'] = [named_results[team]['final4'] for team, _ in champ_probs]
        results_df['Elite Eight'] = [named_results[team]['elite8'] for team, _ in champ_probs]
        results_df['Sweet 16'] = [named_results[team]['sweet16'] for team, _ in champ_probs]
        results_df['Round of 32'] = [named_results[team]['round2'] for team, _ in champ_probs]
        results_df['Round of 64'] = [named_results[team]['round1'] for team, _ in champ_probs]
        
        results_file = os.path.join(RESULTS_DIR, f"mm2025_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        results_df.to_csv(results_file, index=False)
        
        print(f"\nDetailed results saved to: {results_file}")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
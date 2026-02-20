# March Madness Predictor

A machine learning model for predicting NCAA March Madness tournament outcomes.

## Overview

This project provides a comprehensive solution for predicting March Madness tournament outcomes using historical NCAA basketball data. The model analyzes team performance metrics, rankings, tournament seedings, and other factors to generate win probabilities for each potential matchup in the tournament.

Key features:
- Data processing pipeline for NCAA basketball data
- Comprehensive feature engineering for team matchups
- Gradient Boosting classification model for predicting game outcomes
- Tournament simulation capabilities
- Detailed probability estimates for each team's advancement through tournament rounds

## How the Model Works

### Data Sources
The model uses several key datasets from NCAA basketball:
- Regular season game results (both compact and detailed)
- NCAA Tournament results
- Team information
- Tournament seeds
- Massey Ordinals (team rankings from various systems)
- Conference information

### Feature Engineering
For each potential matchup, the model calculates:
1. **Basic team statistics**:
   - Win percentage
   - Scoring metrics (points scored/allowed, margin)
   - Game location performance

2. **Advanced metrics**:
   - Shooting percentages (overall, 3-point, free throws)
   - Rebounding statistics
   - Ball control (assists, turnovers, steals)
   - Offensive and defensive efficiency

3. **Contextual features**:
   - Seed differences and interactions
   - Ranking differences from multiple systems
   - Conference strength indicators

4. **Historical performance factors**:
   - Tournament experience
   - Performance against similarly ranked teams

### Model Training
The model is trained on historical tournament data from 2010-2024, using games where we know the actual outcomes. A gradient boosting classifier is employed to predict the probability of one team winning against another based on their relative metrics.

### Tournament Simulation
To generate final predictions, the model:
1. Creates features for all potential matchups in the current tournament
2. Simulates the entire tournament multiple times (1,000+ simulations)
3. Tracks how often each team advances to each round
4. Calculates probabilities for each team's tournament outcomes

## How to Use

### Prerequisites
- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, pickle

### Data Setup
1. Place all NCAA CSV files in a directory called `data/`
2. Ensure you have the following files:
   - MTeams.csv
   - MRegularSeasonCompactResults.csv
   - MRegularSeasonDetailedResults.csv
   - MNCAATourneyCompactResults.csv
   - MNCAATourneySeeds.csv
   - MMasseyOrdinals.csv
   - MTeamConferences.csv

### Training the Model
```python
from march_madness_predictor import train_and_save_model

# Train model using data from 2010 to 2024
train_and_save_model(
    data_dir="data/",
    model_path="models/march_madness_model.pkl",
    start_season=2010,
    end_season=2024
)
```

### Predicting Tournament Outcomes
```python
from march_madness_predictor import predict_tournament

# Predict outcomes for the 2025 tournament
results = predict_tournament(
    data_dir="data/",
    model_path="models/march_madness_model.pkl",
    season=2025,
    num_simulations=1000
)

# Print championship probabilities
winner_probs = sorted([(team, probs['winner']) for team, probs in results.items()], 
                      key=lambda x: x[1], reverse=True)

print("\nChampionship Probabilities:")
for team, prob in winner_probs[:10]:
    print(f"{team}: {prob:.2%}")
```

## Customization

### Tuning the Model
You can perform hyperparameter tuning to optimize the model:

```python
from march_madness_predictor import load_data, prepare_tournament_data, tune_model, save_model

data = load_data("data/")
tournament_features = prepare_tournament_data(data, 2010, 2024)

# Define custom parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [4, 5, 6],
    'min_samples_split': [2, 5],
    'subsample': [0.8, 0.9, 1.0]
}

# Tune model
tuned_model_dict = tune_model(tournament_features, param_grid=param_grid, cv=5)

# Save tuned model
save_model(tuned_model_dict, "models/tuned_march_madness_model.pkl")
```

### Adding Custom Features
To add custom features, modify the `create_matchup_features` function in the code. For example, to add a feature for recent performance:

```python
# In create_matchup_features function
features['recent_form_diff'] = team_a_stats['last_5_win_pct'] - team_b_stats['last_5_win_pct']
```

## Interpreting Results

The model provides probabilities for each team's advancement to various rounds:
- `round1`: Probability of winning first round game
- `round2`: Probability of reaching the Round of 32
- `sweet16`: Probability of reaching the Sweet 16
- `elite8`: Probability of reaching the Elite 8
- `final4`: Probability of reaching the Final Four
- `championship`: Probability of reaching the championship game
- `winner`: Probability of winning the championship

These probabilities are based on thousands of simulations and account for the complex interactions of the tournament bracket structure.

## Limitations and Considerations

- The model can only use data available prior to the tournament
- Injuries, suspensions, and last-minute changes are not automatically factored in
- The model is probabilistic - even a team with a low championship probability can win
- "Cinderella" runs are inherently difficult to predict
- Home court/fan advantage effects may vary year to year

## Further Enhancements

Potential improvements to consider:
1. Incorporate player-level statistics when available
2. Add recency bias to favor more recent game results
3. Include coaching experience factors
4. Add travel distance/fatigue effects
5. Integrate external factors like injuries and suspensions
6. Incorporate momentum and streak analysis

## License

This project is provided for educational and entertainment purposes. Please use responsibly and check the appropriate regulations regarding sports prediction in your jurisdiction.
